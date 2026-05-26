from utils_NT.file_fd_utils import print_to_log_file, save_pickle, load_pickle, \
    maybe_to_torch, to_cuda, poly_lr, save_json, write_pickle
from utils_NT.loss_funcs import DC_and_CE_loss
from utils_NT.file_fd_utils import no_op, DotDict
from utils_NT.data_aug_utils import get_moreDA_augmentation
import os
from collections import OrderedDict
from typing import List
from typing import Tuple, Union
import numpy as np
from sklearn.model_selection import KFold
from utils_NT.data_utils import load_dataset, DataLoader2D, \
    Convert3DTo2DTransform, Convert2DTo3DTransform, MaskTransform,\
    MoveSegAsOneHotToData, ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform, ConvertSegmentationToRegionsTransform, \
    DownsampleSegForDSTransform2
from scipy.ndimage.filters import gaussian_filter

import torch
from torch import nn
import torch.nn.functional as F
from time import time, gmtime

import sys
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler

import matplotlib
import matplotlib.pyplot as plt

softmax_helper = lambda x: F.softmax(x, 1)
inference_apply_nonlin = lambda x: x  # softmax_helper

# from network_TransUNet.vit_seg_modeling_4c import VisionTransformer as ViT_seg
# from network_TransUNet.vit_seg_modeling_4c import CONFIGS as CONFIGS_ViT_seg
from network_SwinUNet.vision_transformer import SwinUnet
from network_ConvUNeXt.conv_unext import ConvUNeXt
from network_ConvUNeXt.cmed_unext_nt_new import CMedUNextNT
from network_ConvUNeXt.cmed_unext_nt_sym import CMedUNextSymNT
from network_ConvUNeXt.cmed_unext_nt_all import CMedUNextAllNT
from network_ConvUNeXt.cmed_std4_nt import CMedUNextD4NT
from network_GCVUnet.GCVUNet_nt import GCVUnetNT
# from network_ConvUNeXt.MSNet.miccai_msnet import M2SNet
from networks.efficientvit.effvit_unet import EffViTUNet
from networks.UNetPVT.UNet_v2 import UNetV2
# from networks.UMamba.UMambaEnc_2d import UMambaEnc

from utils_NT.data_utils import unpack_dataset

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.channel_selection_transforms import SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessTransform, ContrastAugmentationTransform, \
    GammaTransform
from batchgenerators.transforms.local_transforms import BrightnessGradientAdditiveTransform, LocalGammaTransform
from batchgenerators.transforms.noise_transforms import BlankRectangleTransform, MedianFilterTransform, \
    SharpeningTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import Rot90Transform, TransposeAxesTransform, MirrorTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor, \
    OneOfTransform
from batchgenerators.augmentations.utils import pad_nd_image
from utils_NT.optimizer import Lion


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class TrainerNT:
    def __init__(self, args, training_config, fold, log_file, output_folder=None, dataset_directory=None,
                 batch_dice=True, fp16=False):

        self.fp16 = fp16
        self.amp_grad_scaler = None

        ################# SET THESE IN self.initialize() ###################################
        self.network = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = self.val_gen = None
        self.was_initialized = False

        ################# SET THESE IN INIT ################################################
        # self.loss = None

        ################# SET THESE IN LOAD_DATASET OR DO_SPLIT ############################
        self.dataset = args.train_data  # these can be None for inference mode
        self.dataset_tr = self.dataset_val = None  # do not need to be used, they just appear if you are using the suggested load_dataset_and_do_split

        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.max_num_epochs = 1000
        self.num_batches_per_epoch = 250
        self.num_val_batches_per_epoch = 50
        self.also_val_in_tr_mode = False
        self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold

        ################# LEAVE THESE ALONE ################################################
        self.val_eval_criterion_MA = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []  # does not have to be used
        self.epoch = 0
        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        ################# Settings for saving checkpoints ##################################
        self.save_every = 1
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

        #### from nnUNetTrainer ####
        self.init_args = (args, training_config, fold, log_file, output_folder, dataset_directory, batch_dice,
                          fp16)
        # self.experiment_name = args.exp_name
        self.output_folder = output_folder
        self.output_folder_base = self.output_folder
        self.dataset_directory = dataset_directory
        self.fold = fold

        # if we are running inference only then the self.dataset_directory is set (due to checkpoint loading) but it
        # irrelevant
        if self.dataset_directory is not None and os.path.isdir(self.dataset_directory):
            self.gt_niftis_folder = os.path.join(self.dataset_directory, "gt_segmentations")
        else:
            self.gt_niftis_folder = None

        self.folder_with_preprocessed_data = None

        self.dl_tr = self.dl_val = None

        self.num_input_channels = self.num_classes = self.patch_size = self.batch_size = \
            self.threeD = self.intensity_properties = self.normalization_schemes = \
            None  # loaded automatically from plans_file
        self.basic_generator_patch_size = self.data_aug_params = \
            self.transpose_forward = self.transpose_backward = None

        self.args = args  # added NT
        self.training_config = training_config  # added NT

        self.log_file = log_file

        self.batch_dice = batch_dice
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        self.classes = self.do_dummy_2D_aug = self.use_mask_for_norm = self.only_keep_largest_connected_component = \
            self.min_region_size_per_class = self.min_size_per_class = None

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {'constant_values': 0}

        self.pad_all_sides = None

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30

        if 'oversample_fg' in self.training_config:
            self.oversample_foreground_percent = self.training_config['oversample_fg']
        else:
            self.oversample_foreground_percent = 0.33

        self.conv_per_stage = None
        self.regions_class_order = None
        #### from nnUNetTrainer ####

        # from nnUNetTrainerV2 #
        self.max_num_epochs = training_config['max_epochs']
        self.initial_lr = training_config['base_lr']
        self.weight_decay = training_config['weight_decay']
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        # from nnUNetTrainerV2 #

        self.deterministic = training_config['deterministic']

        # from nnUNetTrainerV2_DA5 #
        self.do_mirroring = True
        self.mirror_axes = None
        self.num_proc_DA = 12
        self.num_cached = 4
        self.regions_class_order = self.regions = None
        self.threeD = False

    def initialize(self, training=True):
        """
        Initialize all training parameters
        Args:
            training ():
            force_load_plans ():

        Returns:

        """

        os.makedirs(self.output_folder, exist_ok=True)

        # initialize basic parameters
        self.batch_size = self.training_config['batch_size']
        self.patch_size = np.array(self.training_config['patch_size']).astype(int)
        self.do_dummy_2D_aug = self.training_config['do_dummy_2D_data_aug']
        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = None
        self.normalization_schemes = OrderedDict()
        self.use_mask_for_norm = OrderedDict()
        for i in range(len(self.training_config['contrast_list'])):
            self.normalization_schemes[i] = 'nonCT'
            self.use_mask_for_norm[i] = True
        self.num_input_channels = len(self.training_config['contrast_list'])
        self.num_classes = self.training_config['num_classes']  # including the background
        self.classes = self.training_config['pdt_classes']
        self.only_keep_largest_connected_component = None
        self.min_region_size_per_class = None
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']
        self.transpose_forward = [0, 1, 2]
        self.transpose_backward = [0, 1, 2]

        if 'oversample_fg' in self.training_config:
            self.oversample_foreground_percent = self.training_config['oversample_fg']
        else:
            self.oversample_foreground_percent = 0.33

        self.deep_supervision_scales = None
        self.pin_memory = True
        self.initial_lr = self.training_config['base_lr']
        self.weight_decay = self.training_config['weight_decay']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("Invalid patch size: %s" % str(self.patch_size))

        # setup augmentation parameters
        self.setup_DA_params()

        self.folder_with_preprocessed_data = os.path.join(self.dataset_directory,
                                                          f'preprocessed_data_{self.args.data_subset}',
                                                          self.args.train_data,
                                                          'preprocessed_results')

        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()

            tr_transforms = self.get_train_transforms()
            val_transforms = self.get_val_transforms()
            self.tr_gen, self.val_gen = self.wrap_transforms(self.dl_tr, self.dl_val, tr_transforms, val_transforms)

        else:
            pass

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.was_initialized = True

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = os.path.join(self.folder_with_preprocessed_data, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not os.path.isfile(splits_file):
                print_to_log_file(self.log_file, "Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                print_to_log_file(self.log_file, "Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                print_to_log_file(self.log_file, "The split file contains %d splits." % len(splits))

            print_to_log_file(self.log_file, "Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                print_to_log_file(self.log_file, "This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                print_to_log_file(self.log_file, "INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                print_to_log_file(self.log_file, "This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                             oversample_foreground_percent=self.oversample_foreground_percent,
                             pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                              oversample_foreground_percent=self.oversample_foreground_percent,
                              pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val

    def setup_DA_params(self):
        self.data_aug_params = dict()
        self.data_aug_params['scale_range'] = (0.7, 1.43)

        # we need this because this is adapted in the cascade
        self.data_aug_params['selected_seg_channels'] = None
        self.data_aug_params["move_last_seg_chanel_to_data"] = False

        if self.threeD:
            if self.do_mirroring:
                self.mirror_axes = (0, 1, 2)
                self.data_aug_params['do_mirror'] = True  # needed for inference
                self.data_aug_params['mirror_axes'] = (0, 1, 2)  # needed for inference
            else:
                self.data_aug_params['mirror_axes'] = tuple()
                self.data_aug_params['do_mirror'] = False

            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

            if self.do_dummy_2D_aug:
                print_to_log_file(self.log_file, "Using dummy2d data augmentation")
                self.data_aug_params["dummy_2D"] = True
                self.data_aug_params["rotation_x"] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        else:
            if self.do_mirroring:
                self.mirror_axes = (0, 1)
                self.data_aug_params['mirror_axes'] = (0, 1)  # needed for inference
                self.data_aug_params['do_mirror'] = True  # needed for inference
            else:
                self.data_aug_params['mirror_axes'] = tuple()
                self.data_aug_params['do_mirror'] = False  # needed for inference

            self.do_dummy_2D_aug = False

            self.data_aug_params['rotation_x'] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)

        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
        self.basic_generator_patch_size = self.patch_size
        # if self.do_dummy_2D_aug:
        #     self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
        #                                                      self.data_aug_params['rotation_x'],
        #                                                      self.data_aug_params['rotation_y'],
        #                                                      self.data_aug_params['rotation_z'],
        #                                                      self.data_aug_params['scale_range'])
        #     self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        # else:
        #     self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
        #                                                      self.data_aug_params['rotation_y'],
        #                                                      self.data_aug_params['rotation_z'],
        #                                                      self.data_aug_params['scale_range'])

    def get_train_transforms(self) -> List[AbstractTransform]:
        # used for transpost and rot90
        matching_axes = np.array([sum([i == j for j in self.patch_size]) for i in self.patch_size])
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])

        tr_transforms = []

        if self.data_aug_params['selected_seg_channels'] is not None:
            tr_transforms.append(SegChannelSelectionTransform(self.data_aug_params['selected_seg_channels']))

        if self.do_dummy_2D_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = self.patch_size[1:]
        else:
            patch_size_spatial = self.patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=self.data_aug_params["rotation_x"],
                angle_y=self.data_aug_params["rotation_y"],
                angle_z=self.data_aug_params["rotation_z"],
                p_rot_per_axis=0.5,
                do_scale=True,
                scale=self.data_aug_params['scale_range'],
                border_mode_data="constant",
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=1,
                random_crop=False,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.4,
                independent_scale_for_each_axis=True,
            )
        )

        if self.do_dummy_2D_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if np.any(matching_axes > 1):
            tr_transforms.append(
                Rot90Transform(
                    (0, 1, 2, 3), axes=valid_axes, data_key='data', label_key='seg', p_per_sample=0.5
                ),
            )

        if np.any(matching_axes > 1):
            tr_transforms.append(
                TransposeAxesTransform(valid_axes, data_key='data', label_key='seg', p_per_sample=0.5)
            )

        tr_transforms.append(OneOfTransform([
            MedianFilterTransform(
                (2, 8),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            ),
            GaussianBlurTransform((0.3, 1.5),
                                  different_sigma_per_channel=True,
                                  p_per_sample=0.2,
                                  p_per_channel=0.5)
        ]))

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

        tr_transforms.append(BrightnessTransform(0,
                                                 0.5,
                                                 per_channel=True,
                                                 p_per_sample=0.1,
                                                 p_per_channel=0.5
                                                 )
                             )

        tr_transforms.append(OneOfTransform(
            [
                ContrastAugmentationTransform(
                    contrast_range=(0.5, 2),
                    preserve_range=True,
                    per_channel=True,
                    data_key='data',
                    p_per_sample=0.2,
                    p_per_channel=0.5
                ),
                ContrastAugmentationTransform(
                    contrast_range=(0.5, 2),
                    preserve_range=False,
                    per_channel=True,
                    data_key='data',
                    p_per_sample=0.2,
                    p_per_channel=0.5
                ),
            ]
        ))

        tr_transforms.append(
            SimulateLowResolutionTransform(zoom_range=(0.25, 1),
                                           per_channel=True,
                                           p_per_channel=0.5,
                                           order_downsample=0,
                                           order_upsample=3,
                                           p_per_sample=0.15,
                                           ignore_axes=ignore_axes
                                           )
        )

        tr_transforms.append(
            GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(
            GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))

        if self.do_mirroring:
            tr_transforms.append(MirrorTransform(self.mirror_axes))

        tr_transforms.append(
            BlankRectangleTransform([[max(1, p // 10), p // 3] for p in self.patch_size],
                                    rectangle_value=np.mean,
                                    num_rectangles=(1, 5),
                                    force_square=False,
                                    p_per_sample=0.4,
                                    p_per_channel=0.5
                                    )
        )

        tr_transforms.append(
            BrightnessGradientAdditiveTransform(
                lambda x, y: np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y]))),
                (-0.5, 1.5),
                max_strength=lambda x, y: np.random.uniform(-5, -1) if np.random.uniform() < 0.5 else np.random.uniform(1, 5),
                mean_centered=False,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5
            )
        )

        tr_transforms.append(
            LocalGammaTransform(
                lambda x, y: np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y]))),
                (-0.5, 1.5),
                lambda: np.random.uniform(0.01, 0.8) if np.random.uniform() < 0.5 else np.random.uniform(1.5, 4),
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5
            )
        )

        tr_transforms.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )

        if any(self.use_mask_for_norm.values()):
            tr_transforms.append(MaskTransform(self.use_mask_for_norm, mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if self.data_aug_params["move_last_seg_chanel_to_data"]:
            all_class_labels = np.arange(1, self.num_classes)
            tr_transforms.append(MoveSegAsOneHotToData(1, all_class_labels, 'seg', 'data'))
            if self.data_aug_params["cascade_do_cascade_augmentations"]:
                tr_transforms.append(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(all_class_labels), 0)),
                        p_per_sample=0.4,
                        key="data",
                        strel_size=(1, 8),
                        p_per_label=1
                    )
                )

                tr_transforms.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(all_class_labels), 0)),
                        key="data",
                        p_per_sample=0.2,
                        fill_with_other_class_p=0.15,
                        dont_do_if_covers_more_than_X_percent=0
                    )
                )

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if self.regions is not None:
            tr_transforms.append(ConvertSegmentationToRegionsTransform(self.regions, 'target', 'target'))

        if self.deep_supervision_scales is not None:
            tr_transforms.append(
                DownsampleSegForDSTransform2(self.deep_supervision_scales, 0, input_key='target',
                                             output_key='target')
            )

        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        return tr_transforms

    def get_val_transforms(self) -> List[AbstractTransform]:
        val_transforms = list()
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if self.data_aug_params['selected_seg_channels'] is not None:
            val_transforms.append(SegChannelSelectionTransform(self.data_aug_params['selected_seg_channels']))

        if self.data_aug_params["move_last_seg_chanel_to_data"]:
            all_class_labels = np.arange(1, self.num_classes)
            val_transforms.append(MoveSegAsOneHotToData(1, all_class_labels, 'seg', 'data'))
        val_transforms.append(RenameTransform('seg', 'target', True))

        if self.regions is not None:
            val_transforms.append(ConvertSegmentationToRegionsTransform(self.regions, 'target', 'target'))

        if self.deep_supervision_scales is not None:
            val_transforms.append(
                DownsampleSegForDSTransform2(
                    self.deep_supervision_scales, 0, input_key='target',
                    output_key='target')
            )

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        return val_transforms

    def wrap_transforms(self, dataloader_train, dataloader_val, train_transforms, val_transforms):
        tr_gen = NonDetMultiThreadedAugmenter(dataloader_train,
                                              Compose(train_transforms),
                                              self.num_proc_DA,
                                              self.num_cached,
                                              seeds=None,
                                              pin_memory=self.pin_memory)
        val_gen = NonDetMultiThreadedAugmenter(dataloader_val,
                                               Compose(val_transforms),
                                               self.num_proc_DA // 2,
                                               self.num_cached,
                                               seeds=None,
                                               pin_memory=self.pin_memory)
        return tr_gen, val_gen

    def initialize_network(self):
        # if self.args.network == 'TransUNet4c':
        #     config_vit = CONFIGS_ViT_seg[self.training_config['vit_name']]
        #     config_vit.n_classes = self.training_config['num_classes']
        #     config_vit.n_skip = self.training_config['n_skip']
        #     if self.training_config['vit_name'].find('R50') != -1:
        #         config_vit.patches.grid = \
        #             (int(self.training_config['img_size'] / self.training_config['vit_patches_size']),
        #              int(self.training_config['img_size'] / self.training_config['vit_patches_size']))

        #     self.network = ViT_seg(config_vit,
        #                            img_size=self.training_config['img_size'],
        #                            num_classes=config_vit.n_classes)
        # elif self.args.network == 'TransUNet3c':
        #     config_vit = CONFIGS_ViT_seg_3c[self.training_config['vit_name']]
        #     config_vit.n_classes = self.training_config['num_classes']
        #     config_vit.n_skip = self.training_config['n_skip']
        #     if self.training_config['vit_name'].find('R50') != -1:
        #         config_vit.patches.grid = \
        #             (int(self.training_config['img_size'] / self.training_config['vit_patches_size']),
        #              int(self.training_config['img_size'] / self.training_config['vit_patches_size']))
        #
        #     self.network = ViT_seg_3c(config_vit,
        #                            img_size=self.training_config['img_size'],
        #                            num_classes=config_vit.n_classes)
        #     self.network.load_from(weights=np.load(config_vit.pretrained_path))

        if self.args.network == 'SwinUNet':
            swin_unet_config = DotDict(self.args.config['MODEL'])
            self.network = SwinUnet(swin_unet_config, img_size=self.training_config['img_size'],
                                    num_classes=self.training_config['num_classes'])
        elif self.args.network == 'ConvUNeXt':
            self.network = ConvUNeXt(in_channels=self.training_config['in_channels'],
                                     num_classes=self.training_config['num_classes'],
                                     base_c=self.training_config['num_features'])
        elif self.args.network == 'CMedUNext':
            self.network = CMedUNextNT(in_chans=self.training_config['in_channels'],
                                       out_chans=self.training_config['num_classes'],
                                       depths=self.training_config['depths'],
                                       feat_size=self.training_config['feat_size'],
                                       hidden_size=self.training_config['hidden_size'])
        elif self.args.network == 'MedNext':
            from networks.MedNext.mednext_nt import MedNeXt
            self.network = MedNeXt(
                in_channels=self.training_config['in_channels'],
                n_channels=self.training_config['n_channels'],
                n_classes=self.training_config['num_classes'],
                exp_r=self.training_config['exp_r'],      # Expansion ratio as in Swin Transformers
                # exp_r = 2,
                kernel_size=self.training_config['kernel_size'],        # Can test kernel_size
                deep_supervision=False,             # Can be used to test deep supervision
                do_res=True,                      # Can be used to individually test residual connection
                do_res_up_down = True,
                # block_counts = [2,2,2,2,2,2,2,2,2],
                block_counts=self.training_config['block_counts'], 
                checkpoint_style=None,
                dim='2d',
                grn=True
                )
        elif self.args.network == 'MambaOutSym':
            from networks.MambaOut.mambaout_unet_nt import MambaOutSymNT
            if 'drop_path_rate' not in self.training_config.keys():
                self.training_config['drop_path_rate'] = 0.0
            self.network = MambaOutSymNT(in_chans=self.training_config['in_channels'],
                                         out_chans=self.training_config['num_classes'],
                                         depths=self.training_config['depths'],
                                         feat_size=self.training_config['feat_size'],
                                         hidden_size=self.training_config['hidden_size'],
                                         drop_path_rate=self.training_config['drop_path_rate'])
        elif self.args.network == 'CMedUNextSym':
            if 'drop_path_rate' not in self.training_config.keys():
                self.training_config['drop_path_rate'] = 0.0
            self.network = CMedUNextSymNT(in_chans=self.training_config['in_channels'],
                                          out_chans=self.training_config['num_classes'],
                                          depths=self.training_config['depths'],
                                          feat_size=self.training_config['feat_size'],
                                          hidden_size=self.training_config['hidden_size'],
                                          drop_path_rate=self.training_config['drop_path_rate'])
        elif self.args.network == 'CMedUNextSymRes':
            from network_ConvUNeXt.cmed_unext_nt_sym_features import CMedUNextSymResNT
            if 'drop_path_rate' not in self.training_config.keys():
                self.training_config['drop_path_rate'] = 0.0
            self.network = CMedUNextSymResNT(in_chans=self.training_config['in_channels'],
                                             out_chans=self.training_config['num_classes'],
                                             depths=self.training_config['depths'],
                                             feat_size=self.training_config['feat_size'],
                                             hidden_size=self.training_config['hidden_size'],
                                             drop_path_rate=self.training_config['drop_path_rate'],
                                             depthwise_kernel_size=self.training_config['depthwise_kernel_size'],
                                             in_dim=len(self.training_config['patch_size']))
        elif self.args.network == 'CMedUNextStemNoResSym':
            from network_ConvUNeXt.cmedunext_sym_features_v081824 import CMedUNextStemNoResSymOutFNT
            if 'drop_path_rate' not in self.training_config.keys():
                self.training_config['drop_path_rate'] = 0.0
            self.network = CMedUNextStemNoResSymOutFNT(in_chans=self.training_config['in_channels'],
                                                       out_chans=self.training_config['num_classes'],
                                                       depths=self.training_config['depths'],
                                                       feat_size=self.training_config['feat_size'],
                                                       drop_path_rate=self.training_config['drop_path_rate'],
                                                       depthwise_kernel_size=self.training_config['depthwise_kernel_size'],
                                                       in_dim=len(self.training_config['patch_size']))
        elif self.args.network == 'CMedUNextStemResDecSym':
            from network_ConvUNeXt.unext_sym_resdec_features_v082924 import CMedUNextStemResDecSymOutFNT
            if 'drop_path_rate' not in self.training_config.keys():
                self.training_config['drop_path_rate'] = 0.0
            self.network = CMedUNextStemResDecSymOutFNT(in_chans=self.training_config['in_channels'],
                                                        out_chans=self.training_config['num_classes'],
                                                        depths=self.training_config['depths'],
                                                        feat_size=self.training_config['feat_size'],
                                                        drop_path_rate=self.training_config['drop_path_rate'],
                                                        depthwise_kernel_size=self.training_config['depthwise_kernel_size'],
                                                        in_dim=len(self.training_config['patch_size']))
        elif self.args.network == 'CMedUNextStemResSym':
            from network_ConvUNeXt.unext_sym_stemres_features_v110724 import CMedUNextStemResSymOutFNT
            if 'drop_path_rate' not in self.training_config.keys():
                self.training_config['drop_path_rate'] = 0.0
            self.network = CMedUNextStemResSymOutFNT(in_chans=self.training_config['in_channels'],
                                                     out_chans=self.training_config['num_classes'],
                                                     depths=self.training_config['depths'],
                                                     feat_size=self.training_config['feat_size'],
                                                     drop_path_rate=self.training_config['drop_path_rate'],
                                                     depthwise_kernel_size=self.training_config['depthwise_kernel_size'],
                                                     in_dim=len(self.training_config['patch_size']))
        elif self.args.network == 'CMedUNextAll':
            self.network = CMedUNextAllNT(in_chans=self.training_config['in_channels'],
                                          out_chans=self.training_config['num_classes'],
                                          depths=self.training_config['depths'],
                                          feat_size=self.training_config['feat_size'],
                                          hidden_size=self.training_config['hidden_size'])
        elif self.args.network == 'CMedUNextB3':
            self.network = CMedUNextNT(in_chans=self.training_config['in_channels'],
                                       out_chans=self.training_config['num_classes'],
                                       depths=self.training_config['depths'],
                                       feat_size=self.training_config['feat_size'],
                                       hidden_size=self.training_config['hidden_size'])
        elif self.args.network == 'CMedUNextB3-36':
            self.network = CMedUNextNT(in_chans=self.training_config['in_channels'],
                                       out_chans=self.training_config['num_classes'],
                                       depths=self.training_config['depths'],
                                       feat_size=self.training_config['feat_size'],
                                       hidden_size=self.training_config['hidden_size'])
        elif self.args.network == 'CMedUNextD4':
            self.network = CMedUNextD4NT(in_chans=self.training_config['in_channels'],
                                         out_chans=self.training_config['num_classes'],
                                         depths=self.training_config['depths'],
                                         feat_size=self.training_config['feat_size'],
                                         hidden_size=self.training_config['hidden_size'])
        elif self.args.network == 'GCVUnet':
            self.network = GCVUnetNT(in_chans=self.training_config['in_channels'],
                                     out_chans=self.training_config['num_classes'],
                                     depths=self.training_config['depths'],
                                     num_heads=self.training_config['num_heads'],
                                     window_size=self.training_config['window_size'],
                                     dim=self.training_config['dim'],
                                     mlp_ratio=self.training_config['mlp_ratio'],
                                     drop_path_rate=self.training_config['drop_path_rate'])
        # elif self.args.network == 'M2SNet':
        #     self.network = M2SNet()
        elif self.args.network == 'EffViTUNet':
            self.network = EffViTUNet(in_chans=self.training_config['in_channels'],
                                      out_chans=self.training_config['num_classes'],
                                      depths=self.training_config['depths'],
                                      feat_size=self.training_config['feat_size'],
                                      hidden_size=self.training_config['hidden_size'])
        elif self.args.network == 'UNetV2':
            self.network = UNetV2(n_classes=self.training_config['num_classes'],
                                  deep_supervision=False, pretrained_path=None)
        elif self.args.network == 'BMamba':
            from networks.UMamba.BMamba_2d import BMamba
            from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
            from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
            
            conv_kernel_sizes = self.training_config['conv_kernel_sizes']
            UNet_base_num_features = self.training_config['UNet_base_num_features']
            unet_max_num_features = self.training_config['unet_max_num_features']
            num_stages = len(conv_kernel_sizes)
            dim = len(conv_kernel_sizes[0])
            conv_op = convert_dim_to_conv_op(dim)
            norm_op = get_matching_instancenorm(conv_op)
            self.network = BMamba(
                input_channels=self.training_config['in_channels'],
                num_classes=self.training_config['num_classes'],
                n_stages=num_stages,
                features_per_stage=[min(UNet_base_num_features * 2 ** i, unet_max_num_features) for i in range(num_stages)],
                conv_op=convert_dim_to_conv_op(dim),
                kernel_sizes=conv_kernel_sizes,
                strides=self.training_config['pool_op_kernel_sizes'],
                n_conv_per_stage=self.training_config['n_conv_per_stage_encoder'],
                n_conv_per_stage_decoder=self.training_config['n_conv_per_stage_decoder'],
                conv_bias=self.training_config['conv_bias'],
                norm_op=norm_op,
                norm_op_kwargs={'eps': 1e-5, 'affine': True},
                dropout_op=None, 
                dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'inplace': True},
                deep_supervision=False,
                stem_channels=None)
        elif self.args.network == 'ResMamba':
            from networks.Mamba.ResMamba import ResMamba
            from utils_NT.dynamic_network_blocks import convert_conv_op_to_dim
            from utils_NT.dynamic_network_blocks import get_matching_instancenorm, convert_dim_to_conv_op
            
            conv_kernel_sizes = self.training_config['conv_kernel_sizes']
            UNet_base_num_features = self.training_config['UNet_base_num_features']
            unet_max_num_features = self.training_config['unet_max_num_features']
            num_stages = len(conv_kernel_sizes)
            dim = len(conv_kernel_sizes[0])
            conv_op = convert_dim_to_conv_op(dim)
            norm_op = get_matching_instancenorm(conv_op)
            if 'mamba_ver' not in self.training_config.keys():
                self.training_config['mamba_ver'] = 'v1'
            if 'n_block_per_stage_encoder' not in self.training_config.keys():
                self.training_config['n_block_per_stage_encoder'] = None
            self.network = ResMamba(
                input_channels=self.training_config['in_channels'],
                num_classes=self.training_config['num_classes'],
                n_stages=num_stages,
                features_per_stage=[min(UNet_base_num_features * 2 ** i, unet_max_num_features) for i in range(num_stages)],
                conv_op=convert_dim_to_conv_op(dim),
                kernel_sizes=conv_kernel_sizes,
                strides=self.training_config['pool_op_kernel_sizes'],
                n_enc_conv_per_stage=self.training_config['n_conv_per_stage_encoder'],
                n_dec_conv_per_stage=self.training_config['n_conv_per_stage_decoder'],
                conv_bias=self.training_config['conv_bias'],
                norm_op=norm_op,
                norm_op_kwargs={'eps': 1e-5, 'affine': True},
                dropout_op=None, 
                dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'inplace': True},
                deep_supervision=False,
                stem_channels=None,
                n_enc_blocks_per_stage=self.training_config['n_block_per_stage_encoder'],
                mamba_ver=self.training_config['mamba_ver'])
        elif self.args.network == 'ResNeXtMamba':
            from networks.Mamba.ResNeXtMamba import ResNeXtMamba, LayerNorm
            from utils_NT.dynamic_network_blocks import convert_conv_op_to_dim
            from utils_NT.dynamic_network_blocks import get_matching_instancenorm, convert_dim_to_conv_op
            
            conv_kernel_sizes = self.training_config['conv_kernel_sizes']
            unet_base_num_features = self.training_config['unet_base_num_features']
            unet_max_num_features = self.training_config['unet_max_num_features']
            num_stages = len(conv_kernel_sizes)
            dim = len(conv_kernel_sizes[0])
            conv_op = convert_dim_to_conv_op(dim)
            if 'norm_op_str' not in self.training_config.keys():
                norm_op = get_matching_instancenorm(conv_op)
                norm_op_kwargs={'eps': 1e-5, 'affine': True}
            else:
                if self.training_config['norm_op_str'] == 'instancenorm':
                    norm_op = get_matching_instancenorm(conv_op)
                    norm_op_kwargs={'eps': 1e-5, 'affine': True}
                elif self.training_config['norm_op_str'] == 'layernorm':
                    norm_op = LayerNorm
                    norm_op_kwargs={'eps': 1e-6, 'in_dim': dim, 'data_format': 'channels_first'}
                else:
                    raise NotImplementedError
            if 'mamba_ver' not in self.training_config.keys():
                self.training_config['mamba_ver'] = 'v1'
            if 'n_next_blocks_per_stage' not in self.training_config.keys():
                self.training_config['n_next_blocks_per_stage'] = None
            if 'n_mamba_blocks_per_stage' not in self.training_config.keys():
                self.training_config['n_mamba_blocks_per_stage'] = None
            if 'depthwise_kernel_size' not in self.training_config.keys():
                self.training_config['depthwise_kernel_size'] = 7
            if 'in_dim' not in self.training_config.keys():
                self.training_config['in_dim'] = 2
            if 'mamba_bottleneck' not in self.training_config.keys():
                self.training_config['mamba_bottleneck'] = True
            self.network = ResNeXtMamba(
                input_channels=self.training_config['in_channels'],
                num_classes=self.training_config['num_classes'],
                n_stages=num_stages,
                features_per_stage=[min(unet_base_num_features * 2 ** i, unet_max_num_features) for i in range(num_stages)],
                conv_op=convert_dim_to_conv_op(dim),
                kernel_sizes=conv_kernel_sizes,
                strides=self.training_config['pool_op_kernel_sizes'],
                n_conv_per_stage=self.training_config['n_conv_per_stage_encoder'],
                n_next_blocks_per_stage=self.training_config['n_next_blocks_per_stage'],
                n_mamba_blocks_per_stage=self.training_config['n_mamba_blocks_per_stage'],
                n_conv_per_stage_decoder=self.training_config['n_conv_per_stage_decoder'],
                conv_bias=self.training_config['conv_bias'],
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=None, 
                dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'inplace': True},
                deep_supervision=False,
                stem_channels=None,
                mamba_ver=self.training_config['mamba_ver'],
                depthwise_kernel_size=self.training_config['depthwise_kernel_size'],
                in_dim=self.training_config['in_dim'],
                mamba_bottleneck=self.training_config['mamba_bottleneck'],
                )
        elif self.args.network == 'VMambaUNet':
            from networks.VMamba.vmamba_unet import VMambaUNetNT
            self.network = VMambaUNetNT(
                in_chans=self.training_config['in_channels'],
                out_chans=self.training_config['num_classes'],
                depths=self.training_config['depths'],
                dims=self.training_config['dims'],
                hidden_size = self.training_config['hidden_size'],
                drop_path_rate=self.training_config['drop_path_rate'],
                patch_size=self.training_config['embeded_patch_size'],
                # =========================
                ssm_d_state=self.training_config['ssm_d_state'],
                ssm_ratio=self.training_config['ssm_ratio'],
                ssm_dt_rank=self.training_config['ssm_dt_rank'],
                ssm_act_layer=self.training_config['ssm_act_layer'],
                ssm_conv=self.training_config['ssm_conv'],
                ssm_conv_bias=self.training_config['ssm_conv_bias'],
                ssm_drop_rate=self.training_config['ssm_drop_rate'],
                ssm_init=self.training_config['ssm_init'],
                forward_type=self.training_config['forward_type'],
                # =========================
                mlp_ratio=self.training_config['mlp_ratio'],
                mlp_act_layer=self.training_config['mlp_act_layer'],
                mlp_drop_rate=self.training_config['mlp_drop_rate'],
                gmlp=self.training_config['gmlp'],
                # =========================
                patch_norm=self.training_config['patch_norm'],
                norm_layer=self.training_config['norm_layer'],
                downsample_version = self.training_config['downsample_version'],
                patchembed_version = self.training_config['patchembed_version'],
                )
        # elif self.args.network == 'PVT':
        #     pvt_net = PVTSeg()

        # num_trainable_par = count_parameters(self.network)
        # print_to_log_file(self.log_file, f"\nNumber of trainable parameters: {num_trainable_par}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.network.cuda()

        # print_to_log_file(self.log_file, 'Using torch.compile...')
        # self.network = torch.compile(self.network)              
        # input = torch.randn([110, 4, 224, 224]).to('cuda')
        # out = self.network(input)
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        if 'opt' in self.training_config:
            if self.training_config['opt'] == 'Lion':
                self.optimizer = Lion(self.network.parameters(), self.initial_lr,
                                      weight_decay=self.weight_decay)
            elif self.training_config['opt'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(self.network.parameters(),
                                                   lr=self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
            elif self.training_config['opt'] == 'SGD':
                self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                                 weight_decay=self.weight_decay,
                                                 momentum=0.99, nesterov=True)
        else:
            self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                             weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)

        self.lr_scheduler = None
        # self.optimizer = torch.optim.AdamW(self.network.parameters(),
        #                                    lr=self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        # self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.max_num_epochs)

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()

    def save_config_parameters(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, os.path.join(self.output_folder,
                                    f"config_{self.args.exp_name}_{self.args.log_sdir}.json"))

    def run_training(self):
        # print_to_log_file(self.log_file, 'Start training')
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training

        self.save_config_parameters()

        if not torch.cuda.is_available():
            print_to_log_file(self.log_file,
                              "WARNING!!! You are attempting to run training on a CPU "
                              "(torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        # print_to_log_file(self.log_file, 'Data ready')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()
        while self.epoch < self.max_num_epochs:
            print_to_log_file(self.log_file, "\nEpoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            for i_batch in range(self.num_batches_per_epoch):
                # print_to_log_file(self.log_file, i_batch)
                l = self.run_iteration(self.tr_gen, True)
                # print_to_log_file(self.log_file,
                #                   f"Batch {i_batch}/{self.num_batches_per_epoch}: train loss={l}")
                train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            print_to_log_file(self.log_file, "train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                print_to_log_file(self.log_file, "validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    print_to_log_file(self.log_file, "validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            print_to_log_file(self.log_file, "This epoch took %f s\n" % (epoch_end_time - epoch_start_time))
            epoch_dur = epoch_end_time - epoch_start_time
            estimated_end_time = (self.max_num_epochs - self.epoch + 1) * epoch_dur + epoch_end_time
            from datetime import datetime
            print_to_log_file(self.log_file, f'Estimated completion time: {datetime.fromtimestamp(estimated_end_time)}')
            

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(os.path.join(self.output_folder,
                                                                         "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if os.path.isfile(os.path.join(self.output_folder, "model_latest.model")):
            os.remove(os.path.join(self.output_folder, "model_latest.model"))
        if os.path.isfile(os.path.join(self.output_folder, "model_latest.model.pkl")):
            os.remove(os.path.join(self.output_folder, "model_latest.model.pkl"))

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        # start_t = time()
        data_dict = next(data_generator)
        # end_t = time()
        # print(f'Data loading time = {end_t - start_t}')
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        # start_t = time()
        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        # end_t = time()
        # print(f'Backprop = {end_t - start_t}')

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        print_to_log_file(self.log_file, "Saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        print_to_log_file(self.log_file, "done, saving took %.2f seconds" % (time() - start_time))

        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['training_config'] = self.training_config

        write_pickle(info, fname + ".pkl")

    def on_epoch_end(self):
        self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        _ = self.manage_patience()

        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                print_to_log_file(self.log_file,
                                  "At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                  "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                  "sometimes causes issues such as this one. Momentum has now been reduced to "
                                  "0.95 and network weights have been reinitialized")

        return continue_training

    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        print_to_log_file(self.log_file, "Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class])
        print_to_log_file(self.log_file, "(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.)")

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 36}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss-tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss-val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss-val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax2.set_ylabel("Evaluation Metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(os.path.join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            print_to_log_file(self.log_file, "Failed to plot: ", sys.exc_info())

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1
        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)
        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        print_to_log_file(self.log_file, "lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            print_to_log_file(self.log_file, "Saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(os.path.join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(os.path.join(self.output_folder, "model_latest.model"))
            print_to_log_file(self.log_file, "done")

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        :return:
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if len(self.all_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                is better, so we need to negate it.
                """
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_eval_metrics[-1]

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            #self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            #self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                #self.print_to_log_file("saving best epoch checkpoint...")
                if self.save_best_checkpoint: self.save_checkpoint(os.path.join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                #self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                pass
                #self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    #self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    #self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                pass
                #self.print_to_log_file(
                #    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def load_latest_checkpoint(self, train=True):
        if os.path.isfile(os.path.join(self.output_folder, "model_final_checkpoint.model")):
            return self.load_checkpoint(os.path.join(self.output_folder, "model_final_checkpoint.model"), train=train)
        if os.path.isfile(os.path.join(self.output_folder, "model_latest.model")):
            return self.load_checkpoint(os.path.join(self.output_folder, "model_latest.model"), train=train)
        if os.path.isfile(os.path.join(self.output_folder, "model_best.model")):
            return self.load_best_checkpoint(train)
        raise RuntimeError("No checkpoint found")

    def load_checkpoint(self, fname, train=True):
        print_to_log_file(self.log_file, "loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train)

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if train:
                if 'amp_grad_scaler' in checkpoint.keys():
                    self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = \
            checkpoint['plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, \
            self.best_val_eval_criterion_MA = checkpoint['best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            print_to_log_file(self.log_file, "WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). "
                                             "This is due to an old bug and should only appear when you are loading "
                                             "old models. New models should have this fixed! self.epoch is now set"
                                             " to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()

    def load_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if os.path.isfile(os.path.join(self.output_folder, "model_best.model")):
            self.load_checkpoint(os.path.join(self.output_folder, "model_best.model"), train=train)
        else:
            print_to_log_file(self.log_file, "WARNING! model_best.model does not exist! "
                                             "Cannot load best checkpoint. Falling "
                                             "back to load_latest_checkpoint")
            self.load_latest_checkpoint(train)

    def predict_preprocessed_data_return_features(self, data: np.ndarray, do_mirroring: bool = True,
                                                  mirror_axes: Tuple[int] = None,
                                                  use_sliding_window: bool = True, step_size: float = 0.5,
                                                  use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                  pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                  verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        # ds = self.network.do_ds
        # self.network.do_ds = False
        self.network._gaussian_3d = self.network._patch_size_for_gaussian_3d = None
        self.network._gaussian_2d = self.network._patch_size_for_gaussian_2d = None

        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        current_mode = self.network.training
        self.network.eval()
        ret = self.predict_patient(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                   use_sliding_window=use_sliding_window, step_size=step_size,
                                   patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                   use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                   pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                   mixed_precision=mixed_precision)
        self.network.train(current_mode)
        return ret
    
    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        # ds = self.network.do_ds
        # self.network.do_ds = False
        self.network._gaussian_3d = self.network._patch_size_for_gaussian_3d = None
        self.network._gaussian_2d = self.network._patch_size_for_gaussian_2d = None

        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        current_mode = self.network.training
        self.network.eval()
        ret = self.predict_patient(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                   use_sliding_window=use_sliding_window, step_size=step_size,
                                   patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                   use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                   pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                   mixed_precision=mixed_precision)
        self.network.train(current_mode)
        return ret

    def preprocess_patient(self, input_files):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_files:
        :return:
        """
        from preprocessing_data.preprocessing import Preprocessor
        from preprocessing_data.cropping import ImageCropper

        preprocessor = Preprocessor(self.normalization_schemes, self.use_mask_for_norm,
                                    self.transpose_forward, self.intensity_properties)

        data, seg, properties = ImageCropper.crop_from_list_of_files(input_files, seg_file=None)

        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        data, seg, properties = preprocessor.resample_and_normalize(data, properties["original_spacing"],
                                                                    properties, seg)

        return data.astype(np.float32), seg, properties
    
    def preprocess_patient_skull(self, input_files):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_files:
        :return:
        """
        from preprocessing_data.preprocessing_skull import Preprocessor
        from preprocessing_data.cropping_skull import ImageCropper

        preprocessor = Preprocessor(self.normalization_schemes, self.use_mask_for_norm,
                                    self.transpose_forward, self.intensity_properties)

        data, seg, properties = ImageCropper.crop_from_list_of_files(input_files, seg_file=None)

        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        data, seg, properties = preprocessor.resample_and_normalize(data, properties["original_spacing"],
                                                                    properties, seg)

        return data.astype(np.float32), seg, properties

    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map

    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...],
                                          image_size: Tuple[int, ...],
                                          step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in
                     zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps

    def predict_patient(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                        use_sliding_window: bool = False,
                        step_size: float = 0.5, patch_size: Tuple[int, ...] = None,
                        regions_class_order: Tuple[int, ...] = None,
                        use_gaussian: bool = False, pad_border_mode: str = "constant",
                        pad_kwargs: dict = None, all_in_gpu: bool = False,
                        verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y, z).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :param mixed_precision: if True, will run inference in mixed precision with autocast()
        :return:
        """
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if self.network.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if use_sliding_window:
                    res = self._internal_predict_3D_2DViT_tiled(x, self.patch_size, do_mirroring, mirror_axes,
                                                                step_size,
                                                                regions_class_order, use_gaussian,
                                                                pad_border_mode,
                                                                pad_kwargs, all_in_gpu, False)
                else:
                    res = self._internal_predict_3D_2DViT(x, self.patch_size, do_mirroring, mirror_axes,
                                                               regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)

        return res

    def _internal_maybe_mirror_and_pred_2D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: Union[np.ndarray, torch.tensor] = None) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(x.shape) == 4, 'x must be (b, c, x, y)'

        x = maybe_to_torch(x)
        result_torch = torch.zeros([x.shape[0], self.num_classes] + list(x.shape[2:]), dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x)
            result_torch = result_torch.cuda()

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult)

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = inference_apply_nonlin(self.network(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                pred = inference_apply_nonlin(self.network(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                pred = inference_apply_nonlin(self.network(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = inference_apply_nonlin(self.network(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_predict_2D_2DViT_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 3, "x must be (c, x, y)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self.network._gaussian_2d is None or not all(
                    [i == j for i, j in zip(patch_size, self.network._patch_size_for_gaussian_2d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self.network._gaussian_2d = gaussian_importance_map
                self.network._patch_size_for_gaussian_2d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self.network._gaussian_2d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(0, non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device='cuda:0')

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device='cuda:0')

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(0, non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device='cuda:0')
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self.network._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]

                predicted_patch = self._internal_maybe_mirror_and_pred_2D(
                    data[None, :, lb_x:ub_x, lb_y:ub_y], mirror_axes, do_mirroring,
                    gaussian_importance_map)[0]

                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities

    def _internal_predict_3D_2DViT_tiled(self, x: np.ndarray, patch_size: Tuple[int, int], do_mirroring: bool,
                                          mirror_axes: tuple = (0, 1), step_size: float = 0.5,
                                          regions_class_order: tuple = None, use_gaussian: bool = False,
                                          pad_border_mode: str = "edge", pad_kwargs: dict =None,
                                          all_in_gpu: bool = False,
                                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError

        assert len(x.shape) == 4, "data must be c, x, y, z"

        predicted_segmentation = []
        softmax_pred = []

        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2DViT_tiled(
                x[:, s], step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred

    def _internal_predict_3D_2DViT(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    all_in_gpu: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, "data must be c, x, y, z"
        predicted_segmentation = []
        softmax_pred = []
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2DViT(
                x[:, s], min_size, do_mirroring, mirror_axes, regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred

    def _internal_predict_2D_2DViT(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 3, "x must be (c, x, y)"

        if verbose: print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_2D(data[None], mirror_axes, do_mirroring,
                                                                          None)[0]

        slicer = tuple(
            [slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) -
                                                                       (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities
