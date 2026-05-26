import json
import os
import pickle
import torch
import shutil
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from datetime import datetime
from time import time, sleep
import sys
import SimpleITK as sitk


def read_nii_sitk(filepath):
    img_sitk = sitk.ReadImage(filepath)
    img = sitk.GetArrayFromImage(img_sitk)
    return img


def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def get_subdir_list(folder: str, prefix: str = None, suffix: str = None, sort: bool = True) -> list:
    res = [i for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def get_common_prefixes_with_all_contrasts(directory, contrast_types=['_t1', '_t1c', '_t2', '_flair']):
    import re
    from collections import defaultdict
    # Define patterns for the four contrast
    contrast_map = {c: re.compile(rf'{c}\.nii\.gz$', re.IGNORECASE) for c in contrast_types}
    # Dictionary to collect prefixes and their contrast types
    prefix_contrasts = defaultdict(set)
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz'):
            for contrast, pattern in contrast_map.items():
                if pattern.search(filename):
                    # Remove the contrast portion to get the prefix
                    prefix = pattern.sub('', filename)
                    prefix_contrasts[prefix].add(contrast)
    # Filter prefixes that have all four contrast3
    complete_sets = [prefix for prefix, contrasts in prefix_contrasts.items() if len(contrasts) == 4]
    return complete_sets


def get_file_list(folder: str, prefix: str = None, suffix: str = None, sort: bool = True) -> list:
    res = [os.path.join(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

        
def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def save_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


def print_to_log_file(log_file, *args, also_print_to_console=True, add_timestamp=True):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)

    if add_timestamp:
        args = ("%s:" % dt_object, *args)

    successful = False
    max_attempts = 5
    ctr = 0
    while not successful and ctr < max_attempts:
        try:
            with open(log_file, 'a+') as f:
                for a in args:
                    f.write(str(a))
                    f.write(" ")
                f.write("\n")
            successful = True
        except IOError:
            print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
            sleep(0.5)
            ctr += 1
    if also_print_to_console:
        print(*args)


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


class DotDict(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]

