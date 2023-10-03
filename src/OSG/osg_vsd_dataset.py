
import torch
import torch.utils.data
import numpy as np
import re
from torch._six import container_abcs, string_classes, int_classes
import h5py

np_str_obj_array_pattern = re.compile(r'[SaUO]')



class OSG_VSD_DATASET(torch.utils.data.Dataset):
    def __init__(self, path_to_h5, device):
        self.path_to_h5 = path_to_h5
        self.device = device
        self.num_of_h5 = 51


    def __len__ (self):
        return self.num_of_h5

    def __getitem__(self, idx):
        the_file = h5py.File(self.path_to_h5+str(idx)+'.hdf5', 'r')
        return torch.tensor(the_file['x'], dtype=torch.float, device=self.device), torch.tensor(the_file['t'], dtype=torch.float, device=self.device)

def my_collate_old(batch):
    data = [item['x'] for item in batch]
    target = [item['t'] for item in batch]
    return [data, target]


my_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
"dicts or lists; found {}")

def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        max_length = max([x.shape[0] for x in batch])
        numel = max_length*len(batch)
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=-1)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(my_collate_err_msg_format.format(elem.dtype))

            return my_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: my_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]

    raise TypeError(my_collate_err_msg_format.format(elem_type))
