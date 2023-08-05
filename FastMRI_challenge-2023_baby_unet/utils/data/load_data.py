import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os


class SliceData(Dataset):
    def __init__(self, root1, root2, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.examples = [] 
        
        files_1 = list(Path(root1).iterdir())
        
        for fname in sorted(files_1):
            if str(fname).endswith(".h5"):
                num_slices = self._get_metadata(fname)

                self.examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
            
        if root2 != 'init':
            
            files_2 = list(Path(root2).iterdir())
            
            for fname in files_2:
                if str(fname).endswith(".h5"):
                    num_slices = self._get_metadata(fname)

                    self.examples += [
                        (fname, slice_ind) for slice_ind in range(num_slices)
                    ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice = self.examples[i]
        with h5py.File(fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            if self.forward:
                target = -1
            else:
                target = hf[self.target_key][dataslice]
            attrs = dict(hf.attrs)
        
        """
        returns input, target, maximum, fname, slice by __call__():
        input = to_tensor(input)
        target = to_tensor(target)
        if isforward is true, set target, maximum to -1
        """
        return self.transform(input, target, attrs, fname.name, dataslice) 
    


def create_data_loaders(data_path_1, data_path_2, args, shuffle=True, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root1=data_path_1,
        root2=data_path_2,
        # initialize isforward, max_key_
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
