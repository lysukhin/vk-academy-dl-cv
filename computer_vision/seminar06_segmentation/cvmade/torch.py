"""Extra tools for PyTorch."""
import torch


class CyclicDataLoader(object):
    """Infinite cyclic dataloader."""
    def __init__(self, dataset, *args, **kwargs):
        if len(dataset) == 0:
            raise ValueError("Empty dataset")
        self._dataset = dataset
        self._dataloader_args = args
        self._dataloader_kwargs = kwargs
        self._dataloader = self._make_dataloader()
        self._iterator = iter(self._dataloader)
        self._step = 0
        
    @property
    def step(self):
        return self._step
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self._step += 1
        try:
            return next(self._iterator)
        except StopIteration:
            self._dataloader = self._make_dataloader()
            self._iterator = iter(self._dataloader)
            return next(self._iterator)
        
    def _make_dataloader(self):
        return torch.utils.data.DataLoader(self._dataset, *self._dataloader_args, **self._dataloader_kwargs)
