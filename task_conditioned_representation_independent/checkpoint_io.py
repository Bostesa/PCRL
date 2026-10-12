"""Read tensors and state-only reference containers without candidate imports."""
from __future__ import annotations
import pickle

class ReferenceState:
    """Pure attribute container for a stored DiagonalReference pickle."""
    pass

class Unpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='task_conditioned_representation' or module.startswith('task_conditioned_representation.'):
            if module=='task_conditioned_representation.kernels' and name=='DiagonalReference':
                return ReferenceState
            raise ImportError('candidate scientific checkpoint class refused: '+module+'.'+name)
        return super().find_class(module,name)

def load(file,**kwargs):return Unpickler(file,**kwargs).load()

def read(path):
    import sys,torch
    return torch.load(path,map_location='cpu',pickle_module=sys.modules[__name__],weights_only=False)
