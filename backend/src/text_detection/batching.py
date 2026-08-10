import numpy as np
class ToBatch:
    def __call__(self,imgs):
        return np.stack(imgs,axis=0).astype(np.float32,copy=False)