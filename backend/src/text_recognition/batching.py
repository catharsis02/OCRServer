from numpy.typing import NDArray
import numpy as np
class ToBatch:
    def pad_imgs(self,imgs:list[NDArray]) -> list[NDArray]:
        max_width = max([img.shape[2] for img in imgs])
        padded_imgs = []
        for img in imgs:
            _,h,w = img.shape
            pad_width = max_width - w
            padded_img = np.pad(img,((0,0),(0,0),(0,pad_width)),constant_values=0,mode="constant")
            padded_imgs.append(padded_img)
        return padded_imgs

    def __call__(self,imgs):
        imgs = self.pad_imgs(imgs)
        return np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)