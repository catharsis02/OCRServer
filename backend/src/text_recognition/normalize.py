import numpy as np

class Normalize:
    def __init__(self,
    mean = [0.5,0.5,0.5],
    std = [0.5,0.5,0.5],
    scale = 1.0/255.0,
    rec_image_shape = [3,48,320],
    padding=True
    ):
        self.padding = padding
        self.alpha = np.array([scale/std[i] for i in range(len(std))])
        self.beta = np.array([mean[i]/std[i] for i in range(len(std))])
        self.rec_image_shape = rec_image_shape
    
    def norm_and_pad(self,img):
        
        img = img.astype("float32")
        img = img*self.alpha - self.beta
        img = img.transpose((2, 0, 1))
        if not self.padding:
            return img

        w = max(img.shape[2], 320)

        pad_img = np.zeros((3, 48, w), dtype=np.float32)

        pad_img[:, :, :img.shape[2]] = img

        return pad_img
    
    def __call__(self,imgs):
        return [self.norm_and_pad(img) for img in imgs]

