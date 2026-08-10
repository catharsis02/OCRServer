import numpy as np
class ToCHWImage:
    """Reorder the dimensions of the image from HWC to CHW."""
    def __call__(self, imgs):
        return [np.transpose(img,(2, 0, 1)) for img in imgs]