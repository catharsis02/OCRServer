from numpy.typing import NDArray
import cv2
import numpy as np
class ReadImage:
    def __init__(self,format="BGR"):
        self.format = format
        self.flags = {
            "BGR":cv2.IMREAD_COLOR,
            "RGB":cv2.IMREAD_COLOR,
            "GRAY":cv2.IMREAD_GRAYSCALE
        }[format]
    
    def __call__(self,imgs):
        return self.read([img for img in imgs])  

    def read_file(self,im_path,flags):
        with open(im_path,"rb") as f:
            img_array = np.frombuffer(buffer=f.read(),dtype=np.uint8)
        return cv2.imdecode(buf=img_array,flags=flags)

    def read(self,img):
        img = img[0]
        if(isinstance(img,str)):
            blob = self.read_file(img,self.flags)
            if blob is None:
                raise RuntimeError("Image not found")
            if self.format=="RGB":
                if blob.ndim != 3:
                    raise RuntimeError("Array is not 3-dimensional.")
                blob = cv2.cvtColor(blob,cv2.COLOR_BGR2RGB)
            return blob
        elif(isinstance(img,np.ndarray)):
            if self.format=="RGB":
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return img

        else:
            raise TypeError(
                f"ReadImage only supports the following types:\n"
                f"1. str, indicating a image file path or a directory containing image files.\n"
                f"2. numpy.ndarray.\n"
                f"However, got type: {type(img).__name__}."
            )
        
def validate_image(img:NDArray):
    if img.ndim < 2:
        raise ValueError("Image needs at least two dimensions")
    h,w = img.shape[0:2]
    if h<=0 or w <= 0 or img.size == 0:
        raise ValueError(f"Invalid Image {img}")