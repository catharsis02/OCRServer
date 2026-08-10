import cv2
import math
from .read_image import validate_image

class Resize:
    def __init__(self,rec_image_shape : list[int] =[3,48,320],input_shape : list[int] | None = None):
        self.rec_image_shape = rec_image_shape
        self.input_shape = input_shape
        self.max_rec_Width = 3200
    
    
    def resize_type_0(self,img):
        if img.shape[2] != 3:
            raise ValueError("Image is not RGB")
        max_rec_Width = self.max_rec_Width
        rec_image_shape = self.rec_image_shape
        rec_Channel,rec_Height,rec_Width = rec_image_shape
        max_wh_ratio = float(rec_Width) / rec_Height
        h,w = img.shape[0:2]
        wh_ratio = float(w) / h
        max_wh_ratio = max(wh_ratio,max_wh_ratio)

        rec_Width = int(max_wh_ratio * rec_Height)
        resize_h = rec_Height
        if rec_Width > max_rec_Width:
            resize_w = max_rec_Width
        else:
            ratio = float(w) / h
            if math.ceil(rec_Height * ratio) > self.max_rec_Width:
                resize_w = max_rec_Width
            else:
                resize_w = int(math.ceil(rec_Height * ratio))
        
        return cv2.resize(img,(resize_w,resize_h))

    
    def resize_type_3(self,img):
        input_shape = self.input_shape
        rec_C,rec_H,rec_W = input_shape # type:ignore
        return cv2.resize(img,(rec_W,rec_H))

    def __call__(self,imgs):
        output = []
        input_shape = self.input_shape

        for img in imgs:
            validate_image(img)

            if input_shape is None:
                resized= self.resize_type_0(img)
            else:
                resized= self.resize_type_3(img)
            output.append(resized)

        return output
