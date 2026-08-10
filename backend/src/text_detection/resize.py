from math import ceil
import cv2
import numpy as np


class Resize:
    def __init__(self, input_shape=None, max_side_limit=4000, **kwargs):
        self.image_resize_type: int = 0
        self.keep_ratio: bool = False
        if input_shape is not None:
            self.input_shape = input_shape
            self.image_resize_type = 3
        elif "input_shape" in kwargs:
            self.input_shape = kwargs["input_shape"]
            self.image_resize_type = 1
            if "keep_ratio" in kwargs:
                self.keep_ratio = kwargs["keep_ratio"]
        elif "limit_side_len" in kwargs:
            self.limit_side_len = kwargs["limit_side_len"]
            self.limit_type = kwargs.get("limit_type", "min")
        elif "resize_long" in kwargs:
            self.image_resize_type = 2
            self.resize_long = kwargs.get("resize_long",960)
        else:
            self.limit_side_len = 736
            self.limit_type = "min"

        self.max_side_limit = max_side_limit
    
    def __call__(
        self,
        img,
        limit_side_len : int |None = None,
        limit_type : str | None = None,
        max_side_limit: int | None = None):

        limit_side_len = limit_side_len if limit_side_len is not None else getattr(self, "limit_side_len", None)
        limit_type = limit_type if limit_type is not None else getattr(self, "limit_type", None)
        max_side_limit = max_side_limit if max_side_limit is not None else self.max_side_limit

        src_h,src_w,_ = img.shape
        # because checking individually will pad if one side is too small but other is too large
        # after padding both become 32, therefore 32+32 = 64
        if(sum([src_h,src_w]) < 64):
            img = self.padding(img=img)
        if self.image_resize_type == 0:
            img,[ratio_h,ratio_w] = self.resize_of_type0(img,limit_side_len,limit_type,max_side_limit)
        elif self.image_resize_type == 2:
            img,[ratio_h,ratio_w] = self.resize_of_type2(img)
        elif self.image_resize_type == 3:
            img,[ratio_h,ratio_w] = self.resize_of_type3(img)
        else:
            self.image_resize_type == 1
            img,[ratio_h,ratio_w] = self.resize_of_type1(img)
        
        return img,np.array([src_h,src_w,ratio_h,ratio_w])


    def resize_of_type0(self,img,limit_side_len,limit_type,max_side_limit):
        h,w,c = img.shape
        if limit_type=="max":
            if max(h,w) > limit_side_len:
                if h>w:
                    ratio = limit_side_len / h
                else:
                    ratio = limit_side_len / w
            else:
                ratio = 1.0
        elif limit_type=="min":
            if min(h,w) < limit_side_len:
                if h<w:
                    ratio = limit_side_len / h
                else:
                    ratio = limit_side_len / w
            else:
                ratio = 1.0
        elif limit_type == "resize_long":
            ratio = limit_side_len / max(h,w)
        
        resize_h = int(ratio * h)
        resize_w = int(ratio * w)

        if max(resize_h,resize_w) > max_side_limit:
            ratio = float(max_side_limit) / max(resize_h,resize_w)
            resize_h = int(ratio*resize_h)
            resize_w = int(ratio*resize_w)

        resize_h = max(int(round(resize_h/32)*32),32)
        resize_w = max(int(round(resize_w/32)*32),32)
        
        if h == resize_h and w == resize_w:
            return img,[1.0,1.0]
        
        try:
            if int(resize_h)<=0 or int(resize_w)<=0:
                return None,[None,None]
            img = cv2.resize(img, (int(resize_w), int(resize_h)))  
        except Exception as exc:
            print(f"Exception : {exc}")
        
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img,[ratio_h,ratio_w]

    def resize_of_type1(self,img):
        input_h,input_w = self.input_shape
        src_h,src_w = img.shape[0:2]
        keep_ratio = self.keep_ratio
        if keep_ratio:
            ratio = float(input_h) / src_h
            resize_h = input_h
            resize_w = int(src_w * ratio)
        else:
            resize_h = input_h
            resize_w = input_w
        
        resize_h = input_h
        resize_w = int((ceil(resize_w/32)*32))

        ratio_h = float(resize_h) / src_h
        ratio_w = float(resize_w) / src_w

        if resize_h == src_h and resize_w == src_w:
            return img, [1.0, 1.0]
        try:
            if resize_h<=0 or resize_w<=0:
                return None,[None,None]
            img = cv2.resize(img, (int(resize_w), int(resize_h)))  
        except Exception as exc:
            print(f"Exception : {exc}")

        return img,[ratio_h,ratio_w]

    def resize_of_type2(self,img):
        resize_long = self.resize_long
        src_h,src_w = img.shape[0:2]
        resize_w = src_w
        resize_h = src_h

        if resize_h > resize_w:
            ratio = float(resize_long) / resize_h
        else:
            ratio = float(resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        
        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        
        if resize_h == src_h and resize_w == src_w:
            return img, [1.0, 1.0]
        
        try:
            if resize_h<=0 or resize_w<=0:
                return None,[None,None]
            img = cv2.resize(img, (int(resize_w), int(resize_h)))  
        except Exception as exc:
            print(f"Exception : {exc}")
        
        ratio_h = float(resize_h) / float(src_h)
        ratio_w = float(resize_w) / float(src_w)

        return img,[ratio_h,ratio_w]

    def resize_of_type3(self,img):
        resize_h,resize_w = self.input_shape
        src_h,src_w = img.shape[0:2]
        if resize_h == src_h and resize_w == src_w:
            return img, [1.0, 1.0]
        try:
            if resize_w <= 0 or resize_h<=0:
                return None,[None,None]
            img = cv2.resize(img, (int(resize_w), int(resize_h)))        
        except Exception as exc:
            print(f"Exception : {exc}")
        ratio_h = float(resize_h) / float(src_h)
        ratio_w = float(resize_w) / float(src_w)

        return img,[ratio_h,ratio_w]


    def padding(self,img:np.ndarray,value=0):
        # Top Left Padding -> Paddle Models accept top left Padding. Image at top left and rest 0s
        h,w,c = img.shape
        img_pad = np.zeros((max(h,32),max(w,32),c)) + value
        img_pad[:h,:w,:] = img
        return img_pad


