import cv2

class NormalizeImage:
    def __init__(self,scale=(1.0/255.0),mean=None,std=None,order="chw"):
        if isinstance(scale, str):
            scale = eval(scale)
        self.order = order

        scale = scale if scale is not None else 1.0/255.0
        mean = mean if mean is not None else [0.485, 0.456, 0.406] # ImageNet images mean
        std = std if std is not None else [0.229, 0.224, 0.225]

        self.alpha = [scale/std[i] for i in range(len(std))]
        self.beta = [mean[i]/std[i] for i in range(len(std))]
    
    def __call__(self,imgs):
        return [self.norm(img) for img in imgs]

    def norm(self,img):
        try:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
        except Exception:
            pass
        r,g,b = cv2.split(img)
        # r,g,b = r.astype(np.float32),g.astype(np.float32),b.astype(np.float32)
        r = r*self.alpha[0] - self.beta[0]
        g = g*self.alpha[1] - self.beta[1]
        b = b*self.alpha[2] - self.beta[2]

        res = cv2.merge((r,g,b))
        
        return res