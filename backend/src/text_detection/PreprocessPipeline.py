from backend.src.text_detection.read_image import ReadImage
from backend.src.text_detection.resize import Resize
from backend.src.text_detection.normalize import NormalizeImage
from backend.src.text_detection.convert_to_chw import ToCHWImage
from backend.src.text_detection.batching import ToBatch

class PreprocessPipeline:
    def __call__(self,img_path):
        batch_raw_imgs = ReadImage(format="BGR")([img_path])
        batch_imgs, batch_shapes = Resize(resize_long=960)(img=batch_raw_imgs)
        batch_imgs = NormalizeImage()([batch_imgs])
        batch_imgs = ToCHWImage()(batch_imgs)
        x = ToBatch()(batch_imgs)

        return x,batch_shapes
