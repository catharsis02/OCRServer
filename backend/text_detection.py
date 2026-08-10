from numpy.typing import NDArray
import json
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from tqdm import tqdm
import shutil

from backend.src.pdf_to_image import clean_image_cache, extract_photos_from_pdf
from backend.src.text_detection.dbpostprocess import DBPostProcess
from backend.src.text_detection.PreprocessPipeline import PreprocessPipeline

SAVE_PDF_PHOTOS : Path  = Path.cwd() / "pdf2image"
OUTPUT_SAVE_PATH : Path = Path.cwd() / "det_outputs"
TEMP_SAVE_PHOTOS : Path = Path.cwd() / "temp_images"

class TextDetection:
    """
    Initialize the Text Detection class.
    """
    def __init__(self,
        model_path:str|Path,
        pdf_to_photo : str | Path = SAVE_PDF_PHOTOS,
        output_save_path : str | Path = OUTPUT_SAVE_PATH,
        temp_save_path : str | Path = TEMP_SAVE_PHOTOS,
        cache_dir : str | Path = "./.ov_cache"
):
        self.model_path = model_path
        core = ov.Core()
        core.set_property({"CACHE_DIR":cache_dir})
        model = core.read_model(str(model_path))
        self.compiled_model = ov.compile_model(model,"AUTO")
        self.infer_request = self.compiled_model.create_infer_request()
        self.preprocessor = PreprocessPipeline()
        self.decode = DBPostProcess(score_mode="slow")
        self.pdf_extractor = extract_photos_from_pdf
        self.clean = clean_image_cache
        dummy = ov.Tensor(type=ov.Type.f32,shape=(1,3,1024,768))
        self.infer_request.set_input_tensor(dummy)
        self.pdf_to_photo = Path(pdf_to_photo)
        self.output_save_path = Path(output_save_path)
        self.temp_save_path = Path(temp_save_path)
        self.pdf_to_photo.mkdir(parents=True,exist_ok=True)
        self.output_save_path.mkdir(parents=True,exist_ok=True)
        self.temp_save_path.mkdir(parents=True,exist_ok=True)

    def draw(self,img_path : str | Path, boxes : NDArray, color=(0, 0, 255), thickness=2):
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot detect image at {img_path}")
        img_copy = img.copy()

        for box in boxes:
            pts = np.array(box)
            pts = pts.reshape(-1, 2).astype(np.int32)
            for i in range(len(pts)):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i + 1) % len(pts)])
                cv2.line(
                    img_copy,
                    pt1,
                    pt2,
                    color,
                    thickness,
                    cv2.LINE_AA
                )
        return img_copy

    def _convert_pdf_to_image(self,item: str|Path) -> None:
        item = Path(item)
        pdf_extractor = self.pdf_extractor
        print(item.suffix.lower())
        if str(item.suffix.lower()) == ".pdf":
            return pdf_extractor(pdf_path=item,save_dir=self.pdf_to_photo,temp_save_dir=self.temp_save_path)
        else:
            self.temp_save_path.mkdir(parents=True,exist_ok=True)
            shutil.copy2(item,self.temp_save_path)
            shutil.copy2(item,self.pdf_to_photo)
    
    def _predict_iter(self,images:str|Path) -> tuple[NDArray,list[list[float]]]:
        x,batch_shapes = self.preprocessor(images)
        x = ov.Tensor(x)
        self.infer_request.set_input_tensor(x)
        self.infer_request.infer()
        output = self.infer_request.get_output_tensor()
        output = np.asarray(output.data,dtype=np.float32)
        boxes,scores = self.decode(output,[batch_shapes])
        return boxes,scores


    def predict(self,pdf_path:str|Path):
        start = time.perf_counter()
        print("Text Detection : Finding Text in Images")
        if not isinstance(pdf_path, (str, Path)):
            raise TypeError(
                f"pdf_path must be str or Path, got {type(pdf_path).__name__}"
            )
        if isinstance(pdf_path,str):
            pdf_path = Path(pdf_path)
        if isinstance(pdf_path,Path):
            pdfs = [f for f in pdf_path.iterdir()]
        
        for pdf in tqdm(pdfs):
            self.clean(self.temp_save_path)
            self._convert_pdf_to_image(pdf)
            images = [img for img in self.temp_save_path.iterdir()]
            for img in images:
                boxes,scores = self._predict_iter(img)
                boxes = boxes[0].tolist()
                scores = [float(score) for score in scores[0]]
                
                out_file = self.output_save_path / "json" / f"{img.stem}.json"
                out_file.parent.mkdir(parents=True, exist_ok=True)

                with open(out_file,"w",encoding="utf-8") as f:
                        json.dump(
                        {
                            "image": str(img),
                            "boxes": boxes,
                            "scores": scores,
                        },
                        f,
                        indent=2,
                    )
        self.clean(self.temp_save_path)
        print(f"{(time.perf_counter() - start )/60} minutes")
                # img_copy = self.draw(img,boxes)
                # out_file = OUTPUT_SAVE_PATH / "img" / f"{img.stem}.jpg"
                # out_file.parent.mkdir(parents=True, exist_ok=True)
                # cv2.imwrite(out_file,img_copy)
            

if __name__ == "__main__":   
    start = time.perf_counter()
    text_det = TextDetection(model_path="./openvino/text_detection/model.xml",pdf_to_photo="../pdf_to_photo",output_save_path="../det_output_save_path",temp_save_path="../temp_save_path") 
    text_det.predict("./client_data") 
    end = time.perf_counter()
    print(f"{end-start} seconds")