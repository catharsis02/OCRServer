import network
from text_detection import TextDetection
from text_recognition import TextRecognition
from layout_detection import LayoutDetection
import time



start = time.perf_counter()
det_model = TextDetection(model_path="openvino/text_detection/model.xml")
rec_model = TextRecognition(model_path="openvino/text_recognition/model.xml")

det_model.predict(pdf_path="./client_data")
final_results = rec_model.predict(image_path="./pdf2image",boxes_path="./det_outputs/json")


lay_dec = LayoutDetection(outputs_path="C:\\Users\\kanishkak3\\Desktop\\onnx_ocr\\rec_outputs\\json")
lay_dec.predict()
end = time.perf_counter()
print(f"{end-start} seconds")