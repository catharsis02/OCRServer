import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from tqdm import tqdm

from backend.src.text_recognition.CTCDecodePostprocess import CTCDecodePostProcess
from backend.src.text_recognition.batching import ToBatch
from backend.src.text_recognition.normalize import Normalize
from backend.src.text_recognition.read_image import validate_image
from backend.src.text_recognition.resize import Resize

OUTPUT_SAVE_PATH: Path = Path.cwd() / "rec_outputs"
sys.stdout.reconfigure(encoding='utf-8')  # type:ignore

# AI Gen
# ---------------------------------------------------------------------------
# Fixed shape "buckets". Instead of letting every page infer at its own
# unique (batch_size, width) shape -- which forces the GPU plugin to
# JIT-compile/select new kernels for every single call and grow its internal
# shape cache without bound -- we round each request UP to the nearest
# bucket. The number of distinct shapes the GPU ever sees becomes small and
# fixed, so kernels get reused across pages instead of recompiled.
# Tune these to your data (e.g. sample your typical box counts / crop
# widths and set the buckets to cover the common cases).
# ---------------------------------------------------------------------------

WIDTH_BUCKETS = [320, 480, 640, 800, 960, 1120, 1280, 1440, 1600, 1920, 2240, 2560, 2880, 3200]

# AI Gen
# Cap on how many crops go into one inference call. Without this, a page's
# entire box set is padded to that page's single widest crop -- e.g. one
# 2168px-wide line box forces 80+ small word-crops to all be computed at
# width 2168. Sorting by width and chunking bounds the padding waste to
# crops of similar size, and keeps sub-batch shapes far more consistent
# across pages (most chunks land on the same handful of buckets).
# BATCH_BUCKETS tops out at CHUNK_SIZE on purpose -- no chunk can ever be
# larger, so any bucket above it would never be used.


CHUNK_SIZE = 8
BATCH_BUCKETS = [1, 2, 4, 8, 16, 32]


def _next_bucket(value: int, buckets: list[int]) -> int:
    """Smallest bucket >= value; falls back to the exact value if it
    exceeds every bucket (better to compile one extra shape than to fail)."""
    for b in buckets:
        if value <= b:
            return b
    return value


class TextRecognition:
    def __init__(self, model_path, device: str = "AUTO",cache_dir : str|Path = "./.ov_cache"):
        core = ov.Core()

        # NOTE: CACHE_DIR must be set BEFORE compile_model() to have any
        # effect. Previously it was set after compilation, so the compiled
        # kernel cache was never actually written to disk / reused across
        # runs.
        core.set_property({"CACHE_DIR": cache_dir})

        model = core.read_model(str(model_path))
        self.compiled_model = core.compile_model(
            model,
            device,
            config={"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY"},
        )

        self.infer_request = self.compiled_model.create_infer_request()
        self.resize = Resize()
        self.normalize = Normalize()
        self.to_batch = ToBatch()
        self.decode = CTCDecodePostProcess()
        # AI Gen
        # Warm up every bucketed batch size at the smallest width bucket so
        # the most common shapes are already compiled before real traffic
        # hits the model. This trades a bit of startup time for consistent
        # per-page latency later.
        for batch_bucket in BATCH_BUCKETS:
            dummy = ov.Tensor(type=ov.Type.f32,shape=(batch_bucket,3,48,WIDTH_BUCKETS[0]))
            self.infer_request.infer(dummy)

    def _load_detection_boxes(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "boxes" in data:
            return data["boxes"]
        raise ValueError(f"Invalid detection format: {path}")

    def _order_points_clockwise(self, points):
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)
        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = points[np.argmin(sums)]
        rect[2] = points[np.argmax(sums)]
        rect[1] = points[np.argmin(diffs)]
        rect[3] = points[np.argmax(diffs)]
        return rect

    def _fallback_crop(self, image, points):
        h, w = image.shape[:2]
        x_min = max(0, int(points[:, 0].min()))
        y_min = max(0, int(points[:, 1].min()))
        x_max = min(w, int(points[:, 0].max()))
        y_max = min(h, int(points[:, 1].max()))
        return image[y_min:y_max, x_min:x_max]

    def _crop_text_region(self, image, box):
        points = np.asarray(box, dtype=np.float32)
        if len(points) != 4:
            return self._fallback_crop(image, points)
        rect = self._order_points_clockwise(points)
        tl, tr, br, bl = rect
        crop_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        crop_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if crop_width <= 0 or crop_height <= 0:
            return self._fallback_crop(image, points)
        target = np.array(
            [[0, 0], [crop_width - 1, 0], [crop_width - 1, crop_height - 1], [0, crop_height - 1]],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(rect, target)
        crop = cv2.warpPerspective(
            image, matrix, (crop_width, crop_height), borderMode=cv2.BORDER_REPLICATE,
        )

        if crop.shape[0] > crop.shape[1] * 1.5:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        return crop

    def _expand_box(self, points, padding_ratio=0.1):
        center = np.mean(points, axis=0)
        expanded = center + (points - center) * (1 + padding_ratio)
        return expanded.astype(np.float32)

    def _infer_chunk(self, chunk_imgs):
        """Run one bucketed inference call on a list of already-resized,
        already-normalized (3, 48, w) crops. Returns (texts, scores) in the
        same order as chunk_imgs."""
        x = self.to_batch(chunk_imgs).astype(np.float32)
        n_real, c, h, w_real = x.shape

        w_bucket = _next_bucket(w_real, WIDTH_BUCKETS)
        n_bucket = _next_bucket(n_real, BATCH_BUCKETS)

        if w_bucket != w_real or n_bucket != n_real:
            padded = np.zeros((n_bucket, c, h, w_bucket), dtype=np.float32)
            padded[:n_real, :, :, :w_real] = x
            x = padded
        self.infer_request.infer(x)
        pred = np.asarray(self.infer_request.get_output_tensor().data)

        # Drop the padding rows before decoding -- only the first n_real
        # entries correspond to real crops.
        pred = pred[:n_real]

        return self.decode(pred)

    def _predict_iter(self, images):
        if len(images) == 0:
            return [], []

        for img in images:
            validate_image(img)

        batch = self.resize(images)
        batch = self.normalize(batch)

        # Sort by width so each chunk contains crops of similar size --
        # otherwise a single wide crop in the page forces every crop in the
        # batch to be padded (and computed) at that width.
        order = np.argsort([img.shape[2] for img in batch])

        texts = [None] * len(batch)
        scores = [None] * len(batch)

        for start in range(0, len(order), CHUNK_SIZE):
            chunk_idx = order[start:start + CHUNK_SIZE]
            chunk_imgs = [batch[i] for i in chunk_idx]
            chunk_texts, chunk_scores = self._infer_chunk(chunk_imgs)
            for i, t, s in zip(chunk_idx, chunk_texts, chunk_scores):
                texts[i] = t
                scores[i] = s

        return texts, scores

    def predict(
        self,
        image_path: str | Path,
        boxes_path: str | Path,
        output_save_path:str|Path = OUTPUT_SAVE_PATH
    ) -> list[list[dict[str, float | str]]] | None:
        self.output_save_path = Path(output_save_path)
        if not isinstance(image_path, (str, Path)):
            raise TypeError(f"Image path must be str or Path, got {type(image_path).__name__}")
        if not isinstance(boxes_path, (str, Path)):
            raise TypeError(f"Image path must be str or Path, got {type(boxes_path).__name__}")
        if isinstance(image_path, str):
            image_path = Path(image_path)
        if isinstance(boxes_path, str):
            boxes_path = Path(boxes_path)
        if image_path.is_file():
            images = [image_path]
        else:
            images = [img for img in image_path.iterdir() if img.is_file()]

        final_results = []
        print("Predicting text from Images")
        for img in tqdm(images):
            image = cv2.imread(str(img))
            if image is None:
                raise FileNotFoundError(image_path)
            boxes = self._load_detection_boxes(boxes_path / f"{img.stem}.json")
            crops = [self._crop_text_region(image, box) for box in boxes]
            texts, scores = self._predict_iter(crops)
            results = [
                {"text": text, "score": float(score), "box": box}
                for box, text, score in zip(boxes, texts, scores)
            ]
            out_file = self.output_save_path / "json" / f"output_{img.stem}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = [
                {
                    "box": item["box"],
                    "text": item["text"],
                    "score": item["score"]
                }
                for item in results
            ]

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data,f,indent=2)
            final_results.append(results)
        return final_results

if __name__ == "__main__":
    start = time.perf_counter()
    text_rec = TextRecognition(model_path="./openvino/text_recognition/model.xml")
    text_rec.predict(image_path="../pdf_to_photo", boxes_path="../det_output_save_path/json",output_save_path="../rec_output_save_path")
    end = time.perf_counter()
    print(f"{end - start} seconds")