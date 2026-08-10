import numpy as np
import cv2
import math
from typing import Union
import pyclipper

class DBPostProcess:
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(
        self,
        thresh=0.4,   # Higher value, more clear text get's detected
        box_thresh=0.6, # Higher value, smaller but more clearer boxes
        max_candidates=1000,
        unclip_ratio=1.5, # How big a box around detected text must be
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
        **kwargs,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in [
            "slow",
            "fast",
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)
        self.use_dilation = use_dilation
    
    def polygons_from_bitmap(
        self,
        pred,
        _bitmap, # mask -> 0/1 for prediction
        dest_width,
        dest_height,
        box_thresh,
        unclip_ratio,
    ):
        """_bitmap: single map with shape (1, H, W), whose values are binarized as {0, 1}"""

        bitmap = _bitmap
        height, width = bitmap.shape
        width_scale = dest_width / width
        height_scale = dest_height / height
        boxes = []
        scores = []
        
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours[: self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if box_thresh > score:
                continue
            
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            if len(box) > 0:
                _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
                if sside < self.min_size + 2:
                    continue
            else:
                continue

            box = np.array(box)

            for i in range(box.shape[0]):
                box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
                box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))

            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(
        self,
        pred,
        _bitmap,
        dest_width,
        dest_height,
        box_thresh,
        unclip_ratio,
    ):
        """_bitmap: single map with shape (1, H, W), whose values are binarized as {0, 1}"""

        bitmap = _bitmap
        height, width = bitmap.shape
        width_scale = dest_width / width
        height_scale = dest_height / height

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2] 
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            for i in range(box.shape[0]):
                box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
                box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))

            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box, unclip_ratio):
        """unclip"""
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, True)
        distance = area * unclip_ratio / length
        offset = pyclipper.PyclipperOffset() #type:ignore
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON) # type:ignore
        try:
            expanded = np.array(offset.Execute(distance))
        except ValueError:
            expanded = np.array(offset.Execute(distance)[0])
        return expanded
    
    def get_mini_boxes(self, contour):
        """get mini boxes"""
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """box_score_fast: use bbox mean score as the mean score"""
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
        xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
        ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
        ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        """box_score_slow: use polygon mean score as the mean score"""
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]
    
    def process(
        self,
        pred,
        img_shape,
        thresh,
        box_thresh,
        unclip_ratio,
    ):
        pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[0, :, :]
        elif pred.ndim != 2:
            raise ValueError(f"Expected a 2D or 3D prediction map, got shape {pred.shape}")

        segmentation = pred > thresh
        dilation_kernel = None if not self.use_dilation else np.array([[1, 1], [1, 1]])
        src_h, src_w, ratio_h, ratio_w = img_shape
        if dilation_kernel is not None:
            mask = cv2.dilate(
                np.array(segmentation).astype(np.uint8),
                dilation_kernel,
            )
        else:
            mask = segmentation
        if self.box_type == "poly":
            boxes, scores = self.polygons_from_bitmap(
                pred, mask, src_w, src_h, box_thresh, unclip_ratio
            )
        elif self.box_type == "quad":
            boxes, scores = self.boxes_from_bitmap(
                pred, mask, src_w, src_h, box_thresh, unclip_ratio
            )
        else:
            raise ValueError("box_type can only be one of ['quad', 'poly']")
        return boxes, scores
    
    def __call__(
        self,
        preds,
        img_shapes,
        thresh: Union[float, None] = None,
        box_thresh: Union[float, None] = None,
        unclip_ratio: Union[float, None] = None,
    ):
        """apply"""
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = np.asarray(preds)
        if preds.ndim == 5 and preds.shape[0] == 1:
            preds = preds[0]
        elif preds.ndim == 2:
            preds = np.expand_dims(preds, axis=0)

        boxes, scores = [], []
        for pred, img_shape in zip(preds, img_shapes):
            box, score = self.process(
                pred,
                img_shape,
                thresh or self.thresh,
                box_thresh or self.box_thresh,
                unclip_ratio or self.unclip_ratio,
            )
            boxes.append(box)
            scores.append(score)
        return boxes, scores
    
