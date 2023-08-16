from typing import Iterable, Optional, List

import cv2
import numpy as np


def tf_lite_inference(interpreter, input_data) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(np.concatenate([input_data, input_data], axis=0).shape)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


def prepare_image_numpy(image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    We want to keep data in range 0-255
    """
    # print(path)
    # image = cv2.imread(path)
    size = 640
    image = np.transpose(image, (1, 2, 0))
    image_base = cv2.resize(image, (size, size))
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image, dtype=np.float32)
    return np.expand_dims(image, axis=0), image_base


def IoU_numpy(bboxes_a, bboxes_b):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    tl = np.maximum(bboxes_a[:, None, :2], bboxes_b[:, :2])
    br = np.minimum(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
    area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).astype(tl.dtype).prod(axis=2)
    area_i = np.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def nms2_numpy(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # x1 = boxes[:, 0]
    # y1 = boxes[:, 1]
    # x2 = boxes[:, 2]
    # y2 = boxes[:, 3]

    # areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maximum iou box
        keep.append(i)
        
        bboxes_a = boxes[i, :][None, :]
        bboxes_b = boxes[order[1:], :]

        ovr = IoU_numpy(bboxes_a, bboxes_b)[0]
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def batched_nms_numpy(boxes: np.ndarray, scores: np.ndarray, idxs: np.ndarray, iou_threshold: float) -> List[int]:
    max_coordinate = boxes.max()
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms2_numpy(boxes_for_nms, scores, iou_threshold)
    return keep


def postprocess_numpy(
    prediction: np.ndarray, 
    num_classes: int, 
    conf_thre: float=0.7, 
    nms_thre: float=0.45, 
    class_agnostic: bool=False, 
    filter_class_ids: Iterable=None
) -> List[Optional[np.ndarray]]:
    box_corner = np.empty_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if not image_pred.any():
            continue
        
        # print(image_pred, image_pred.shape)
        # Get score and class with highest confidence
        a = np.max(image_pred[:, 5: 5 + num_classes], axis=1, keepdims=True)
        # print(a, a.shape)
        class_conf = np.max(image_pred[:, 5: 5 + num_classes], axis=1, keepdims=True)
        class_pred = np.argmax(image_pred[:, 5: 5 + num_classes], axis=1, keepdims=True)

        conf_mask = np.squeeze(image_pred[:, 4] * np.squeeze(class_conf) >= conf_thre)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.concatenate((image_pred[:, :5], class_conf, class_pred.astype(np.float32)), axis=1)
        detections = detections[conf_mask]
        if filter_class_ids is not None:
            detections = detections[np.isin(detections[:, 6], filter_class_ids)]
        if not detections.any():
            continue

        if class_agnostic:
            nms_out_index = nms2_numpy(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = batched_nms_numpy(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = np.concatenate((output[i], detections))

    # print(output)
    return output
