import colorsys
import os
from collections import defaultdict

import cv2
import numpy as np
import tqdm
import yaml
from filterpy.kalman import KalmanFilter


def video_probe(filename):
    """Returns width, height, num_frames, frame_rate."""
    c = cv2.VideoCapture(filename)
    try:
        return (
            int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(c.get(cv2.CAP_PROP_FRAME_COUNT)),
            c.get(cv2.CAP_PROP_FPS)
        )
    finally:
        c.release()


def ltwh2xysr(ltwh):
    """Преобразовать формат прямоугольника из [l, t, w, h] в [x, y, s, r].
    
    Вход может содержать вещественные числа. Выход не нужо округлять до целых.
    
    Вход: прямоугольник в формате [l, t, w, h].
    Выход: прямоугольник в формате [x, y, s, r].
    """
    return [
        ltwh[0] + ltwh[2] / 2,
        ltwh[1] + ltwh[3] / 2,
        (ltwh[2] + ltwh[3]) / 2,
        ltwh[2] / ltwh[3]
    ]


def xysr2ltwh(xysr):
    """Преобразовать формат прямоугольника из [x, y, s, r] в [l, t, w, h].
    
    Вход может содержать вещественные числа. Выход не нужо округлять до целых.
    
    Вход: прямоугольник в формате [x, y, s, r].
    Выход: прямоугольник в формате [l, t, w, h].
    """
    h = 2 * xysr[2] / (xysr[3] + 1)
    w = xysr[3] * h
    return [
        xysr[0] - w / 2,
        xysr[1] - h / 2,
        w,
        h
    ]


def get_F():
    """Возвращает матрицу F."""
    return [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ]


def get_Q(pos_std=1, scale_std=1, aspect_std=1,
          pos_velocity_std=0.1, scale_velocity_std=0.01):
    """Возвращает матрицу Q.
    
    Вход:
    1. pos_std, scale_std, aspert_std - стандартные отклонения ошибок (x, y), s, и r.
    2. pos_velocity_std, scale_velocity_std - стандартные отклонения ошибок скоростей (x, y) и s.
    
    Выход: матрица Q.
    """
    return np.diag([pos_std, pos_std, scale_std, aspect_std, pos_velocity_std, pos_velocity_std, scale_velocity_std]) ** 2


def get_H():
    """Возвращает матрицу H."""
    return [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ]


def get_R(pos_std=0.1, scale_std=3, aspect_std=3):
    """Возвращает матрицу R."""
    return np.diag([pos_std, pos_std, scale_std, aspect_std]) ** 2


def get_P(pos_std=3, scale_std=3, aspect_std=3,
          pos_velocity_std=100, scale_velocity_std=100):
    """Возвращает матрицу P.
    
    Вход:
    1. pos_std, scale_std, aspert_std - стандартные отклонения ошибок (x, y), s, и r.
    2. pos_velocity_std, scale_velocity_std - стандартные отклонения ошибок скоростей (x, y) и s.
    
    Выход: матрица P.
    """
    return np.diag([pos_std, pos_std, scale_std, aspect_std, pos_velocity_std, pos_velocity_std, scale_velocity_std]) ** 2


def create_kalman_filter(initial_state=None):
    f = KalmanFilter(dim_x=7, dim_z=4)
    
    if initial_state is not None:
        f.x = np.asarray(initial_state)

    f.F = np.asarray(get_F())
    f.Q = np.asarray(get_Q())
    f.H = np.asarray(get_H())
    f.R = np.asarray(get_R())
    f.P = np.asarray(get_P())
    
    return f


def batch_iou(predictions, detections):
    """Вычислить Intersection over Union между каждым предсказанием и детектом.
    
    Все прямоугольники в формате [left, top, width, height].
    
    Вход:
    1. predictions: Предсказания фильтра Калмана, матрица размера (N, 4).
    2. detections: Обнаруженные детектором прямоугольники, матрица размера (K, 4).
    
    Выход: Матрица размера (N, K) с попарными IoU.
    """
    result = np.zeros((len(predictions), len(detections)))
    for i, bp in enumerate(predictions):
        for j, bd in enumerate(detections):
            inter_left = max(bp[0], bd[0])
            inter_top = max(bp[1], bd[1])
            inter_right = min(bp[0] + bp[2], bd[0] + bd[2])
            inter_bottom = min(bp[1] + bp[3], bd[1] + bd[3])
            inter_w = max(inter_right - inter_left, 0)
            inter_h = max(inter_bottom - inter_top, 0)
            inter_area = inter_w * inter_h
            union_area = bp[2] * bp[3] + bd[2] * bd[3] - inter_area
            result[i, j] = inter_area / union_area
    return result


def match_bboxes(iou):
    """Найти соответствия между предсказаниями и детектами.
    
    ВНИМАНИЕ: Связывать прямоугольники с нулевым IoU не нужно.
    
    В обозначениях ниже N - число предсказаний, K - число детектов.
    
    Вход: Матрица попарных значений IoU с размером (N, K).
    
    Выход: Список длины N, в котором для каждого предсказания указан номер соответствующего детекта.
           Если какое-то предсказание оказалось несвязанным, в списке нужно указать None.
    """
    iou = iou.copy()
    ninf = -1
    matches = [None] * len(iou)
    if iou.size == 0:
        return matches
    while True:
        candidates = np.argmax(iou, axis=1)
        values = np.take_along_axis(iou, candidates[:, None], 1)[:, 0]
        best_candidate = np.argmax(values)
        best_value = values[best_candidate]
        if best_value <= 0:
            break
        matches[best_candidate] = candidates[best_candidate]
        iou[best_candidate] = ninf
        iou[:, candidates[best_candidate]] = ninf
    return matches


def batch_cosine_distance(embeddings1, embeddings2):
    """Посчитать косинусное расстояние для двух пар эмбедингов.
    
    Вход:
        embeddings1: Первый набор эмбеддингов, матрица размера (N, 128).
        embeddings2: Второй набор эмбеддингов, матрица размера (K, 128).
        
    Выход: Матрица размера (N, K) с косинусными расстояниями между векторами.
    """
    embeddings1 /= np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 /= np.linalg.norm(embeddings2, axis=1, keepdims=True)
    return (embeddings1[:, None, :] * embeddings2[None, :, :]).sum(2) / np.sqrt((embeddings1 ** 2).sum(1)[:, None]) / np.sqrt((embeddings2 ** 2).sum(1)[None, :])


def batch_similarity(bboxes1, embeddings1, bboxes2, embeddings2, alpha=0.5):
    return alpha * batch_iou(bboxes1, bboxes2) + (1 - alpha) * batch_cosine_distance(embeddings1, embeddings2)


def read_data(filename):
    with open(filename) as fp:
        tracks = yaml.safe_load(fp)
    by_frame = defaultdict(list)
    by_frame_labels = defaultdict(list)
    for track in tracks:
        for detection in track["detections"]:
            by_frame[detection["frame_idx"]].append(detection)
            by_frame_labels[detection["frame_idx"]].append(track["track_idx"])
    num_frames = max(by_frame) + 1
    detections = [[] for _ in range(num_frames)]
    embeddings = [[] for _ in range(num_frames)]
    markup = [[] for _ in range(num_frames)]
    for frame, frame_detections in by_frame.items():
        bboxes = [detection["bbox"] for detection in frame_detections]
        detections[frame] = [list(map(int, bbox)) for bbox in bboxes]
        embeddings[frame] = [detection["embedding"] for detection in frame_detections]
        markup[frame] = by_frame_labels[frame]
    return detections, embeddings, markup


def load_track_detections(filename, track_idx):
    with open(filename) as fp:
        tracks = yaml.safe_load(fp)
    for track in tracks:
        if track["track_idx"] == track_idx:
            break
    else:
        raise IndexError("Track {} was not found".format(track_idx))
    frames, bboxes = zip(*[(detection["frame_idx"], detection["bbox"]) for detection in track["detections"]])
    first_frame = frames[0]
    num_frames = frames[-1] - first_frame + 1
    full_bboxes = [None] * num_frames
    for frame, bbox in zip(frames, bboxes):
        full_bboxes[frame - first_frame] = bbox
    return full_bboxes


def eval_mismatch_rate(labels, markup, exclude_single_frame=False):
    assert len(labels) == len(markup)

    if exclude_single_frame:
        counts = defaultdict(int)
        for frame_labels in labels:
            for label in frame_labels:
                counts[label] += 1
        exclude = {label for label, count in counts.items() if count == 1}
    else:
        exclude = set()
    
    n_mismatched = 0
    n_total = 0
    
    seen_predicted = set()
    label_mapping = {}  # Mapping from markup label to predicted.
    for frame in range(len(labels)):
        for predicted, target in zip(labels[frame], markup[frame]):
            if target in exclude:
                continue
            if target not in label_mapping:
                label_mapping[target] = predicted
                if predicted in seen_predicted:
                    n_mismatched += 1
            elif label_mapping[target] != predicted:
                n_mismatched += 1
                label_mapping[target] = predicted
            seen_predicted.add(predicted)
            n_total += 1
    return n_mismatched / max(n_total, 1)


def id2color(index, ncolors=50, step=29):
    """Convert index to color in RGB format."""
    hue = ((index * step) % ncolors) / ncolors
    r, g, b = map(int, np.asarray(colorsys.hsv_to_rgb(hue, 1, 1)) * 255)
    return (r, g, b)


def draw_bbox_inplace(image, bbox, color=(0, 0, 255), thickness=2 / 400):
    """Draw bounding box in LTWH format."""
    l, t, w, h = map(int, bbox)
    image_width, image_height, _ = image.shape
    width = int(max(1, min(image_width, image_height) * thickness))
    cv2.rectangle(image, (l, t), (l + w, t + h), color, width)
    
    
def draw_track_inplace(image, bboxes, color=(0, 0, 255), thickness=2 / 400):
    for i, bbox in enumerate(bboxes):
        if bbox is None:
            continue
        brightness = 0.9 * (1 - i / max(len(bboxes) - 1, 1))
        bbox_color = (
            int(255 * brightness + color[0] * (1 - brightness)),
            int(255 * brightness + color[1] * (1 - brightness)),
            int(255 * brightness + color[2] * (1 - brightness))
        )
        draw_bbox_inplace(image, bbox, color=bbox_color, thickness=thickness)
        
        
def format_bboxes(bboxes, is_xysr=False, origin=None):
    bboxes = [(list(bbox) if bbox is not None else None)
              for bbox in bboxes]
    if is_xysr:
        bboxes = [(xysr2ltwh(bbox) if bbox is not None else None)
                  for bbox in bboxes]
    if origin is not None:
        for bbox in bboxes:
            if bbox is None:
                continue
            bbox[0] -= origin[0]
            bbox[1] -= origin[1]
    return bboxes


def render_bboxes(bboxes, predictions=None, is_xysr=False):
    all_bboxes = [bbox for bbox in bboxes if bbox is not None]
    if predictions is not None:
        all_bboxes = all_bboxes + list(predictions)[1:]
    if is_xysr:
        all_bboxes = list(map(xysr2ltwh, all_bboxes))
    all_bboxes = np.asarray(all_bboxes)
    left, top = all_bboxes[:, :2].min(0)
    right, bottom = all_bboxes[:, :2].max(0)
    max_width, max_height = all_bboxes[:, 2:].max(0)
    left = int(left - max_width)
    right = int(right + 2 * max_width)
    top = int(top - max_height)
    bottom = int(bottom + 2 * max_height)
    origin = (left, top)
    
    image = np.full((bottom - top, right - left, 3), 255, dtype=np.uint8)
    draw_track_inplace(image, format_bboxes(bboxes, is_xysr=is_xysr, origin=origin), color=(50, 50, 50))
    if predictions is not None:
        draw_track_inplace(image, format_bboxes(predictions, is_xysr=is_xysr, origin=origin), color=(255, 0, 0))
    return image


def render_video_bboxes(video_path, output_path, detections, labels):
    width, height, num_frames, frame_rate = video_probe(video_path)
    
    reader = cv2.VideoCapture(video_path)
    for codec in ["mjpg", "divx", "xvid", "h264", "x264"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
        if writer.isOpened():
            break
    else:
        print("ОШИБКА: Подходящий кодек не найден.")
        reader.release()
        return
    for frame in tqdm.tqdm(range(num_frames)):
        frame_bgr = reader.read()[1]
        if frame_bgr is None:
            break
        for bbox, label in zip(detections[frame], labels[frame]):
            color = id2color(label)
            draw_bbox_inplace(frame_bgr, bbox, color)
        writer.write(frame_bgr)
    writer.release()
    reader.release()