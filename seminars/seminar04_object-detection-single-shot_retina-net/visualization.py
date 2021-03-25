import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def show_image(image, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def draw_predictions(image, bboxes, scores, classes, class_to_label_map=None, threshold=0.5):
    palette = sns.color_palette(None, len(class_to_label_map))
    image_with_predictions = image.copy()
    
    for bbox, score, class_int in zip(bboxes, scores, classes):
        if score < threshold:
            continue
        
        x1, y1, x2, y2 = bbox.numpy().astype(np.int32)
        label = class_to_label_map[class_int.item()]
        color = palette[class_int.item()]
        
        cv2.rectangle(image_with_predictions, (x1, y1), (x2, y2), np.array(color) * 255, 2)
        if class_to_label_map is not None:
            cv2.putText(image_with_predictions, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(color) * 255, 2)
        
        
    return image_with_predictions
