import torch
import cv2
import numpy as np
from torchvision.ops import nms
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from tqdm import tqdm


class YOLOv5Interface:
    def __init__(self, model_path='yolov5s.pt', device='cpu', conf_threshold=0.5, iou_threshold=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def load_model(self, model_path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.to(self.device).eval()
        return model

    def preprocess_image(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            img = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img = img.float() / 255.0
        return img

    def postprocess_result(self, results):
        pred = results[0]  # Assuming batch size of 1
        boxes = pred[:, :4].cpu()
        scores = pred[:, 4].cpu()
        labels = pred[:, 5].cpu()

        # Apply confidence threshold
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Apply NMS
        keep = nms(boxes, scores, self.iou_threshold)
        boxes = boxes[keep].numpy()
        scores = scores[keep].numpy()
        labels = labels[keep].numpy()

        return boxes, scores, labels

    def run_inference(self, image):
        img = self.preprocess_image(image)
        with torch.no_grad():
            results = self.model(img)
        boxes, scores, labels = self.postprocess_result(results)
        return boxes, scores, labels

    def format_results(self, boxes, scores, labels):
        formatted_results = []
        for box, score, label in zip(boxes, scores, labels):
            formatted_results.append({
                'box': box.tolist(),
                'score': score,
                'label': int(label)
            })
        return formatted_results


def evaluate_model(yolo_interface, image_paths, ground_truths, conf_thresholds, iou_thresholds):
    best_conf_threshold = 0
    best_iou_threshold = 0
    best_f1_score = 0

    for conf_threshold in conf_thresholds:
        for iou_threshold in iou_thresholds:
            yolo_interface.conf_threshold = conf_threshold
            yolo_interface.iou_threshold = iou_threshold

            all_true_labels = []
            all_pred_labels = []

            for image_path, ground_truth in zip(image_paths, ground_truths):
                boxes, scores, labels = yolo_interface.run_inference(image_path)
                pred_labels = labels.tolist()
                true_labels = ground_truth  # Assuming ground_truth is a list of labels

                all_true_labels.extend(true_labels)
                all_pred_labels.extend(pred_labels)

            precision = precision_score(all_true_labels, all_pred_labels, average='weighted')
            recall = recall_score(all_true_labels, all_pred_labels, average='weighted')
            f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')

            if f1 > best_f1_score:
                best_f1_score = f1
                best_conf_threshold = conf_threshold
                best_iou_threshold = iou_threshold

            print(
                f'Conf Threshold: {conf_threshold}, IoU Threshold: {iou_threshold}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    return best_conf_threshold, best_iou_threshold, best_f1_score


if __name__ == '__main__':
    yolo_interface = YOLOv5Interface(model_path='yolov5s.pt', device='cuda')

    # Example data
    image_dir = 'D:/runa/train_69/pytorch/yolov5-5.0/images/'  # 替换为实际图像目录
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
    ground_truths = [[0, 1], [1], [0, 2], ...]  # 替换为实际的标签列表

    conf_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
    iou_thresholds = [0.4, 0.5, 0.6]

    best_conf_threshold, best_iou_threshold, best_f1_score = evaluate_model(yolo_interface, image_paths, ground_truths,
                                                                            conf_thresholds, iou_thresholds)

    print(
        f'Best Conf Threshold: {best_conf_threshold}, Best IoU Threshold: {best_iou_threshold}, Best F1-Score: {best_f1_score:.4f}')
