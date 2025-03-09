from ultralytics import YOLO
import os
import cv2
import numpy as np
import argparse

class Detector(object):
    def __init__(self, model_path="../weights/best.onnx"):
        self.model = YOLO(model_path, task="detect")
        self.class_names = self.model.names
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        print(f"Loaded successfully AI model!")
    
    def detect(self, image):
        results = self.model(image)
        # self.class_names = results[0].names
        bboxes = results[0].boxes.xyxy.cpu().numpy() # type float 32
        class_ids = results[0].boxes.cls.cpu().numpy().astype(np.int64)
        scores = results[0].boxes.conf.cpu().numpy()
        speed = results[0].speed

        return bboxes, scores, class_ids, speed
    
    def count(self, class_ids):
        unique_ids, counts = np.unique(class_ids, return_counts=True)
        contents = [f"{count} {self.class_names[val]}" for val, count in zip(unique_ids, counts)]
        return contents
    
    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0):
        """
        Combines drawing masks, boxes, and text annotations on detected objects.
        
        Parameters:
        - image: Input image.
        - boxes: Array of bounding boxes.
        - scores: Confidence scores for each detected object.
        - class_ids: Detected object class IDs.
        - mask_alpha: Transparency of the mask overlay.
        """
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        mask_img = image.copy()

        # Draw bounding boxes, masks, and text annotations
        for class_id, box, score in zip(class_ids, boxes, scores):
            color = self.colors[class_id]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Draw fill rectangle for mask
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            
            # Draw bounding box
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Prepare text (label and score)
            label = self.class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            
            # Calculate text size and position
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_size, thickness=text_thickness)
            th = int(th * 1.2)
            
            # Draw filled rectangle for text background
            cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            
            # Draw text over the filled rectangle
            cv2.putText(det_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                        (255, 255, 255), text_thickness, cv2.LINE_AA)

        # Blend the mask image with the original image
        det_img = cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

        return det_img
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script mẫu sử dụng argparse.")

    # Thêm các tham số dòng lệnh
    parser.add_argument("--model_path", type=str, required=True, default="../weights/best.onnx", help="Onnx model path")
    parser.add_argument("--im_path", type=str, required=True, default="../data/20180824-13-35-55-2.jpg", help="Input image path")
    parser.add_argument("--verbose", action="store_true", help="Bật chế độ verbose.")

    # Parse tham số
    args = parser.parse_args()

    # Initiate YOLO detector model
    detector = Detector(args.model_path)

    # Read input image
    im = cv2.imread(args.im_path)

    # Run detector
    bboxes, scores, class_ids, speed = detector.detect(im)

    # Count objects
    contents = detector.count(class_ids)

    # Visualize output
    vis_im = detector.draw_detections(im, bboxes, scores, class_ids)
