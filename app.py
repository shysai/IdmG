code = """
# app.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import gradio as gr

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))
yolo_model = YOLO('yolov8n.pt')

def detect_heads(image):
    image_rgb = image.copy()
    faces = face_app.get(image_rgb)
    face_boxes = [face.bbox.astype(int) for face in faces]
    results = yolo_model(image_rgb)
    person_boxes = []
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        if int(cls) == 0:
            person_boxes.append([int(coord) for coord in box])
    final_detections = []
    used_face_indices, used_yolo_indices = set(), set()
    for i, p_box in enumerate(person_boxes):
        matching_face_idx = None
        for j, f_box in enumerate(face_boxes):
            if compute_iou(p_box, f_box) > 0.3:
                matching_face_idx = j
                break
        if matching_face_idx is not None and matching_face_idx not in used_face_indices:
            final_detections.append(face_boxes[matching_face_idx])
            used_face_indices.add(matching_face_idx)
            used_yolo_indices.add(i)
        else:
            x1, y1, x2, y2 = p_box
            head_box = [x1, y1, x2, y1 + (y2 - y1) // 3]
            final_detections.append(head_box)
            used_yolo_indices.add(i)
    for j, f_box in enumerate(face_boxes):
        if j not in used_face_indices:
            final_detections.append(f_box)
    annotated = image_rgb.copy()
    for i, box in enumerate(final_detections):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f'{i+1}', (x1, max(y1-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return annotated

demo = gr.Interface(
    fn=detect_heads,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Image(type="numpy", label="Annotated Output"),
    title="👤 Head Detection (YOLOv8 + InsightFace)"
)

if __name__ == "__main__":
    demo.launch()
"""

with open("app.py", "w") as f:
    f.write(code)
