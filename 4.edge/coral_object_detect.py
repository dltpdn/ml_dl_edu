import numpy as np
import cv2
import time
from PIL import Image
from edgetpu.detection.engine import DetectionEngine


model = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
label_path = "coco_labels.txt"
labels = {}
box_colors = {}
engine = DetectionEngine(model)
prevTime = 0

with open(label_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        id, name = line.strip().split(maxsplit=1)
        labels[id] = name

cap = cv2.VideoCapture(-1)
while True:
    ret, frame = cap.read()
    if not ret:
        print("cannot read frame.")
        break
    img = frame[:, :, ::-1].copy()
    img = Image.fromarray(img)
    candidate = engine.detect_with_image(img, threshold=0.5, keep_aspect_ratio=True, relative_coord=False, top_k=3)
    if candidate:
        for obj in candidate:
            box = obj.bounding_box.flatten().tolist()
            box_left = int(box[0])
            box_top = int(box[1])
            box_right = int(box[2])
            box_bottom = int(box[3])

            if obj.label_id in box_colors:
                box_color = box_colors[obj.label_id]
            else :
                box_color = [int(j) for j in np.random.randint(0,255, 3)]
                box_colors[obj.label_id] = box_color

            cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), box_color, 2)

            percentage = int(obj.score * 100)
            label_text = labels[obj.label_id] + " (" + str(percentage) + "%)" 

            txt, base = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 1, 3)[0]
            cv2.rectangle(frame, (box_left - 1, box_top - base-txt[1]), (box_left + txt[0], box_top + txt[1]), box_color, -1)
            cv2.putText(frame, label_text, (box_left, box_top), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)

            

    currTime = time.time()
    fps = 1/ (prevTime - currTime)
    prevTime = currTime
    cv2.putText(frame, "fps:%.1f"%fps, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    cv2.imshow('Object Detecting', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break