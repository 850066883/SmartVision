import numpy as np
import time
import cv2
import os
import imutils
import subprocess
import pyttsx3

# Initialize pyttsx3
engine = pyttsx3.init()

# Load the labels
# LABELS = open("C:/Users/850066883/Downloads/Smart Vision-NNFC/Smart Vision-NNFC/smart_vision/coco.names").read().strip().split("\n")

# Load the YOLO model
print("[INFO] loading YOLO from disk...")
# net = cv2.dnn.readNetFromDarknet("C:/Users/850066883/Downloads/Smart Vision-NNFC/Smart Vision-NNFC/smart_vision/yolov3-coco/yolov3.cfg", "C:/Users/850066883/Downloads/Smart Vision-NNFC/Smart Vision-NNFC/smart_vision/yolov3-coco/yolov3.weights")

ln = np.array(net.getLayerNames())
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)
frame_count = 0
start = time.time()
first = True
frames = []
DELAY_SECONDS = 8

# Directions dictionary
position_directions = {
    "top left": "Move right to avoid the object.",
    "top center": "Change direction to avoid the object.",
    "top right": "Move left to avoid the object.",
    "mid left": "Move right to avoid the object.",
    "mid center": "Change direction to avoid the object.",
    "mid right": "Move left to avoid the object.",
    "bottom left": "Move right and up to avoid the object.",
    "bottom center": "Move up to avoid the object.",
    "bottom right": "Move left and up to avoid the object."
}

while True:
    frame_count += 1
    ret, frame = cap.read()
    frame1 = cv2.flip(frame, 1)
    frames.append(frame1)

    if ret:
        if frame_count % 60 == 0:
            end = time.time()
            (H, W) = frame1.shape[:2]
            blob = cv2.dnn.blobFromImage(frame1, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            boxes = []
            confidences = []
            classIDs = []
            centers = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        centers.append((centerX, centerY))

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
            texts = []

            if len(idxs) > 0:
                for i in idxs.flatten():
                    centerX, centerY = centers[i][0], centers[i][1]

                    if centerX <= W / 3:
                        W_pos = "left "
                    elif centerX <= (W / 3 * 2):
                        W_pos = "center "
                    else:
                        W_pos = "right "

                    if centerY <= H / 3:
                        H_pos = "top "
                    elif centerY <= (H / 3 * 2):
                        H_pos = "mid "
                    else:
                        H_pos = "bottom "

                    texts.append(H_pos + W_pos + LABELS[classIDs[i]])

            print(texts)
            cv2.imshow("Output", frame)

            
            if texts:
                description = ', '.join(texts)
                engine.say(description)

                # Get position label
                position_label = ' '.join(texts[0].split()[:-1])  # Convert list to string

                # Get corresponding direction from the dictionary
                direction = position_directions.get(position_label, "Change the direction to avoid the obstacle.")
                engine.say(direction)

                engine.runAndWait()


cap.release()
cv2.destroyAllWindows()
