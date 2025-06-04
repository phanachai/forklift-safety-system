import cv2
import numpy as np
import onnxruntime as ort

# ----- CONFIG -----
MODEL_PATH = "bestv6.onnx"  # ONNX path
VIDEO_PATH = 'London Bank Holiday Walk 2025 _ Easter Sunday _ Summer Street Walking Tour In Central London 4K.mp4'
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5
CLASS_NAME = "Personas" 

# ----- ONNX Session -----
session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
img_size = input_shape[2]

# ----- Load Video -----
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดวิดีโอได้")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]
    center_x, center_y = w0 // 2, int(h0 * 1)

    aspect_ratio = 2
    danger_size = (int(200 * aspect_ratio), 200)
    warning_size = (int(400 * aspect_ratio), 400)
    safe_size = (int(600 * aspect_ratio), 600)

    cv2.ellipse(frame, (center_x, center_y), safe_size, 0, 180, 360, (0, 255, 0), 2)
    cv2.ellipse(frame, (center_x, center_y), warning_size, 0, 180, 360, (0, 255, 255), 2)
    cv2.ellipse(frame, (center_x, center_y), danger_size, 0, 180, 360, (0, 0, 255), 2)

    danger_contour = cv2.ellipse2Poly((center_x, center_y), danger_size, 0, 180, 360, 1)
    warning_contour = cv2.ellipse2Poly((center_x, center_y), warning_size, 0, 180, 360, 1)
    safe_contour = cv2.ellipse2Poly((center_x, center_y), safe_size, 0, 180, 360, 1)

    # ----- Preprocess -----
    resized = cv2.resize(frame, (img_size, img_size))
    img = resized[:, :, ::-1].transpose(2, 0, 1) / 255.0  # BGR→RGB & HWC→CHW
    img = np.expand_dims(img.astype(np.float32), axis=0)

    outputs = session.run(None, {input_name: img})
    output = np.squeeze(outputs[0])  # (5, 8400)

    boxes = output[:4, :]
    conf = output[4, :]
    mask = conf > CONF_THRESHOLD
    boxes = boxes[:, mask]
    scores = conf[mask]

    if boxes.shape[1] > 0:
        x_c, y_c, w, h = boxes
        x1 = (x_c - w / 2) * w0 / img_size
        y1 = (y_c - h / 2) * h0 / img_size
        x2 = (x_c + w / 2) * w0 / img_size
        y2 = (y_c + h / 2) * h0 / img_size
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)

        bboxes = boxes_xyxy.tolist()
        confidences = scores.tolist()
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        for i in indices.flatten():
            x1, y1, x2, y2 = bboxes[i]
            color = (100, 100, 100)
            label = "Outside"

            # ตรวจโซน
            check_points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2), ((x1+x2)//2, (y1+y2)//2)]
            detected_zone = None
            for pt in check_points:
                if pt[1] < center_y:
                    continue
                if cv2.pointPolygonTest(danger_contour, pt, False) >= 0:
                    detected_zone = "DANGER"
                    color = (0, 0, 255)
                    break
                elif cv2.pointPolygonTest(warning_contour, pt, False) >= 0:
                    detected_zone = "WARNING"
                    color = (0, 255, 255)
                elif cv2.pointPolygonTest(safe_contour, pt, False) >= 0:
                    if detected_zone != "WARNING":
                        detected_zone = "SAFE"
                        color = (0, 255, 0)

            if detected_zone:
                label = detected_zone

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ----- Show -----
    cv2.imshow("YOLOv11 ONNX + Zones", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
