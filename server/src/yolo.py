from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model (YOLOv8)
YOLO_MODEL_NANO = YOLO("models/yolov8n.pt")
YOLO_MODEL_MEDIUM = YOLO("models/yolov8m.pt")
YOLO_MODEL_LARGE = YOLO("models/yolov8x.pt")

def write_boxes_to_image(results, image, names):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confidence
            conf = float(box.conf[0])

            # Class label
            cls = int(box.cls[0])
            label = names[cls]

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

def get_model(t):
    if t == "nano":
        return YOLO_MODEL_NANO
    elif t == "medium":
        return YOLO_MODEL_MEDIUM
    elif t == "large":
        return YOLO_MODEL_LARGE
    else:
        return False

def infer(image, model, device):
    results = model(image, device=device)
    return results

def object_detection(image, model_type, device):
    model = get_model(model_type)
    results = infer(image, model, device)
    write_boxes_to_image(results, image, model.names)
    return image

