from ultralytics import YOLO
import cv2

def write_boxes_to_image(results, image):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confidence
            conf = float(box.conf[0])

            # Class label
            cls = int(box.cls[0])
            label = model.names[cls]

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

# Load a pretrained YOLO model (YOLOv8)
model = YOLO("ai/models/yolov8n.pt")  # 'n' = nano (fast). Options: s, m, l, x

# Load an image
image_path = "ai/input.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not read image at {image_path}")
    exit(1)

print(image.shape)

# Run detection
results = model(image)
write_boxes_to_image(results, image)

# Show image
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()