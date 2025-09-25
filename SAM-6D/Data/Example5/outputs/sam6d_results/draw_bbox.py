import json
import cv2
import numpy as np

# 1. Load your json file
with open("./detection_ism.json", "r") as f:
    detections = json.load(f)


# Create blank image (640x480, white background)
image = 255 * np.ones((480, 640, 3), dtype=np.uint8)

for det in detections:
    x, y, w, h = det["bbox"]
    score = det["score"]
    cat_id = det["category_id"]

    # top-left and bottom-right corners
    pt1 = (int(x), int(y))
    pt2 = (int(x + w), int(y + h))

    # draw rectangle
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

    # put text (category_id + score)
    label = f"cat:{cat_id} {score:.2f}"
    cv2.putText(image, label, (pt1[0], pt1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Show image
# cv2.imshow("BBoxes", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save result
cv2.imwrite("bboxes_drawn.png", image)