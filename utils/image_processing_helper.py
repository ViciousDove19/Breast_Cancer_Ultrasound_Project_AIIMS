import cv2 
import numpy as np
import matplotlib.pyplot as plt
#function to segment manually

def manual_mask_segmentation(image):
    pts = []
    drawing = [False]  # Using a mutable type to modify inside callback
    image_copy = image.copy()

    def draw_contour(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            pts.append((x, y))
            cv2.circle(image_copy, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow("Image", image_copy)
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            pts.append((x, y))
            cv2.line(image_copy, pts[-2], pts[-1], (0, 255, 0), 2)
            cv2.imshow("Image", image_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_contour)

    print("Draw the contour using left mouse button. Press 's' to finish and return the mask.")

    while True:
        cv2.imshow("Image", image_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break

    if pts:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        contour = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [contour], 255)
        cv2.imshow("Binary Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return mask
    else:
        cv2.destroyAllWindows()
        return None

