import cv2 
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import io
import os
import json
from PIL import Image


def read_as_grayscale(img_path):
    # Read image with OpenCV in grayscale mode
    img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        raise ValueError(f"Failed to read image from {img_path}")
    img_cv = img_cv.astype(np.float32) / 255.0
    return img_cv

def reshape_to(img_to, img_from):
    h,w = img_from.shape
    return cv2.resize(img_to, (w, h), interpolation=cv2.INTER_LINEAR)

def imdisp(img):
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.show()



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


def show_image_and_mask(image_path, mask_path, alpha=0.5, mask_color=(0, 0, 255)):
    """
    Display:
      - Original image
      - Image with mask overlaid (mask as transparent colored overlay)

    Args:
        image_path (str or Path): Path to the original image.
        mask_path  (str or Path): Path to the corresponding mask.
        alpha (float): Transparency of the mask overlay (0..1).
        mask_color (tuple): BGR color for the mask overlay.
    """
    # Read original image (BGR)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Read mask (grayscale or unchanged)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read mask from {mask_path}")

    # If mask has multiple channels, convert to single-channel
    if mask.ndim == 3:
        # Take one channel or convert to gray
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    # Ensure same spatial size
    if image.shape[:2] != mask_gray.shape[:2]:
        raise ValueError(
            f"Image and mask must have same size, got {image.shape[:2]} vs {mask_gray.shape[:2]}"
        )

    # Create a 3-channel color mask for overlay
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:] = mask_color  # BGR

    # Create boolean mask: where mask > 0 is foreground
    # Adjust threshold as needed (e.g., > 127) depending on your masks
    mask_binary = mask_gray > 0

    # Prepare overlay image (copy of original)
    overlay = image.copy()

    # Blend only where mask is 1
    overlay[mask_binary] = cv2.addWeighted(
        image[mask_binary], 1 - alpha,
        color_mask[mask_binary], alpha,
        0
    )

    # Show original and overlay
    cv2.imshow("Original", image)
    cv2.imshow("Image + Mask Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally return the overlay for further use
    return overlay


# Configure your API key

from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def get_gemini_model(task_type="detection"):
    """Returns the configured Gemini model."""
    # Using Gemini 1.5 Flash for speed, or Pro for higher reasoning accuracy
    if task_type == "localization":
        # Use the 'Pro' model for high-precision spatial coordinates
        # Try 'gemini-3-pro' or 'gemini-2.5-pro' depending on your API access
        return genai.GenerativeModel('gemini-2.5-pro') 
        
    else:
        # Use 'Flash' for fast, cheap binary classification
        return genai.GenerativeModel('gemini-2.5-flash')

# ==========================================
# Task 1: Artifact Detection
# ==========================================

def detect_artifacts(image_path):
    """
    Uses Gemini to analyze an ultrasound image for UI artifacts.
    Returns 1 if artifacts are present, 0 otherwise.
    """
    model = get_gemini_model()
    
    # Load image for Gemini
    img = Image.open(image_path)
    
    # Strict prompt for classification
    prompt = """
    Analyze this breast cancer ultrasound image.
    Determine if the image contains any artificial UI elements or overlays.
    
    CRITERIA FOR "ARTIFACTS" (Present = 1):
    - Cross-hair markers or cursors (often green or yellow cyan).
    - Patches of RGB color (e.g., elastography overlays or Doppler boxes).
    - Text annotations, numbers, or rulers inside the scan area.
    
    CRITERIA FOR "CLEAN" (Absent = 0):
    - Pure grayscale tissue texture.
    - Black background in corners is NOT an artifact. Do not count the conical shape borders as artifacts.
    
    Output purely the integer '1' if artifacts are present, or '0' if the image is clean. 
    Do not write any other text.
    """
    
    try:
        response = model.generate_content([prompt, img])
        # Clean response to ensure we get just the number
        result = response.text.strip()
        return int(result)
    except Exception as e:
        print(f"Error in detection: {e}")
        return 0

# ==========================================
# Task 2: Artifact Removal (Inpainting)
# ==========================================

def get_artifact_coordinates(image_path):
    """
    Asks Gemini to locate the artifacts and return bounding boxes.
    """
    model = get_gemini_model(task_type="localization")
    img = Image.open(image_path)
    
    # Prompt to get precise coordinates in JSON format
    prompt = """
    Identify the bounding boxes for all artificial UI elements in this ultrasound image.
    Target elements: Cross-hairs, text overlays, rulers, and colored (RGB) patches.
    Ignore: The natural black background corners and the tissue itself.
    
    Return a JSON list of objects. Each object must have a 'box_2d' field with [ymin, xmin, ymax, xmax] coordinates.
    The coordinates must be normalized (0 to 1).
    
    Example Output Format:
    [
      {"box_2d": [0.1, 0.2, 0.15, 0.3], "label": "crosshair"},
      {"box_2d": [0.8, 0.8, 0.9, 0.9], "label": "text"}
    ]
    """
    
    # Set response MIME type to JSON for easier parsing
    response = model.generate_content(
        [prompt, img],
        generation_config={"response_mime_type": "application/json"}
    )
    
    return json.loads(response.text)

def clean_ultrasound_image(image_path, output_path):
    """
    Pipeline: 
    1. Check if artifacts exist (Task 1).
    2. If yes, get coordinates from Gemini (Task 2a).
    3. Use OpenCV to inpaint those specific regions (Task 2b).
    """
    # Step 1: Check for artifacts
    has_artifacts = detect_artifacts(image_path)
    
    if has_artifacts == 0:
        print("Image is clean. No processing needed.")
        # Save copy of original if needed, or just return
        img = cv2.imread(image_path)
        cv2.imwrite(output_path, img)
        return

    print("Artifacts detected. Proceeding to inpaint...")

    # Step 2: Get coordinates from Gemini
    # Note: Gemini 1.5 Pro is recommended here for better spatial reasoning
    artifacts_data = get_artifact_coordinates(image_path)
    
    # Step 3: Processing with OpenCV
    img = cv2.imread(image_path)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w = img.shape[:2]
    
    found_boxes = False
    
    for item in artifacts_data:
        # Gemini returns [ymin, xmin, ymax, xmax] normalized 0-1
        ymin, xmin, ymax, xmax = item['box_2d']
        
        # Convert to pixel coordinates
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)
        
        # Create mask: White areas are what we want to remove
        # We add a small padding (dilating) to ensure the edge of the UI element is covered
        pad = 5 
        cv2.rectangle(mask, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 255, -1)
        found_boxes = True

    if not found_boxes:
        print("Gemini detected artifacts but failed to return valid boxes.")
        return

    # Inpaint: Replaces marked pixels using neighboring pixels (Telea algorithm)
    # radius=3 is usually good for thin lines/text
    cleaned_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    # Verify size is unchanged
    assert cleaned_img.shape == img.shape, "Error: Image size changed!"
    
    # Save output
    cv2.imwrite(output_path, cleaned_img)
    print(f"Cleaned image saved to {output_path}")

# ==========================================
# Example Usage
# ==========================================

# Create a dummy image or use a real path
# clean_ultrasound_image("patient_scan_001.jpg", "patient_scan_001_clean.jpg")
