import cv2 
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import io
import os
import json
from PIL import Image
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np



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

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def get_gemini_model(task_type="detection"):
    return genai.GenerativeModel('gemini-2.5-flash')

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



def create_rect_mask(image_path):
    """
    Draw rectangles over crossmarks to create a binary mask.

    Controls:
    - Left click + drag: draw rectangle (region to inpaint, will be white in mask)
    - 'u': undo last rectangle
    - 'c': clear all rectangles
    - 's': save mask and exit
    - 'q' or ESC: quit without saving
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    img_disp = img.copy()
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)     # final mask
    rectangles = []                             # list of (x1, y1, x2, y2)
    drawing = False
    x0, y0 = -1, -1

    def redraw():
        """Redraw display image and mask overlay from rectangles."""
        nonlocal img_disp, mask
        img_disp = img.copy()
        mask[:] = 0
        for (x1, y1, x2, y2) in rectangles:
            cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, x0, y0, img_disp

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x0, y0 = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # show live rectangle
            redraw()
            cv2.rectangle(img_disp, (x0, y0), (x, y), (0, 0, 255), 1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = x0, y0
            x2, y2 = x, y
            # normalize coords
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            # avoid zero-area rectangles
            if abs(x2 - x1) > 1 and abs(y2 - y1) > 1:
                rectangles.append((x1, y1, x2, y2))
            redraw()

    cv2.namedWindow("Rect Mask Editor", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Rect Mask Editor", mouse_callback)

    print("Instructions:")
    print("  Left drag: draw rectangle over crossmark")
    print("  u: undo last rectangle")
    print("  c: clear all")
    print("  s: save mask and exit")
    print("  q or ESC: quit without saving")

    while True:
        cv2.imshow("Rect Mask Editor", img_disp)
        cv2.imshow("Current Mask", mask)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('s'):
            cv2.imwrite(output_mask_path, mask)
            print(f"Mask saved to {output_mask_path}")
            break
        elif key == ord('u'):
            if rectangles:
                rectangles.pop()
                redraw()
        elif key == ord('c'):
            rectangles.clear()
            redraw()
        elif key in [ord('q'), 27]:  # 'q' or ESC
            break

    cv2.destroyAllWindows()
    return Image.fromarray(mask)




# Load pre-trained inpainting model
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def inpaint_ultrasound(image_path, mask_path, prompt="clean medical ultrasound image without artifacts", strength=0.99, guidance_scale=7.5):
    """
    Inpaint ultrasound image to remove crossmarks.
    
    Args:
    - image_path: Path to original ultrasound PNG/JPG.
    - mask_path: Path to mask (white=area to inpaint, black=keep) same size as image.
    - prompt: Describes desired fill (e.g., "homogeneous breast tissue ultrasound").
    - strength: Denoising strength (0.8-1.0 for strong artifacts).
    - guidance_scale: Prompt adherence (5-10).
    
    Returns: Inpainted PIL Image.
    """
    init_image = Image.open(image_path).convert("RGB").resize((512, 512))
    mask_image = Image.open(mask_path).convert("L").resize((512, 512))
    
    result = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=50,
        strength=strength,
        guidance_scale=guidance_scale
    ).images[0]
    
    return result



