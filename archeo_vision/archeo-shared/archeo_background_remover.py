import base64
import io
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"   # MODIFY IF NEEDED
OLLAMA_MODEL = "llama3.2-vision:90b"                      # MODIFY IF NEEDED


# ----------------------------------------------------------------------
# LLM COMMUNICATION
# ----------------------------------------------------------------------

def ask_llm_for_measurements(image_base64: str):
    """
    Send the image to the LLM and request ONLY:
    - longest_side_in_cm
    - artifact_length_cm
    - artifact_width_cm
    """

    prompt = """
Locate the scale bar located on the left, bottom, or below the artefact.
1. Establish the length of the longest side of the image in centimeters ("longest_side_in_cm").
2. Establish the height and length of the artefact in centimeters.

Return ONLY the following JSON structure and nothing else:

{
  "longest_side_in_cm": <float>,
  "artifact_length_cm": <float>,
  "artifact_width_cm": <float>
}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }

    r = requests.post(OLLAMA_ENDPOINT, json=payload)
    r.raise_for_status()
    response = r.json()["response"]

    # Extract JSON from the text response
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        data = json.loads(response[start:end])
    except Exception:
        raise ValueError("LLM did not return valid JSON: " + response)

    return (
        float(data["longest_side_in_cm"]),
        float(data["artifact_length_cm"]),
        float(data["artifact_width_cm"])
    )


# ----------------------------------------------------------------------
# IMAGE PROCESSING
# ----------------------------------------------------------------------

def load_png_rgba(png_bytes: bytes) -> Image.Image:
    """Load PNG (with alpha) into a Pillow Image."""
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def extract_object_pixel_dimensions(image_rgba: Image.Image):
    """
    Extract width/height (in pixels) of the object by reading
    the alpha channel and computing the bounding box.
    """

    arr = np.array(image_rgba)
    alpha = arr[:, :, 3]

    ys, xs = np.where(alpha > 0)

    if len(xs) == 0:
        raise ValueError("Object not found — alpha mask is empty")

    width_px = xs.max() - xs.min()
    height_px = ys.max() - ys.min()

    return width_px, height_px


# ----------------------------------------------------------------------
# MEASUREMENT CONVERSION
# ----------------------------------------------------------------------

def compute_pixels_per_cm(image_rgba: Image.Image, longest_side_in_cm: float):
    """
    Compute px/cm using the NEW method:
    pixels_per_cm = longest_side_px / longest_side_in_cm
    """

    w, h = image_rgba.size
    longest_side_px = max(w, h)

    return longest_side_px / longest_side_in_cm


def convert_px_to_cm(px: float, px_per_cm: float) -> float:
    return px / px_per_cm


# ----------------------------------------------------------------------
# SCALE BAR RENDERING
# ----------------------------------------------------------------------

def draw_scale_bar(image_rgba: Image.Image, px_per_cm: float, bar_length_cm=10):
    """
    Draw a 10 cm scale bar (or custom cm length) in the lower-left corner.
    """

    draw = ImageDraw.Draw(image_rgba)
    bar_length_px = int(bar_length_cm * px_per_cm)

    margin = 40
    thickness = 6

    x0 = margin
    y0 = image_rgba.height - margin
    x1 = x0 + bar_length_px
    y1 = y0 - thickness

    # Draw rectangle
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, 255))

    # Draw label
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    label = f"{bar_length_cm} cm"
    draw.text((x0, y1 - 30), label, fill=(255, 255, 255, 255), font=font)

    return image_rgba


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------

def process_artifact(image_bytes: bytes):
    """
    Full workflow:
    1. Load PNG
    2. Ask LLM for longest_side_in_cm + artifact (real) cm sizes
    3. Compute px/cm locally
    4. Measure object size in pixels from alpha mask
    5. Convert pixel → cm
    6. Draw scale bar
    7. Return all measurements + final PNG base64
    """

    # --------------------------------------------------
    # STEP 1: Decode PNG
    # --------------------------------------------------
    image_rgba = load_png_rgba(image_bytes)

    # Prepare base64 for LLM
    img_buffer = io.BytesIO()
    image_rgba.save(img_buffer, format="PNG")
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

    # --------------------------------------------------
    # STEP 2: Query LLM
    # --------------------------------------------------
    (
        longest_side_in_cm_from_llm,
        artifact_length_cm_llm,
        artifact_width_cm_llm
    ) = ask_llm_for_measurements(img_b64)

    # --------------------------------------------------
    # STEP 3: Compute px/cm locally
    # --------------------------------------------------
    px_per_cm = compute_pixels_per_cm(image_rgba, longest_side_in_cm_from_llm)

    # --------------------------------------------------
    # STEP 4: Measure pixel size of object
    # --------------------------------------------------
    artifact_width_px, artifact_height_px = extract_object_pixel_dimensions(image_rgba)

    # --------------------------------------------------
    # STEP 5: Convert to real cm
    # --------------------------------------------------
    artifact_width_cm = convert_px_to_cm(artifact_width_px, px_per_cm)
    artifact_height_cm = convert_px_to_cm(artifact_height_px, px_per_cm)

    # --------------------------------------------------
    # STEP 6: Draw scale bar
    # --------------------------------------------------
    final_img = draw_scale_bar(image_rgba, px_per_cm)

    # --------------------------------------------------
    # STEP 7: Encode PNG output
    # --------------------------------------------------
    buffer = io.BytesIO()
    final_img.save(buffer, format="PNG")
    buffer.seek(0)
    out_base64 = base64.b64encode(buffer.read()).decode()

    # --------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------
    return {
        "pixels_per_cm": px_per_cm,
        "artifact_width_cm_calc": artifact_width_cm,
        "artifact_height_cm_calc": artifact_height_cm,
        "artifact_width_cm_llm": artifact_width_cm_llm,
        "artifact_height_cm_llm": artifact_width_cm_llm,
        "longest_side_in_cm": longest_side_in_cm_from_llm,
        "image_base64": out_base64
    }

