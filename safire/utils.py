import json
import os
import base64
import logging

import re
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def pil_image_to_base64(image):
    """Convert a PIL image to a base64 string."""
    try:
        logger.debug("Converting PIL image to base64")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_str = f"data:image/jpeg;base64,{img_str}"  # Add data URI scheme
        logger.debug("Successfully converted PIL image to base64")
        return img_str
    except Exception as e:
        logger.error(f"Error converting PIL image to base64: {e}")
        raise

def base64_to_pil_image(base64_str):
    """Convert a base64 string to a PIL image."""
    try:
        logger.debug("Converting base64 string to PIL image")
        img_data = base64.b64decode(base64_str.split(",")[1])  # Remove the data URI scheme
        image = Image.open(BytesIO(img_data))
        logger.debug("Successfully converted base64 to PIL image")
        return image
    except Exception as e:
        logger.error(f"Error converting base64 to PIL image: {e}")
        raise

def image_to_base64_str(sample):
    return {
        'base64_image': pil_image_to_base64(sample['image']),
    }

def extract_answer_letter(text):
    """
    Extracts the answer letter from the ground truth text.
    Assumes the format "A. Smoke above the fire" or similar.
    Returns the first character.
    """
    if not text:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    return text[0].upper()

def parse_model_output(output_text):
    """
    Robustly parses the model output to extract the answer letter (A-D).
    Strategies:
    1. Look for explicit pattern "Answer: X".
    2. Look for "X." or "X)" at the start.
    3. Look for single letter output.
    4. Fallback: Search for first "X." or "X)" pattern if verbose.
    """
    if not output_text:
        return None
        
    text = output_text.strip()

    # Strip </think> tags if present (common in some thinking models)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()  # Take the part after the last </think>
    
    # Strategy 1: Explicit "Answer: X" or "The answer is X"
    # Search from end to avoid finding reasoning steps like "Option A is bad... so Answer: B"
    match = re.search(r'(?:answer|option) is\s*([A-E])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'answer:\s*([A-E])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 2: Starts with "X." or "X)"
    match = re.match(r'^([A-E])[\.\)]', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # Strategy 3: Exact match (just the letter)
    if re.match(r'^([A-E])$', text, re.IGNORECASE):
        return text.upper()
        
    # Strategy 4: Fallback - Look for "X." anywhere, but this is risky.
    # Let's try looking for the last occurrence of something that looks like an answer if it's verbose.
    # But usually models trained for MCQA output the letter first or last.
    
    # If standard patterns fail, let's look for just the letter at the start again, maybe with some loose punctuation
    match = re.search(r'^([A-E])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None

