from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np
import cv2

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

emotion_prompt_map = {
    "happy": "vivid impressionist landscape, bright colors, joyful mood",
    "sad": "dark rainy cityscape, blue tones, melancholy",
    "angry": "intense expressionist portrait, bold red strokes, dramatic lighting",
    "surprise": "abstract fireworks, vibrant colors, sharp details",
    "fear": "surreal night scene, ambiguous shapes, cold colors",
    "disgust": "abstract textures with green and brown hues, unsettling, organic patterns",
    "neutral": "minimalist art, soft tones, simple composition"
}

def generate_background(prompt):
    image = pipe(prompt).images[0]
    return image

def replace_background(foreground_frame, background_image):
    # Convert pil to OpenCV
    bg = np.array(background_image.resize((foreground_frame.shape[1], foreground_frame.shape[0])))
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    # Simple segmentation by color thresholding (placeholder, replace with proper segmentation model as needed)
    hsv = cv2.cvtColor(foreground_frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 50])  # dark pixels as background example
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)

    fg = cv2.bitwise_and(foreground_frame, foreground_frame, mask=mask_inv)
    bg_masked = cv2.bitwise_and(bg, bg, mask=mask)
    combined = cv2.add(fg, bg_masked)

    return combined
