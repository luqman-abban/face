import streamlit as st
from PIL import Image
from PIL import Image, ImageOps
import requests
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
import numpy as np

# Load the model outside the function for efficiency
feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

def create_face_mask_from_segmentation(segmentation_map):
    # This is a placeholder. Assume the face segment has a label of '1'
    face_mask = np.where(segmentation_map == 1, 255, 0).astype(np.uint8)
    return face_mask

def apply_mask(image, mask):
    # Convert numpy mask to PIL image
    mask_image = Image.fromarray(mask)
    # Apply mask to the image
    output_image = ImageOps.composite(image, Image.new("RGB", image.size, "black"), mask_image)
    return output_image

def process_image(input_image):
    image = Image.open(requests.get(input_image, stream=True).raw)

    # Segmentation using MaskFormer
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Extract segmentation results
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_panoptic_map = result["segmentation"].numpy()

    # Create a mask for the face
    face_mask = create_face_mask_from_segmentation(predicted_panoptic_map)

    # Apply the mask to the original image
    output_image = apply_mask(image, face_mask)

    return output_image

# Gradio interface setup
def setup_gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            # Choose 'pil' if you want to work with PIL Image objects
            image_input = gr.Image(label="Input Image", type="pil")
            image_output = gr.Image(label="Segmented Output")
        image_input.change(fn=process_image, inputs=image_input, outputs=image_output)
    
    return demo

demo = setup_gradio_interface()
demo.launch()
