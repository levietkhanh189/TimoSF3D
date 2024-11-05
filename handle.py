import torch
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download

class StableFast3DHandler:
    def __init__(self, model_name="stabilityai/stable-fast-3d"):
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def preprocess(self, image_base64):
        """
        Preprocesses the input image.
        """
        # Decode the base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        # Resize to 512x512 as required by the model
        image = image.resize((512, 512))
        
        # Preprocess with processor
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs

    def inference(self, inputs):
        """
        Performs inference on preprocessed inputs.
        """
        # Pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert model output to the desired format (e.g., 3D mesh)
        # For simplicity, assuming outputs contain vertex positions and texture
        vertices = outputs.vertices.detach().cpu().numpy()  # Example output extraction
        texture = outputs.texture.detach().cpu().numpy()
        
        # Return results
        return vertices, texture

    def postprocess(self, vertices, texture):
        """
        Converts vertices and texture to a standardized 3D format (e.g., OBJ or glTF).
        """
        # Simple OBJ-like output format, adapt as needed for actual format
        obj_data = "# 3D Model\n"
        for vert in vertices:
            obj_data += f"v {vert[0]} {vert[1]} {vert[2]}\n"
        
        # Texture coordinates (assumed to be part of the model output)
        for tex in texture:
            obj_data += f"vt {tex[0]} {tex[1]}\n"
        
        return obj_data

    def __call__(self, request):
        """
        Main handler function to be called with each inference request.
        """
        # Parse request (assuming JSON with 'image' in base64 format)
        image_base64 = request.get("image", None)
        if not image_base64:
            return {"error": "No image data provided."}

        # Preprocess the image
        inputs = self.preprocess(image_base64)
        
        # Perform inference
        vertices, texture = self.inference(inputs)
        
        # Post-process to convert to 3D model format
        model_data = self.postprocess(vertices, texture)
        
        # Return 3D model as a string (or in other formats if required)
        return {"3d_model": model_data}
