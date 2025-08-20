import requests
import time
import json
from PIL import Image
from io import BytesIO
import numpy as np
import torch

class ModelScopeKontextAPI:
    """
    A ComfyUI custom node to call the ModelScope Kontext (FLUX) API for image-to-image generation.
    This node handles image uploading, API payload construction, and result polling.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Define the maximum seed value accepted by the ModelScope API (32-bit integer limit)
        MAX_SEED = 2147483647
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False, "default": "your_modelscope_api_key"}),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of churning water."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                # Set the max value in the UI to guide the user, though the code will handle larger values.
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 1.5, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ModelScope"

    def tensor_to_pil(self, tensor):
        """Converts a torch tensor (CHW or HWC) to a PIL Image."""
        image_np = tensor.cpu().numpy().squeeze()
        if image_np.ndim == 3 and image_np.shape in:
             image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def pil_to_tensor(self, pil_image):
        """Converts a PIL Image to a torch tensor."""
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def upload_image_to_host(self, pil_image):
        """Uploads a PIL image to a free image hosting service to get a public URL."""
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        buffered.seek(0)
        
        url = 'https://freeimage.host/api/1/upload'
        # This is a public key for anonymous uploads to freeimage.host
        api_key = '6d207e02198a847aa98d0a2a901485a5'
        
        payload = {'key': api_key, 'action': 'upload', 'format': 'json'}
        files = {'source': ('image.png', buffered, 'image/png')}
        
        try:
            print("Uploading image to freeimage.host...")
            response = requests.post(url, data=payload, files=files, timeout=60)
            response.raise_for_status()
            result = response.json()
            if result.get("status_code") == 200 and result.get("image"):
                image_url = result["image"].get("url")
                print(f"Successfully uploaded to freeimage.host: {image_url}")
                return image_url
            else:
                error_msg = result.get("error", {}).get("message", "Unknown error from freeimage.host")
                raise Exception(f"Failed to upload to freeimage.host: {error_msg}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Image hosting service connection error: {e}")

    def generate_image(self, image, api_key, prompt, width, height, seed, steps, guidance):
        
        # --- CRITICAL FIX: Ensure the seed is within the valid range for the API ---
        MAX_SEED = 2147483647
        # Use the modulo operator to wrap the seed value, preventing out-of-bounds errors.
        seed_to_use = seed % (MAX_SEED + 1)
        
        print(f"Original seed: {seed}, Seed sent to API: {seed_to_use}")

        # Convert the input tensor to a PIL image
        pil_image = self.tensor_to_pil(image)

        # Upload the image to get a public URL
        image_url = self.upload_image_to_host(pil_image)
        
        # Prepare the data payload for the ModelScope API request
        size_str = f"{width}x{height}"
        
        payload = {
            "model": "MusePublic/FLUX.1-Kontext-Dev",
            "prompt": prompt,
            "image_url": image_url,
            "size": size_str,
            "seed": seed_to_use, # Use the corrected seed value
            "steps": steps,
            "guidance": guidance
        }
        
        print(f"Sending payload to ModelScope: {payload}")

        # Perform the API call to ModelScope
        base_url = 'https://api-inference.modelscope.cn/'
        common_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Step 1: Submit the asynchronous generation task
        try:
            response = requests.post(
                f"{base_url}v1/images/generations",
                headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
            )
            response.raise_for_status()
            task_id = response.json()["task_id"]
            print(f"ModelScope task started with ID: {task_id}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to start ModelScope task: {e}")
        
        # Step 2: Poll for the task result
        while True:
            try:
                result_response = requests.get(
                    f"{base_url}v1/tasks/{task_id}",
                    headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                )
                result_response.raise_for_status()
                data = result_response.json()

                if data["task_status"] == "SUCCEED":
                    output_image_url = data["output_images"]
                    print(f"Image generation successful: {output_image_url}")
                    # Download the resulting image
                    image_response = requests.get(output_image_url)
                    image_response.raise_for_status()
                    result_image = Image.open(BytesIO(image_response.content)).convert("RGB")
                    return (self.pil_to_tensor(result_image),)
                
                elif data["task_status"] == "FAILED":
                    error_message = data.get('message', 'Unknown error')
                    raise Exception(f"ModelScope image generation failed. Reason: {error_message}")

            except requests.exceptions.RequestException as e:
                raise Exception(f"Error while checking ModelScope task status: {e}")

            time.sleep(5)

# Required by ComfyUI to map the node class
NODE_CLASS_MAPPINGS = {
    "ModelScopeKontextAPI": ModelScopeKontextAPI
}

# Required by ComfyUI to display a friendly name
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeKontextAPI": "ModelScope Kontext API"
}