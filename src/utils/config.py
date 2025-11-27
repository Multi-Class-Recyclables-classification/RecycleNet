import os 
import torch 
from torchvision.transforms import transforms
from  PIL import Image
from .inference import resnet ,test_transform , device
import io


def prepare_image(image_bytes):
    
    
    
    try:
            
            image_buffer = io.BytesIO(image_bytes)
            image = Image.open(image_buffer).convert("RGB")
            image_tensor = test_transform(image).unsqueeze(0)
            return image_tensor
    except Exception as e:
            raise ValueError(f"Error preparing image: {str(e)}")



def predict(img_tensor: torch.Tensor) -> str:

    
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outputs = resnet(img_tensor)
        _, predicted = torch.max(outputs, 1)

    classes = {
        0: 'biological',
        1: 'cardboard',
        2: 'glass',
        3: 'metal',
        4: 'paper',
        5: 'plastic',
        6: 'trash'
    }

    return classes.get(predicted.item(), "Unknown")

            
            
    
    
    
    
