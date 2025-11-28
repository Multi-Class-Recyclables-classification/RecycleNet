from fastapi import FastAPI, UploadFile, File, HTTPException
from src.utils.config import prepare_image, predict
from PIL import Image
import io

app = FastAPI()


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def validate_image(image_bytes: bytes) -> tuple[bool, str]:
    """Validate uploaded image file."""
    if len(image_bytes) > MAX_FILE_SIZE:
        return False, f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit"
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.format.lower() not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS}"
        image.verify()
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
   
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    is_valid, message = validate_image(image_bytes)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    try:
        img_tensor = prepare_image(image_bytes)
        result = predict(img_tensor)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
