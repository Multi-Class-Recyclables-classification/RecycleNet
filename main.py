from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.utils.config import prepare_image, predict
from PIL import Image
import io
import os

app = FastAPI(title="RecycleNet API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the frontend HTML page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RecycleNet - AI Waste Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            color: #666;
            font-size: 1.1em;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #d4edda;
            color: #155724;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-top: 10px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #28a745;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
            background: #f8f9ff;
        }

        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }

        .upload-area.dragover {
            border-color: #764ba2;
            background: #e8ebff;
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }

        .upload-text {
            color: #667eea;
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .upload-subtext {
            color: #999;
            font-size: 0.9em;
        }

        #fileInput {
            display: none;
        }

        .preview-container {
            display: none;
            margin-bottom: 20px;
        }

        .preview-container.active {
            display: block;
        }

        .preview-image {
            width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 10px;
            margin-bottom: 15px;
            border: 2px solid #e0e0e0;
        }

        .btn-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .btn {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-secondary {
            background: #e0e0e0;
            color: #666;
        }

        .btn-secondary:hover {
            background: #d0d0d0;
        }

        .result-container {
            display: none;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            animation: slideIn 0.3s ease;
        }

        .result-container.active {
            display: block;
        }

        .result-container.success {
            background: #d4edda;
            border: 2px solid #28a745;
        }

        .result-container.error {
            background: #f8d7da;
            border: 2px solid #dc3545;
        }

        .result-icon {
            font-size: 4em;
            margin-bottom: 15px;
            animation: bounceIn 0.5s ease;
        }

        .result-label {
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 10px;
            text-transform: capitalize;
            color: #2d3748;
        }

        .result-description {
            color: #666;
            line-height: 1.6;
            font-size: 1em;
        }

        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }

        .loading.active {
            display: block;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes bounceIn {
            0% { transform: scale(0); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }

        .category-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-top: 30px;
            padding-top: 30px;
            border-top: 2px solid #e0e0e0;
        }

        .category-item {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            background: #f8f9ff;
            transition: all 0.3s ease;
        }

        .category-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        }

        .category-emoji {
            font-size: 2.2em;
            margin-bottom: 8px;
        }

        .category-name {
            font-size: 0.9em;
            color: #667eea;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>♻️ RecycleNet</h1>
            <p>AI-Powered Waste Classification</p>
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>Connected to AI Model</span>
            </div>
        </div>

        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📸</div>
            <div class="upload-text">Click or drag image here</div>
            <div class="upload-subtext">Supports: PNG, JPG, JPEG, WEBP, BMP (max 10MB)</div>
        </div>

        <input type="file" id="fileInput" accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp">

        <div class="preview-container" id="previewContainer">
            <img id="previewImage" class="preview-image" alt="Preview">
            <div class="btn-group">
                <button class="btn btn-primary" id="classifyBtn">
                    🔍 Classify Image
                </button>
                <button class="btn btn-secondary" id="resetBtn">
                    🔄 Choose Another
                </button>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>🤖 Analyzing with ResNet50 model...</p>
        </div>

        <div class="result-container" id="resultContainer">
            <div class="result-icon" id="resultIcon"></div>
            <div class="result-label" id="resultLabel"></div>
            <div class="result-description" id="resultDescription"></div>
        </div>

        <div class="category-info">
            <div class="category-item">
                <div class="category-emoji">🍂</div>
                <div class="category-name">Biological</div>
            </div>
            <div class="category-item">
                <div class="category-emoji">📦</div>
                <div class="category-name">Cardboard</div>
            </div>
            <div class="category-item">
                <div class="category-emoji">🍷</div>
                <div class="category-name">Glass</div>
            </div>
            <div class="category-item">
                <div class="category-emoji">🔩</div>
                <div class="category-name">Metal</div>
            </div>
            <div class="category-item">
                <div class="category-emoji">📄</div>
                <div class="category-name">Paper</div>
            </div>
            <div class="category-item">
                <div class="category-emoji">🥤</div>
                <div class="category-name">Plastic</div>
            </div>
            <div class="category-item">
                <div class="category-emoji">🗑️</div>
                <div class="category-name">Trash</div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        const classifyBtn = document.getElementById('classifyBtn');
        const resetBtn = document.getElementById('resetBtn');
        const loading = document.getElementById('loading');
        const resultContainer = document.getElementById('resultContainer');
        const resultIcon = document.getElementById('resultIcon');
        const resultLabel = document.getElementById('resultLabel');
        const resultDescription = document.getElementById('resultDescription');

        let selectedFile = null;

        const categoryInfo = {
            biological: {
                icon: '🍂',
                description: 'Organic waste including food scraps, garden waste, and biodegradable materials. Should be composted.'
            },
            cardboard: {
                icon: '📦',
                description: 'Recyclable cardboard boxes, packaging materials, and paper-based containers. Flatten before recycling.'
            },
            glass: {
                icon: '🍷',
                description: 'Glass bottles, jars, and containers that can be recycled indefinitely. Remove lids before recycling.'
            },
            metal: {
                icon: '🔩',
                description: 'Metal cans, foils, and other metallic items suitable for recycling. Rinse before recycling.'
            },
            paper: {
                icon: '📄',
                description: 'Paper products including newspapers, magazines, office paper, and books. Keep dry for recycling.'
            },
            plastic: {
                icon: '🥤',
                description: 'Plastic bottles, containers, and packaging materials for recycling. Check recycling number.'
            },
            trash: {
                icon: '🗑️',
                description: 'General waste that cannot be recycled or composted. Dispose in regular trash bin.'
            }
        };

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }

            if (file.size > 10 * 1024 * 1024) {
                alert('File size exceeds 10MB limit');
                return;
            }

            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewContainer.classList.add('active');
                resultContainer.classList.remove('active');
            };
            reader.readAsDataURL(file);
        }

        classifyBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            loading.classList.add('active');
            resultContainer.classList.remove('active');
            classifyBtn.disabled = true;

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch('/classify', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    displayResult(data.prediction, true);
                } else {
                    displayResult(data.detail || 'Classification failed', false);
                }
            } catch (error) {
                console.error('Classification error:', error);
                displayResult('Connection error. Please try again.', false);
            } finally {
                loading.classList.remove('active');
                classifyBtn.disabled = false;
            }
        });

        resetBtn.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            previewContainer.classList.remove('active');
            resultContainer.classList.remove('active');
        });

        function displayResult(prediction, success) {
            resultContainer.classList.add('active');
            
            if (success) {
                resultContainer.classList.remove('error');
                resultContainer.classList.add('success');
                const info = categoryInfo[prediction.toLowerCase()] || categoryInfo.trash;
                resultIcon.textContent = info.icon;
                resultLabel.textContent = prediction;
                resultDescription.textContent = info.description;
            } else {
                resultContainer.classList.remove('success');
                resultContainer.classList.add('error');
                resultIcon.textContent = '❌';
                resultLabel.textContent = 'Error';
                resultDescription.textContent = prediction;
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Classify uploaded waste image."""
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

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "ResNet50", "classes": 7}