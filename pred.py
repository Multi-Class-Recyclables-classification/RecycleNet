import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Recreate the same ResNet-50 architecture ---
resnet = models.resnet50(weights=None)  # no pretrained weights, since we’ll load our own
num_features = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 7)
)
resnet = resnet.to(device)

# --- Load saved weights ---
weights_path = r"D:\MATERIAL\ml\backend\src\assets\resnet_feature_extractor_7class2.pth"
resnet.load_state_dict(torch.load(weights_path, map_location=device))
resnet.eval()  # set to evaluation mode

# --- Transform for input image ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Load and prepare image ---
image_path = r"D:\MATERIAL\ml\backend\glass_0001.jpg"
image = Image.open(image_path).convert("RGB")
img_tensor = test_transform(image).unsqueeze(0).to(device)  # add batch dimension

# --- Prediction ---
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

print(f"Predicted class: {classes[predicted.item()]}")
