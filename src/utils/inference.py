from pydantic import BaseModel 
from dotenv import load_dotenv 
import os 
from torchvision.transforms import transforms
from torchvision import models
import torch.nn as nn 
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resnet = models.resnet50(weights=None)  # no pretrained weights
num_features = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 7)
)
resnet = resnet.to(device)


src_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights_path = os.path.join(src_file, "assets", "resnet_feature_extractor_7class2.pth")
resnet.load_state_dict(torch.load(weights_path, map_location=device))
resnet.eval()


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
