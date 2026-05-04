import torch
from torchvision.models.optical_flow import Raft_Large_Weights
transforms = Raft_Large_Weights.DEFAULT.transforms()
img1 = torch.randint(0, 256, (1, 3, 576, 768), dtype=torch.uint8)
img2 = torch.randint(0, 256, (1, 3, 576, 768), dtype=torch.uint8)
try:
    img1, img2 = transforms(img1, img2)
    print("uint8 works")
except Exception as e:
    print(f"Error: {e}")
    
img1 = torch.rand((1, 3, 576, 768), dtype=torch.float32) * 255
img2 = torch.rand((1, 3, 576, 768), dtype=torch.float32) * 255
try:
    img1, img2 = transforms(img1, img2)
    print("float32 [0, 255] works")
except Exception as e:
    print(f"Error: {e}")
