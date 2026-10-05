import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"
weights = Raft_Large_Weights.DEFAULT
model = raft_large(weights=weights, progress=False).to(device)
model.eval()

img1 = torch.rand(1, 3, 576, 768).to(device)
img2 = torch.rand(1, 3, 576, 768).to(device)

# preprocess
transforms = weights.transforms()
img1, img2 = transforms(img1, img2)

with torch.no_grad():
    list_of_flows = model(img1, img2)
    predicted_flow = list_of_flows[-1] # shape [1, 2, H, W]

flow_mag = torch.norm(predicted_flow, dim=1) # [1, H, W]
print(f"Flow magnitude shape: {flow_mag.shape}")
