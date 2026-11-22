import sys
import torch
sys.path.insert(0, "src")
from models import get_model

m = get_model("resnet18")
m.load_state_dict(torch.load("outputs/resnet18_seed1.pt", map_location="cpu"))
m.eval()
torch.onnx.export(m, torch.randn(1, 3, 224, 224), "frontend/model.onnx",
                  input_names=["input"], output_names=["logits"],
                  opset_version=17, dynamo=False)
print("saved frontend/model.onnx")
