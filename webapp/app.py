import base64, io, threading
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
tf = T.Compose([T.Grayscale(num_output_channels=3), T.Resize((224, 224)),
                T.ToTensor(), T.Normalize(MEAN.tolist(), STD.tolist())])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet18_seed1.pt", map_location="cpu"))
model.eval()
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
lock = threading.Lock()  # Grad-CAM hooks are shared, so one request at a time

app = FastAPI()
MAX_BYTES = 5_000_000


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 5 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")

    warnings = []
    arr = np.asarray(img, dtype=np.float32)
    colorfulness = float(np.mean(np.abs(arr[..., 0] - arr[..., 1]) + np.abs(arr[..., 1] - arr[..., 2])))
    w, h = img.size
    if colorfulness > 12:
        warnings.append("This image has strong colors. The model was trained on grayscale EL images, so the result may be meaningless.")
    if max(w, h) / min(w, h) > 1.3:
        warnings.append("This image is not roughly square. The model expects a single cell, so it may be a whole panel or another kind of photo.")

    x = tf(img).unsqueeze(0)
    with lock:
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)[0]
        heat = cam(input_tensor=x, targets=None)[0]

    rgb = np.clip(x[0].permute(1, 2, 0).numpy() * STD + MEAN, 0, 1).astype(np.float32)
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")

    return {
        "healthy": float(p[0]),
        "defective": float(p[1]),
        "label": "defective" if p[1] > p[0] else "healthy",
        "heatmap": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "warnings": warnings,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")