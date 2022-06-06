import uvicorn
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse, FileResponse
import numpy as np
import io
from PIL import Image
import torch
import RRDBNet_arch as arch 
import cv2

description=(""" 
# Super Resolution Model API

This API uses GANs to generate an enhanced image from a base image.

A generative adversarial network is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in June 2014. Two neural networks contest with each other in a game. Given a training set, this technique learns to generate new data with the same statistics as the training set.

[GitHub Repository](https://github.com/Nneji123/Super-Resolution-Model)

""")

app = FastAPI(description=description, debug=True)

model_path = 'RRDB_ESRGAN_x4.pth'  
#device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device('cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

@app.post("/")
async def root(file: UploadFile = File(...)):
    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    

    res, im_png = cv2.imencode(".png", output)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

if __name__ == '__main__': 
    uvicorn.run(app)
