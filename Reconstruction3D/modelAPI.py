from fastapi import FastAPI,UploadFile,File
from fastapi.responses import Response
from starlette.middleware.cors import CORSMiddleware

from runners.predictor import Predictor
from options import options
from utils.logger import create_logger

options.dataset.name = "single_image"

def predict_pipeline():
    logger = create_logger(options, phase='predict')
    predictor = Predictor(options, logger)
    predictor.predict()

app = FastAPI()
app.add_middleware(CORSMiddleware,
        allow_origins=["http://localhost:4200"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(image_file:UploadFile = File(...)):
    with open("predictions/imagePredict.png","wb") as buffer:
        buffer.write(await image_file.read())
    predict_pipeline()
    with open("predictions/imagePredict.obj", "rb") as f:
        predicted = f.read()

    return Response(content=predicted, media_type="application/octet-stream", headers={"Content-Disposition": "attachment;filename=predicted.obj"})
