from fastapi import FastAPI,UploadFile,File
from fastapi.responses import Response
from runners.predictor import Predictor
from options import options
from utils.logger import create_logger


def predict_pipeline():
    options.dataset.name += '_demo'
    logger = create_logger(options, phase='predict')
    predictor = Predictor(options, logger)
    predictor.predict()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(image_file:UploadFile = File(...)):
    with open("tmp/imagePredict.png","wb") as buffer:
        buffer.write(await image_file.read())
    predict_pipeline()
    with open("tmp/imagePredict.obj", "rb") as f:
        predicted = f.read()

    return Response(content=predicted, media_type="application/octet-stream")
