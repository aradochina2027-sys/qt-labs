import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.detector import ObjectDetector

app = FastAPI(title="AI Local Object Detection API")
templates = Jinja2Templates(directory="app/templates")

detector = ObjectDetector(model_path="yolov8n.pt")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )


@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Некоректний тип файлу. Завантажте зображення."
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Файл порожній.")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Не вдалося обробити зображення.")

    results = detector.detect(image=image, conf_threshold=conf_threshold)
    return JSONResponse(content=results)