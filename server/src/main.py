from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from typing import List, Literal, Optional, Tuple
import io
import numpy as np
from yolo import object_detection
from framework3d import complete_3d_object_detection
import cv2

app = FastAPI()

# Example parameter schema
class ImageParams2D(BaseModel):
    model: Literal["nano", "medium", "large"] = "nano"
    device: Literal["cpu", "cuda"] = "cpu"

@app.post("/2d-object-detection/")
async def object_detection_2d(
    file: UploadFile = File(...),
    params: str = Form(...)
):
    # Parse JSON params
    try:
        parsed_params = ImageParams2D.model_validate_json(params)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=e.errors()
        )

    # Read image
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    image_with_boxes = object_detection(image, parsed_params.model, parsed_params.device)

    _, buffer = cv2.imencode(".jpg", image_with_boxes)
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpg"
    )


# Example parameter schema
class ImageParams3D(BaseModel):
    threshold: Optional[float] = 0.25
    focal_length: Optional[float] = 0
    principal_point: Optional[Tuple[float, float]] = []
    model: Literal["DLA", "Res"] = "Res"
    device: Literal["cpu", "cuda"] = "cpu"
    config_file: Optional[str] = None
    opts: Optional[List[str]] = None

    def model_post_init(self, __context):
        if self.model == "DLA":
            self.config_file = "cubercnn://omni3d/cubercnn_DLA34_FPN.yaml"
            self.opts=['MODEL.WEIGHTS', 'cubercnn://omni3d/cubercnn_DLA34_FPN.pth', 'MODEL.DEVICE', self.device]
        elif self.model == "Res":
            self.config_file = "cubercnn://omni3d/cubercnn_Res34_FPN.yaml"
            self.opts=['MODEL.WEIGHTS', 'cubercnn://omni3d/cubercnn_Res34_FPN.pth', 'MODEL.DEVICE', self.device]

@app.post("/3d-object-detection/")
async def object_detection_3d(
    file: UploadFile = File(...),
    params: str = Form(...)
):
    # Parse JSON params
    try:
        parsed_params = ImageParams3D.model_validate_json(params)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=e.errors()
        )

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image_with_boxes = complete_3d_object_detection(parsed_params, image)

    if type(image_with_boxes) == type(False) and image_with_boxes == False:
        image_with_boxes = image


    _, buffer = cv2.imencode(".jpg", image_with_boxes)
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpg"
    )

if __name__ == "__main__":
    import uvicorn

    print("Server running")

    uvicorn.run(
        "main:app",  # filename:variable
        host="0.0.0.0",
        port=8000,
        reload=True
    )

# Run with:
# uvicorn main:app --reload
# or simply
# python src/main.py

# Example curl:
# curl -X POST "http://localhost:8000/process-image/" \
#   -F "file=@input.png" \
#   -F 'params={"text":"Hello","x":50,"y":50}' \
#   --output output.png

# Test url
# http://localhost:8000/docs