from api.routes.ais import router as ais_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.detection import router as detection_router
from api.routes.hindcast import router as hindcast_router


app = FastAPI(
    title="Oil Spill Detection API",
    description="SIH 2026 Oil Spill Detection and Environmental Intelligence Backend",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "oil-spill-detection-api",
    }


app.include_router(detection_router)
app.include_router(hindcast_router)
app.include_router(ais_router)