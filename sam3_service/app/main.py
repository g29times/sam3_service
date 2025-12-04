"""
SAM3 服务入口
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .core.config import STATIC_DIR
from .core.sam3_model import sam3_model
from .api.health import router as health_router
from .api.v1.segmentation import router as segmentation_router
from .api.v1.privacy import router as privacy_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    print("[Startup] Loading SAM3 model...")
    sam3_model.load()
    print("[Startup] Model loaded.")
    yield
    # 关闭时清理（如有需要）
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="SAM3 Service",
    description="基于 SAM3 的图像分割与隐私过滤服务",
    version="0.1.0",
    lifespan=lifespan,
)

# 注册路由
app.include_router(health_router)
app.include_router(segmentation_router, prefix="/v1")
app.include_router(privacy_router, prefix="/v1")

# 静态文件服务（前端页面）
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """根路径重定向提示"""
    return {
        "message": "SAM3 Service is running",
        "docs": "/docs",
        "health": "/health",
        "static": "/static/index.html",
    }
