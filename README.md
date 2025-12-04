# SAM3

## Windows DEV Mock步骤

```bash
# 1. 创建虚拟环境（可选，推荐）
python -m venv venv
venv\Scripts\activate

# 2. 安装服务层依赖（不需要安装完整 SAM3 和 CUDA）
pip install -r sam3_service/requirements-service.txt

# 3. 启动服务（开发模式，自动重载）
uvicorn sam3_service.app.main:app --reload --host 127.0.0.1 --port 8000
```

启动后访问：
- **API 文档**：http://127.0.0.1:8000/docs
- **前端页面**：http://127.0.0.1:8000/static/index.html
- **健康检查**：http://127.0.0.1:8000/health

Mock 模式下，上传任意图片会返回一个圆形区域被模糊的效果，用于验证整体流程。


## Linux Deploy Model
```
# 如果你已经有一个满足要求的环境（Python 3.12 + PyTorch 2.7 + CUDA 12.6），
# 可以先检查一下版本，确认无误后直接跳到“# 安装 SAM3 官方包”部分。
# 版本检查示例：
python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('Torch:', torch.__version__)"

# 若当前没有合适环境，按下面步骤创建新环境：
# 创建 conda 环境
conda create -n sam3 python=3.12
conda activate sam3

# 安装 PyTorch + CUDA 12.6
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装 SAM3 官方包（注意：这里的 sam3 是“官方 SAM3 仓库目录”）
# 例如：git clone https://github.com/facebookresearch/sam3.git
# 然后进入该目录：
cd sam3   # 进入官方 SAM3 仓库根目录（包含 pyproject.toml 的那个）
pip install -e .

# 安装服务依赖（在本项目根目录下执行）
pip install -r sam3_service/requirements-service.txt

# HuggingFace 认证
huggingface-cli login

# 启动服务（Real 模式，使用真实 SAM3 模型）
SAM3_MODE=real uvicorn sam3_service.app.main:app --host 0.0.0.0 --port 8000

# 或者先导出环境变量再启动
export SAM3_MODE=real
uvicorn sam3_service.app.main:app --host 0.0.0.0 --port 8000
```

### 模式说明

| 环境变量 | 值 | 说明 |
|---------|-----|------|
| `SAM3_MODE` | `mock`（默认） | 不加载真实模型，返回假的分割结果，用于前端开发调试 |
| `SAM3_MODE` | `real` | 加载真实 SAM3 模型，首次启动会从 HuggingFace 下载权重 |
