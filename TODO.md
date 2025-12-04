
# SAM3 服务开发 TODO

- [x] **基础架构与环境**
  - [x] 基于官方 `sam3` 仓，在仓库中新增 `sam3_service/` 目录，不修改核心模型代码
  - [x] 在服务层新增依赖清单（如 `requirements-service.txt`），包含 FastAPI / Uvicorn / Pillow 等
  - [x] 规划模型权重目录 `models/sam3/`，并下载/放置默认 SAM3 权重（单模型、单 GPU）

- [x] **FastAPI 服务骨架**
  - [x] 创建 `sam3_service/app/main.py`，初始化 FastAPI 应用
  - [x] 实现基础路由 `/health`，返回模型加载状态、设备信息等
  - [ ] 配置 Uvicorn 启动命令（开发模式）

- [x] **核心模块骨架**
  - [x] `sam3_service/app/core/config.py`：设备、模型路径、图像尺寸、阈值配置
  - [x] `sam3_service/app/core/sam3_model.py`：封装 SAM3 模型加载为单例（暂时可用 mock）
  - [x] `sam3_service/app/core/image_io.py`：图片文件 / base64 编解码工具
  - [x] `sam3_service/app/core/pipeline_privacy.py`：隐私过滤流水线（mode=auto，先写接口再补实现）

- [x] **API 层骨架**
  - [x] `sam3_service/app/api/v1/segmentation.py`：定义 `/v1/segment/auto` 与 `/v1/segment/prompt` 接口结构
  - [x] `sam3_service/app/api/v1/privacy.py`：定义 `/v1/privacy/filter` 接口（mode=auto）
  - [x] `sam3_service/app/api/health.py`：健康检查接口

- [ ] **隐私过滤第一版实现（mode=auto）**
  - [ ] 在 `sam3_model.py` 中对接官方 SAM3 推理接口，实现自动分割调用
  - [ ] 在 `pipeline_privacy.py` 中基于自动分割结果，按最小面积比例等规则筛选 mask
  - [ ] 实现 `gaussian / pixelate / solid` 三种基础模糊/遮挡方式
  - [ ] `/v1/privacy/filter` 返回带遮挡后的图片（base64）和应用区域信息

- [x] **极简 HTML 前端**
  - [x] 在 `sam3_service/static/index.html` 中实现最小页面：图片上传 + 参数选择 + 结果预览
  - [x] 在 `sam3_service/static/js/main.js` 中封装对 `/v1/privacy/filter` 的调用逻辑
  - [ ] 本地联调前端与 FastAPI 服务，完成端到端验证

- [ ] **Gradio Demo（可选但推荐）**
  - [ ] 新增 `sam3_service/demos/gradio_demo.py`，直接调用 `sam3_model` 与 `pipeline_privacy`
  - [ ] 用 Gradio 提供本地交互界面，方便调参与效果验证

- [ ] **部署与优化（后续）**
  - [ ] 针对单 GPU 环境优化加载与推理（如半精度、图像尺寸限制）
  - [ ] 视需要增加多模型/多实例支持与模型热切换能力
  - [ ] 补充 README：接口说明、部署方式和前端接入示例

