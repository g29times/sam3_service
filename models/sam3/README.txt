SAM3 模型权重说明
==================

SAM3 模型通过 HuggingFace 自动下载，无需手动放置权重文件。

使用前准备：
1. 申请访问权限：https://huggingface.co/facebook/sam3
2. 安装 huggingface-cli：pip install huggingface_hub
3. 登录认证：huggingface-cli login

模型信息：
- HuggingFace 仓库：facebook/sam3
- 模型参数量：848M
- 首次运行时会自动下载到 HuggingFace 缓存目录

环境要求：
- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+（推荐 Linux 环境）

官方仓库：https://github.com/facebookresearch/sam3
