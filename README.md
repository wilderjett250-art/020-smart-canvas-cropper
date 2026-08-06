# SmartCanvasCropper

一个面向 Windows 的智能画布裁图工具。它先用 YOLO-World 定位画布、海报或墙面作品，再用 MobileSAM 细化边缘，最后完成透视校正和批量裁剪。

SmartCanvasCropper is a Windows image utility that detects canvases, posters, framed pictures, and wall art, refines their boundaries, applies perspective correction, and exports clean crops in batches.

## 主要能力 | Features

- 单张图片和文件夹批量处理 | Single-image and batch-folder processing
- YOLO-World 画布候选定位 | YOLO-World candidate detection
- MobileSAM 边缘细化 | MobileSAM boundary refinement
- 透视校正与矩形输出 | Perspective correction and rectangular output
- 保持原比例或指定宽高比 | Preserve the original ratio or choose an aspect ratio
- 中文路径和 Windows 10/11 | Chinese paths and Windows 10/11
- NVIDIA CUDA、AMD/Intel DirectML、CPU 三种运行路线 | NVIDIA CUDA, AMD/Intel DirectML, and CPU modes
- 后台推理线程，界面保持响应 | Background inference workers keep the UI responsive

## 快速运行 | Quick start

要求 Python 3.11 或更高版本。CPU 环境可以直接安装默认依赖；NVIDIA 用户可按照 PyTorch 官方 CUDA 安装方式替换 `torch` 和 `torchvision`。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\smart_canvas_cropper.py
```

运行后选择一张图片或一个文件夹，工具会输出带有 `_裁图` 后缀的结果文件。

## 项目结构 | Project layout

```text
smart_canvas_cropper.py              主程序：检测、分割、透视校正和界面
models/                              三个推理模型
universal_launcher/                  Windows 启动器和环境检查脚本源码
docs/MODEL_SHA256.txt                模型完整性校验
requirements.txt                     CPU/NVIDIA 基础依赖
requirements-directml.txt            AMD/Intel DirectML 可选依赖
```

## 模型说明 | Models

| 文件 | 作用 |
|---|---|
| `yolov8l-worldv2-canvas.pt` | 主定位模型，已预计算画布相关文本特征 |
| `yolov8s-worldv2-canvas.pt` | 主模型没有找到可信候选时使用的恢复模型 |
| `mobile_sam.pt` | 对候选区域进行边缘细化和分割 |

模型的 SHA256 值记录在 [`docs/MODEL_SHA256.txt`](docs/MODEL_SHA256.txt)。模型和第三方组件的来源、许可证说明见 [`universal_launcher/THIRD_PARTY_NOTICES.txt`](universal_launcher/THIRD_PARTY_NOTICES.txt)。

## 运行路线 | Runtime modes

- `auto`：优先使用可用的 NVIDIA CUDA，其次尝试 DirectML，最后使用 CPU。
- `nvidia`：适合安装 CUDA 版 PyTorch 的 NVIDIA 电脑。
- `directml`：适合 AMD 或 Intel GPU，需要额外安装 `requirements-directml.txt`。
- `cpu`：适合没有可用 GPU 的电脑，速度较慢但流程一致。

便携 Python、完整依赖和双击启动 EXE 属于发布包，不放入源码仓库。这样仓库保持适合阅读和二次开发，普通用户版本可以通过 GitHub Releases 获取。

## 许可证与第三方组件 | Licensing

本项目集成 Ultralytics、YOLO-World 和 MobileSAM。使用、修改或重新分发时，请阅读 [`THIRD_PARTY_NOTICES.txt`](universal_launcher/THIRD_PARTY_NOTICES.txt) 中的原始许可证要求，尤其注意 Ultralytics 组件的 AGPL-3.0 或商业许可边界。

## English summary

SmartCanvasCropper turns a photo of a canvas or poster into a clean, corrected crop. The pipeline combines open-vocabulary detection, segmentation, geometry refinement, and batch export. The repository contains the application source, launcher source, model weights, checksums, and dependency definitions; bundled Python runtimes and generated experiment folders are kept out of the source tree.
