智能裁图工具 v1.9.2 通用正式版

一、推荐启动方式

1. 将 ZIP 完整解压到任意文件夹，中文路径可以正常使用。
2. 双击“命令行检查并启动.cmd”。
3. 命令行会先检查发布包、Windows、显卡、驱动、磁盘空间和依赖源，再选择推理后端：
   - NVIDIA 显卡：优先 CUDA GPU；CUDA 验证未通过时自动尝试 DirectML。
   - AMD 显卡：DirectML GPU。
   - Intel Arc、Iris、UHD 等显卡：DirectML GPU。
   - GPU 环境安装、驱动或设备验证异常：自动安装并切换 CPU 推理环境。
4. 下载时会显示 pip 的包名、已下载大小、总大小、进度和速度。每个网络连接或读取阶段明确设置 30 秒超时；当前来源失败后会自动切换下一来源。
5. 推理环境验证完成后保存在本机，后续运行直接复用；已有缓存的普通启动会跳过依赖源探测。

也可以双击 SmartCanvasCropper.exe 使用图形化配置窗口。模型预加载、单图定位和批量推理都在后台执行，界面在推理期间仍可移动和操作。

定位采用 YOLOv8l-Worldv2 主模型；仅当主模型没有找到可信画布时，才按需加载 YOLOv8s-Worldv2 恢复模型，以兼顾复杂广告模板和小型墙面画框。

二、命令行用法

命令行脚本位于 tools\setup_and_run.ps1，不需要管理员权限。

只检查，不修改电脑环境：
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\tools\setup_and_run.ps1" -Action doctor -Mode auto

安装并验证环境，但不启动程序：
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\tools\setup_and_run.ps1" -Action prepare -Mode auto

使用已验证环境启动；没有可用缓存时自动准备 CPU 环境：
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\tools\setup_and_run.ps1" -Action run -Mode auto

检查、安装、验证并启动：
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\tools\setup_and_run.ps1" -Action prepare-run -Mode auto

Mode 支持 auto、nvidia、directml、cpu。ForceReinstall 可以重建选定推理环境；旧环境会改名为带时间戳的备份，不会直接删除。

三、首次运行空间与网络

- NVIDIA CUDA 模式首次约需下载 2.5–3 GB，建议预留至少 6 GB 可用空间。
- AMD / Intel DirectML 模式首次约需下载 300–500 MB，建议预留至少 2 GB 可用空间。
- CPU 模式首次约需下载 250–450 MB，建议预留至少 2 GB 可用空间。
- Python 3.11、pip 与 Tk GUI 基础运行时已经包含在 ZIP 内；配置工具复制包内运行时，不调用系统 Python 安装器。
- 推理环境默认保存在：%LOCALAPPDATA%\SmartCanvasCropper\v1.9.2
- 配置与应用日志保存在：%LOCALAPPDATA%\SmartCanvasCropper\v1.9.2\logs
- 不修改、不注册也不依赖系统 Python，不需要管理员权限。

四、图片与文件夹

- 可以选择单张图片，也可以选择文件夹批量处理。
- 支持 JPG、JPEG、PNG、WebP、BMP、TIFF。
- 输入路径、输出路径和 Windows 用户名支持中文。
- 默认保持检测画面的宽高比，也可以选择或输入固定宽高比。
- 边缘净化、分割排序和四角定位使用正式版固定参数；界面不提供专业微调，选择图片后直接自动定位即可。
- 输出文件保留原文件名并追加“_裁图”，不会覆盖原图。
- 置信度不足的图片进入“需要复核原图”文件夹，CSV 结果清单会记录原因。

五、兼容范围

- Windows 10 / Windows 11，64 位 x86 电脑。
- 建议至少 8 GB 内存；CPU 模式兼容范围最广，首次加载和批量处理速度会低于 GPU。
- NVIDIA 使用 CUDA，AMD 与 Intel 使用 DirectML；所有模式使用相同定位、透视校正和输出逻辑。
- 依赖安装使用官方 HTTPS 来源和国内 HTTPS 镜像自动切换，并隔离电脑原有的 pip.ini。
- Windows 7、32 位 Windows、Windows ARM、macOS 和 Linux不属于此 Windows 发布包的运行范围。

六、模型与许可证

程序使用免训练的公开预训练模型：Ultralytics YOLO-World v2 与 MobileSAM。完整来源与许可证说明见 THIRD_PARTY_NOTICES.txt；应用源文件保存在 app 文件夹中。
