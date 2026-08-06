# Models

The three model files in this directory are the exact weights used by the application. Their SHA256 values are recorded in `../docs/MODEL_SHA256.txt`.

- `yolov8l-worldv2-canvas.pt`: primary canvas detector with precomputed text features.
- `yolov8s-worldv2-canvas.pt`: recovery detector.
- `mobile_sam.pt`: MobileSAM segmentation model.

The application automatically uses this directory when it is present. To use another model directory, set `SMART_CROPPER_MODEL_DIR` before launching.
