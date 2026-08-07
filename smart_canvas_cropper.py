"""Local zero-training cropper for photos that contain a canvas or poster.

The tool uses pre-trained YOLO-World and MobileSAM models, then applies a
locked, benchmarked geometry pipeline so end users do not need to tune expert
parameters or move corner handles.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import queue
import re
import shutil
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL_ROOT = _PROJECT_ROOT / "models"
ROOT = Path(
    os.environ.get(
        "SMART_CROPPER_MODEL_DIR",
        _DEFAULT_MODEL_ROOT if _DEFAULT_MODEL_ROOT.is_dir() else _PROJECT_ROOT,
    )
).resolve()
APP_VERSION = "1.9.2"
EDGE_TRIM_PERCENT = 0.8
WEIGHTS = ROOT / "yolov8l-worldv2-canvas.pt"
RECOVERY_WEIGHTS = ROOT / "yolov8s-worldv2-canvas.pt"
SEGMENT_WEIGHTS = ROOT / "mobile_sam.pt"
DETECTOR_CLASSES = [
    "a large canvas print", "a large poster", "a framed photograph",
    "a painting", "wall art", "a picture", "a smartphone", "",
]
CANVAS_SIZE = (820, 650)
PREVIEW_SIZE = (360, 500)
RUNTIME_MODE = os.environ.get("SMART_CROPPER_RUNTIME_MODE", "auto").strip().lower()
SEGMENT_CACHE_POLICY = "reset"
SEGMENT_REFINEMENT_MODE = "ranked"
_DIRECTML_PATCHED = False


def _create_app_logger() -> tuple[logging.Logger, Path | None]:
    logger = logging.getLogger("smart_canvas_cropper")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        existing = getattr(logger.handlers[0], "baseFilename", None)
        return logger, Path(existing) if existing else None

    configured = os.environ.get("SMART_CROPPER_LOG_DIR", "").strip()
    if configured:
        log_dir = Path(configured)
    else:
        local_data = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        log_dir = local_data / "SmartCanvasCropper" / f"v{APP_VERSION}" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"app-{datetime.now():%Y%m%d}.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger, log_path
    except OSError:
        logger.addHandler(logging.NullHandler())
        return logger, None


LOGGER, APP_LOG_PATH = _create_app_logger()


def _move_torch_value_to_cpu(value):
    """Move nested torch outputs to CPU before unsupported DirectML post-processing."""
    import torch

    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, list):
        return [_move_torch_value_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_torch_value_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _move_torch_value_to_cpu(item) for key, item in value.items()}
    return value


def configure_directml_backend():
    """Enable Ultralytics on DirectML while retaining GPU-heavy model stages."""
    global _DIRECTML_PATCHED

    import torch
    import torch_directml

    device = torch_directml.device()
    if _DIRECTML_PATCHED:
        return device

    # torch-directml 0.2.5 cannot execute some tensors created by
    # inference_mode(). no_grad() still disables autograd without those tensor
    # semantics and is safe for local inference.
    torch.inference_mode = torch.no_grad

    from ultralytics.models.sam.predict import Predictor as SAMPredictor
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.nn.autobackend import AutoBackend

    if not getattr(AutoBackend, "_smart_cropper_directml_patch", False):
        def skip_directml_warmup(self, *args, **kwargs):
            return None

        AutoBackend.warmup = skip_directml_warmup
        AutoBackend._smart_cropper_directml_patch = True

    if not getattr(DetectionPredictor, "_smart_cropper_directml_patch", False):
        original_detection_postprocess = DetectionPredictor.postprocess

        def cpu_detection_postprocess(self, preds, img, orig_imgs, **kwargs):
            return original_detection_postprocess(
                self, _move_torch_value_to_cpu(preds), img, orig_imgs, **kwargs
            )

        DetectionPredictor.postprocess = cpu_detection_postprocess
        DetectionPredictor._smart_cropper_directml_patch = True

    if not getattr(SAMPredictor, "_smart_cropper_directml_patch", False):
        original_sam_inference_features = SAMPredictor._inference_features

        def hybrid_sam_inference_features(
            self,
            features,
            bboxes=None,
            points=None,
            labels=None,
            masks=None,
            multimask_output=False,
        ):
            # The image encoder is MobileSAM's expensive stage and remains on
            # DirectML. The compact prompt and mask heads run on CPU to avoid
            # DirectML operators unsupported by torch-directml 0.2.5.
            self.model.prompt_encoder.cpu()
            self.model.mask_decoder.cpu()
            return original_sam_inference_features(
                self,
                _move_torch_value_to_cpu(features),
                bboxes=_move_torch_value_to_cpu(bboxes),
                points=_move_torch_value_to_cpu(points),
                labels=_move_torch_value_to_cpu(labels),
                masks=_move_torch_value_to_cpu(masks),
                multimask_output=multimask_output,
            )

        SAMPredictor._inference_features = hybrid_sam_inference_features
        SAMPredictor._smart_cropper_directml_patch = True

    _DIRECTML_PATCHED = True
    return device


def choose_compute_device() -> tuple[object, str]:
    """Use the backend selected by the hardware-aware launcher."""
    try:
        import torch

        if RUNTIME_MODE in {"directml", "dml"}:
            device = configure_directml_backend()
            probe = torch.ones(1).to(device).cpu()
            if float(probe[0]) == 1.0:
                return device, "GPU：DirectML（AMD / Intel / NVIDIA）"
        if RUNTIME_MODE not in {"cpu", "directml", "dml"} and torch.cuda.is_available():
            return 0, f"GPU: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "cpu", "CPU（GPU 后端不可用，已自动切换）" if RUNTIME_MODE != "cpu" else "CPU"


def read_image_file(path: str | Path) -> tuple[np.ndarray | None, str]:
    """Decode an image through Unicode-safe Windows paths with a Pillow fallback."""
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        return None, f"无法访问文件：{error}"
    if size <= 0:
        return None, "文件大小为 0 字节"

    errors: list[str] = []
    try:
        encoded = np.fromfile(str(source), dtype=np.uint8)
        if encoded.size:
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is not None:
                return image, ""
        errors.append("OpenCV无法识别图片内容")
    except (OSError, ValueError, cv2.error) as error:
        errors.append(f"OpenCV读取失败：{error}")

    try:
        with Image.open(source) as pil_image:
            rgb = ImageOps.exif_transpose(pil_image).convert("RGB")
            image = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(image), ""
    except Exception as error:
        errors.append(f"Pillow读取失败：{error}")

    suffix = source.suffix.lower() or "无扩展名"
    return None, f"扩展名 {suffix}；" + "；".join(errors)


def write_image_file(
    path: str | Path,
    image: np.ndarray,
    params: list[int] | None = None,
) -> tuple[bool, str]:
    """Encode and write an image through Unicode-safe Windows paths."""
    target = Path(path)
    suffix = target.suffix.lower()
    extension = ".png" if suffix == ".png" else ".jpg"
    encode_params = params
    if encode_params is None and extension == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    try:
        ok, encoded = cv2.imencode(extension, image, encode_params or [])
        if not ok:
            return False, f"无法编码为 {extension} 图片"
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded.tofile(str(target))
        return True, ""
    except (OSError, ValueError, cv2.error) as error:
        return False, str(error)


def order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    total = points.sum(axis=1)
    delta = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [points[np.argmin(total)], points[np.argmin(delta)],
         points[np.argmax(total)], points[np.argmax(delta)]],
        dtype=np.float32,
    )


def warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = order_corners(corners)
    width = max(2, round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = max(2, round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), target)
    return cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_CUBIC)


def clean_edges(image: np.ndarray, percent: float) -> np.ndarray:
    """Remove a controlled, even rim after perspective correction."""
    height, width = image.shape[:2]
    inset = int(round(min(width, height) * percent / 100))
    if inset <= 0 or width <= inset * 2 + 2 or height <= inset * 2 + 2:
        return image
    return image[inset:height - inset, inset:width - inset].copy()


def parse_aspect_ratio(value: str | None) -> float | None:
    """Parse an optional W:H target ratio; empty/original keeps the detected ratio."""
    text = "" if value is None else str(value).strip().lower()
    if not text or text in {"不设置", "原始比例", "原图比例", "original", "auto", "none"}:
        return None
    parts = [part for part in re.split(r"\s*[:x/]\s*", text) if part]
    if len(parts) == 2:
        try:
            width, height = float(parts[0]), float(parts[1])
        except ValueError as error:
            raise ValueError("比例格式应为 宽:高，例如 4:3 或 9:16。") from error
        if width <= 0 or height <= 0:
            raise ValueError("比例中的宽和高必须大于 0。")
        return width / height
    try:
        ratio = float(text)
    except ValueError as error:
        raise ValueError("比例格式应为 宽:高，例如 4:3 或 9:16。") from error
    if ratio <= 0:
        raise ValueError("目标比例必须大于 0。")
    return ratio


def crop_to_aspect(image: np.ndarray, ratio: float | None) -> np.ndarray:
    """Center-crop a corrected image to an optional ratio without stretching."""
    if ratio is None:
        return image
    height, width = image.shape[:2]
    current = width / max(height, 1)
    if abs(current - ratio) < 1e-4:
        return image
    if current > ratio:
        target_width = max(2, min(width, round(height * ratio)))
        left = max(0, (width - target_width) // 2)
        return image[:, left:left + target_width].copy()
    target_height = max(2, min(height, round(width / ratio)))
    top = max(0, (height - target_height) // 2)
    return image[top:top + target_height, :].copy()


def mask_to_quad(polygon: np.ndarray, expected_area: float) -> np.ndarray | None:
    """Fit a reliable canvas quadrilateral to a SAM mask outline."""
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(polygon)
    perimeter = cv2.arcLength(hull, True)
    for epsilon in (0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.040):
        candidate = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
        if len(candidate) != 4 or not cv2.isContourConvex(candidate.reshape(-1, 1, 2)):
            continue
        quad = order_corners(candidate)
        top, right, bottom, left = (
            np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]),
            np.linalg.norm(quad[3] - quad[2]), np.linalg.norm(quad[0] - quad[3]),
        )
        area = abs(cv2.contourArea(quad.reshape(-1, 1, 2)))
        # A mask that captures only part of the print creates a highly uneven quad.
        if min(top, right, bottom, left) < 24 or max(top, bottom) / min(top, bottom) > 1.7:
            continue
        if max(left, right) / min(left, right) > 1.65:
            continue
        if not 0.42 * expected_area <= area <= 2.1 * expected_area:
            continue
        return quad
    return None


class SmartCropper(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"画布与海报智能裁图 {APP_VERSION} 正式版")
        self.minsize(1240, 760)
        self.configure(bg="#f4f6f8")
        self.model = None
        self.recovery_model = None
        self.segmenter = None
        self._model_lock = threading.RLock()
        self._ui_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.model_preloading = False
        self.inference_running = False
        self.batch_running = False
        self.compute_device, self.compute_label = choose_compute_device()
        self.image_path: Path | None = None
        self.original: np.ndarray | None = None
        self.corners: np.ndarray | None = None
        self.preview: np.ndarray | None = None
        self.raw_preview: np.ndarray | None = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.detection_ready = False
        self.canvas_photo: ImageTk.PhotoImage | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_detection_mode = ""
        self.last_detection_review_reason = ""
        self.status = tk.StringVar(value="请选择一张包含画布、海报或装裱照片的场景图。")
        self.result_info = tk.StringVar(value="裁切后会在这里显示横竖方向与宽高比。")
        self.aspect_ratio = tk.StringVar(value="不设置")
        self.compute_label_var = tk.StringVar(value=f"当前推理：{self.compute_label}。")
        self._build_ui()
        self.after(75, self._drain_ui_events)
        self.after(250, self._start_model_preload)
        LOGGER.info("application_started runtime_mode=%s compute=%s", RUNTIME_MODE, self.compute_label)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=(16, 12))
        toolbar.pack(fill="x")
        self.select_button = ttk.Button(toolbar, text="选择图片", command=self.load_image)
        self.select_button.pack(side="left")
        self.auto_button = ttk.Button(toolbar, text="自动定位画布", command=self.auto_locate)
        self.auto_button.pack(side="left", padx=(8, 0))
        self.export_button = ttk.Button(toolbar, text="导出裁切图", command=self.export, state="disabled")
        self.export_button.pack(side="left", padx=(8, 0))
        ttk.Label(toolbar, text="输出比例").pack(side="left", padx=(20, 4))
        ratio_box = ttk.Combobox(
            toolbar,
            textvariable=self.aspect_ratio,
            values=("不设置", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"),
            width=9,
        )
        ratio_box.pack(side="left")
        ratio_box.bind("<<ComboboxSelected>>", lambda _event: self.redraw())
        ratio_box.bind("<Return>", lambda _event: self.redraw())
        ratio_box.bind("<FocusOut>", lambda _event: self.redraw())
        ttk.Label(toolbar, textvariable=self.compute_label_var, foreground="#4b5563").pack(side="left", padx=18)

        self.batch_button = ttk.Button(toolbar, text="批量处理文件夹", command=self.batch_folder)
        self.batch_button.pack(side="left", padx=(12, 0))
        body = ttk.Frame(self, padding=(16, 0, 16, 12))
        body.pack(fill="both", expand=True)
        left = ttk.Labelframe(body, text="原图与自动定位边界", padding=8)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Labelframe(body, text="透视校正后的裁切结果", padding=8)
        right.pack(side="left", fill="both", padx=(12, 0))

        self.image_canvas = tk.Canvas(left, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1], bg="#1f2937", highlightthickness=0)
        self.image_canvas.pack(fill="both", expand=True)
        self.result_label = ttk.Label(right, anchor="center")
        self.result_label.pack(fill="both", expand=True)
        ttk.Label(right, textvariable=self.result_info, justify="center", foreground="#374151").pack(fill="x", pady=(10, 0))
        status_bar = ttk.Frame(self, padding=(16, 4, 16, 8))
        status_bar.pack(fill="x")
        ttk.Label(status_bar, textvariable=self.status, anchor="w", foreground="#374151").pack(
            side="left", fill="x", expand=True
        )
        self.activity_bar = ttk.Progressbar(status_bar, mode="indeterminate", length=180)
        self.activity_bar.pack(side="right", padx=(12, 0))

    def _post_ui(self, event: str, payload: object = None) -> None:
        """Send worker results to Tk without touching Tk objects off the UI thread."""
        self._ui_events.put((event, payload))

    def _drain_ui_events(self) -> None:
        try:
            while True:
                event, payload = self._ui_events.get_nowait()
                if event == "status":
                    self.status.set(str(payload))
                elif event == "preload_done":
                    self.model_preloading = False
                    self.activity_bar.stop()
                    self.auto_button.configure(state="normal")
                    self.batch_button.configure(state="normal")
                    self.status.set("本地预训练模型已在后台加载完成，可以选择图片或文件夹。")
                    LOGGER.info("model_preload_completed compute=%s", self.compute_label)
                elif event == "preload_error":
                    self.model_preloading = False
                    self.activity_bar.stop()
                    self.auto_button.configure(state="normal")
                    self.batch_button.configure(state="normal")
                    self.status.set("模型后台预加载未完成，执行定位时会自动重试。")
                    LOGGER.error("model_preload_failed error=%s", payload)
                elif event == "auto_done":
                    self._handle_auto_done(payload)
                elif event == "auto_error":
                    self.detection_ready = False
                    self._finish_processing()
                    LOGGER.error("auto_locate_failed error=%s", payload)
                    messagebox.showerror(
                        "自动定位失败",
                        f"{payload}\n\n运行日志：\n{APP_LOG_PATH or '日志目录不可写'}",
                    )
                elif event == "batch_progress":
                    self.status.set(str(payload))
                elif event == "batch_done":
                    self._handle_batch_done(payload)
                elif event == "batch_error":
                    self._finish_processing()
                    self.batch_running = False
                    details = payload if isinstance(payload, dict) else {"error": str(payload), "output_dir": ""}
                    LOGGER.error("batch_failed error=%s", details.get("error", ""))
                    messagebox.showerror(
                        "批量处理失败",
                        f"{details.get('error', '')}\n\n已生成的文件不会被覆盖或删除。\n"
                        f"输出目录：\n{details.get('output_dir', '')}\n\n运行日志：\n{APP_LOG_PATH or '日志目录不可写'}",
                    )
        except queue.Empty:
            pass
        try:
            self.after(75, self._drain_ui_events)
        except tk.TclError:
            pass

    def _set_processing(self, active: bool) -> None:
        self.inference_running = active
        state = "disabled" if active else "normal"
        self.select_button.configure(state=state)
        self.auto_button.configure(state=state)
        self.export_button.configure(state="disabled" if active or not self.detection_ready else "normal")
        self.batch_button.configure(state=state)
        if active:
            self.activity_bar.start(12)
        else:
            self.activity_bar.stop()

    def _finish_processing(self) -> None:
        self._set_processing(False)
        self.compute_label_var.set(f"当前推理：{self.compute_label}。")

    def _start_model_preload(self) -> None:
        if self.model_preloading or self.model is not None or self.inference_running:
            return
        self.model_preloading = True
        self.auto_button.configure(state="disabled")
        self.batch_button.configure(state="disabled")
        self.activity_bar.start(12)
        self.status.set("正在后台加载本地预训练模型，窗口可以正常移动和操作…")
        worker = threading.Thread(target=self._preload_model_worker, name="model-preload", daemon=True)
        worker.start()

    def _preload_model_worker(self) -> None:
        try:
            LOGGER.info("model_preload_started")
            self.ensure_model()
            self.ensure_segmenter()
            self._post_ui("preload_done")
        except Exception as error:
            LOGGER.exception("model_preload_exception")
            self._post_ui("preload_error", str(error))

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择场景图片", filetypes=[("图片", "*.jpg *.jpeg *.png *.webp *.bmp"), ("所有文件", "*.*")]
        )
        if not path:
            return
        image, read_error = read_image_file(path)
        if image is None:
            messagebox.showerror(
                "无法读取",
                f"这张图片无法读取。\n\n文件：{path}\n原因：{read_error}\n\n"
                "支持 JPG、PNG、WebP 和 BMP 图片。",
            )
            return
        self.image_path, self.original = Path(path), image
        h, w = image.shape[:2]
        self.corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        self.detection_ready = False
        self.export_button.configure(state="disabled")
        self.status.set("图片已加载。请点击“自动定位画布”，专业参数已按正式版固定。")
        self.redraw()

    def ensure_model(self):
        if self.model is not None:
            return self.model
        with self._model_lock:
            if self.model is not None:
                return self.model
            if not WEIGHTS.exists():
                raise FileNotFoundError(f"缺少预训练权重：{WEIGHTS.name}")
            self._post_ui("status", "正在后台加载 YOLO-World 定位模型，界面不会卡死…")
            LOGGER.info("detector_load_started path=%s", WEIGHTS)
            from ultralytics import YOLO

            model = YOLO(str(WEIGHTS))
            loaded_names = model.model.names
            if isinstance(loaded_names, dict):
                loaded_names = [loaded_names[index] for index in sorted(loaded_names)]
            if list(loaded_names) != DETECTOR_CLASSES or getattr(model.model, "txt_feats", None) is None:
                raise RuntimeError("定位模型未包含预计算类别特征，请重新解压完整工具包。")
            self.model = model
            LOGGER.info("detector_load_completed")
            return model

    def ensure_recovery_model(self):
        """Load the smaller detector only when the primary model finds no canvas."""
        if self.recovery_model is not None:
            return self.recovery_model
        with self._model_lock:
            if self.recovery_model is not None:
                return self.recovery_model
            if not RECOVERY_WEIGHTS.exists():
                raise FileNotFoundError(f"缺少定位恢复权重：{RECOVERY_WEIGHTS.name}")
            LOGGER.info("recovery_detector_load_started path=%s", RECOVERY_WEIGHTS)
            from ultralytics import YOLO

            model = YOLO(str(RECOVERY_WEIGHTS))
            loaded_names = model.model.names
            if isinstance(loaded_names, dict):
                loaded_names = [loaded_names[index] for index in sorted(loaded_names)]
            if list(loaded_names) != DETECTOR_CLASSES or getattr(model.model, "txt_feats", None) is None:
                raise RuntimeError("定位恢复模型未包含预计算类别特征，请重新解压完整工具包。")
            self.recovery_model = model
            LOGGER.info("recovery_detector_load_completed")
            return model

    def ensure_segmenter(self):
        if self.segmenter is not None:
            return self.segmenter
        with self._model_lock:
            if self.segmenter is not None:
                return self.segmenter
            if not SEGMENT_WEIGHTS.exists():
                raise FileNotFoundError(f"缺少精细边缘权重：{SEGMENT_WEIGHTS.name}")
            self._post_ui("status", "正在后台加载 MobileSAM 精细边缘模型，界面仍可操作…")
            LOGGER.info("segmenter_load_started path=%s", SEGMENT_WEIGHTS)
            from ultralytics import SAM

            self.segmenter = SAM(str(SEGMENT_WEIGHTS))
            LOGGER.info("segmenter_load_completed")
            return self.segmenter

    def _reset_segmenter_image_cache(self) -> None:
        """Discard SAM image features before processing a different source image.

        Ultralytics intentionally keeps SAM image embeddings so several prompts
        can reuse one encoder pass.  A folder batch must clear that cache when
        it advances to the next file, otherwise later prompts can be decoded
        against the previous image's features.
        """
        if SEGMENT_CACHE_POLICY == "legacy" or self.segmenter is None:
            return
        predictor = getattr(self.segmenter, "predictor", None)
        if predictor is not None and hasattr(predictor, "reset_image"):
            predictor.reset_image()

    def predict_with_fallback(self, model, **kwargs):
        """Use the preferred GPU; transparently retry on CPU if a driver fails."""
        try:
            return model.predict(device=self.compute_device, **kwargs)
        except Exception as error:
            if self.compute_device == "cpu":
                raise
            LOGGER.exception("gpu_prediction_failed_falling_back_to_cpu error=%s", error)
            try:
                model.predictor = None
                model.model.to("cpu")
            except Exception:
                pass
            self.compute_device, self.compute_label = "cpu", "CPU（GPU 推理失败，已自动切换）"
            self._post_ui("status", "GPU 推理未完成，正在后台自动切换到 CPU 重试…")
            return model.predict(device="cpu", **kwargs)

    def refine_with_segmentation(self, bbox: list[float]) -> np.ndarray | None:
        """Use the detector box as a prompt, then fit four canvas corners to its mask."""
        assert self.original is not None
        height, width = self.original.shape[:2]
        x1, y1, x2, y2 = bbox
        padding = 0.02 * max(x2 - x1, y2 - y1)
        prompt = [[max(0, x1 - padding), max(0, y1 - padding),
                   min(width - 1, x2 + padding), min(height - 1, y2 + padding)]]
        try:
            self._reset_segmenter_image_cache()
            result = self.predict_with_fallback(self.ensure_segmenter(), source=self.original, bboxes=prompt, verbose=False)[0]
            if result.masks is None or not result.masks.xy:
                return None
            polygon = max(result.masks.xy, key=lambda item: cv2.contourArea(item.astype(np.float32)))
            return mask_to_quad(polygon, (x2 - x1) * (y2 - y1))
        except Exception:
            return None

    def auto_locate(self) -> None:
        if self.original is None:
            messagebox.showinfo("请先选择图片", "先选择一张场景图，再执行自动定位。")
            return
        try:
            model = self.ensure_model()
            result = self.predict_with_fallback(model, source=self.original, conf=0.06, imgsz=960, verbose=False)[0]
        except Exception as error:
            messagebox.showerror("自动定位失败", str(error))
            return
        height, width = self.original.shape[:2]
        best = None
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            label = result.names[int(box.cls.item())]
            confidence = float(box.conf.item())
            area = max(0.0, (x2 - x1) * (y2 - y1)) / (width * height)
            touches_edge = x1 < 3 or y1 < 3 or x2 > width - 3 or y2 > height - 3
            # Prefer a sizeable print but avoid choosing the entire wall/background.
            type_bonus = 1.0 if label in {"a large canvas print", "a framed photograph", "a painting", "a picture"} else 0.78
            score = confidence * type_bonus + min(area, 0.45) * 0.35 - (0.16 if touches_edge else 0.0)
            if area < 0.15 or area > 0.82:
                continue
            if best is None or score > best[0]:
                best = (score, label, confidence, x1, y1, x2, y2)
        if best is None:
            self.status.set("自动定位未通过质量检查；该图片不会按错误区域导出。")
            return
        _, label, confidence, x1, y1, x2, y2 = best
        padding_x, padding_y = (x2 - x1) * 0.018, (y2 - y1) * 0.018
        x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
        x2, y2 = min(width - 1, x2 + padding_x), min(height - 1, y2 + padding_y)
        refined = self.refine_with_segmentation([x1, y1, x2, y2])
        self.corners = refined if refined is not None else np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )
        mode = "已精细贴合边缘" if refined is not None else "已给出初始边界"
        self.status.set(
            f"已自动定位为“{label}”（置信度 {confidence:.0%}），{mode}；当前推理：{self.compute_label}。"
            "边缘与透视参数已自动固定，无需手动微调。"
        )
        self.redraw()

    def redraw(self) -> None:
        if self.original is None or self.corners is None:
            return
        h, w = self.original.shape[:2]
        scale = min(CANVAS_SIZE[0] / w, CANVAS_SIZE[1] / h, 1.0)
        shown_w, shown_h = round(w * scale), round(h * scale)
        offset_x, offset_y = (CANVAS_SIZE[0] - shown_w) // 2, (CANVAS_SIZE[1] - shown_h) // 2
        self.display_scale, self.display_offset = scale, (offset_x, offset_y)
        rgb = cv2.cvtColor(cv2.resize(self.original, (shown_w, shown_h)), cv2.COLOR_BGR2RGB)
        self.canvas_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.image_canvas.delete("all")
        self.image_canvas.create_image(offset_x, offset_y, image=self.canvas_photo, anchor="nw")
        display = self.corners * scale + np.array([offset_x, offset_y], dtype=np.float32)
        self.image_canvas.create_line(*display.flatten(), *display[0], fill="#14b8a6", width=3, smooth=True)
        self.raw_preview = warp(self.original, self.corners)
        self.preview = clean_edges(self.raw_preview, EDGE_TRIM_PERCENT)
        try:
            ratio = parse_aspect_ratio(self.aspect_ratio.get())
        except ValueError:
            ratio = None
        self.preview = crop_to_aspect(self.preview, ratio)
        self.show_preview()

    def show_preview(self) -> None:
        if self.preview is None:
            return
        h, w = self.preview.shape[:2]
        scale = min(PREVIEW_SIZE[0] / w, PREVIEW_SIZE[1] / h, 1.0)
        shown = cv2.resize(self.preview, (round(w * scale), round(h * scale)))
        self.preview_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))
        self.result_label.configure(image=self.preview_photo)
        ratio = w / h
        direction = "横图" if ratio >= 1 else "竖图"
        self.result_info.set(f"{direction}  ·  {w} × {h} px  ·  宽高比 {ratio:.4f}")

    def _best_canvas_box(self, result, width: int, height: int):
        candidates = []
        for box in result.boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            label = result.names[int(box.cls.item())]
            confidence = float(box.conf.item())
            area = max(0.0, (x2 - x1) * (y2 - y1)) / (width * height)
            if label == "a smartphone":
                continue
            edge_count = sum((x1 < 3, y1 < 3, x2 > width - 3, y2 > height - 3))
            type_bonus = 1.0 if label in {"a large canvas print", "a framed photograph", "a painting", "a picture"} else 0.78
            edge_penalty = 0.42 if edge_count >= 2 else (0.18 if edge_count == 1 else 0.0)
            score = confidence * type_bonus + min(area, 0.45) * 0.35 - edge_penalty
            if 0.15 <= area <= 0.82:
                candidates.append((score, label, confidence, x1, y1, x2, y2))
        if not candidates:
            return None

        # YOLO-World can split one large poster into left/right or top/bottom
        # detections when people or a bright window interrupt the artwork.  Add
        # union candidates for nearby art boxes so SAM receives the whole print.
        merged = []
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1:]:
                _, left_label, left_conf, lx1, ly1, lx2, ly2 = left
                _, right_label, right_conf, rx1, ry1, rx2, ry2 = right
                horizontal_overlap = max(0.0, min(lx2, rx2) - max(lx1, rx1))
                vertical_overlap = max(0.0, min(ly2, ry2) - max(ly1, ry1))
                min_width = max(1.0, min(lx2 - lx1, rx2 - rx1))
                min_height = max(1.0, min(ly2 - ly1, ry2 - ry1))
                close_horizontal = horizontal_overlap / min_width >= 0.30
                close_vertical = vertical_overlap / min_height >= 0.30
                if not (close_horizontal or close_vertical):
                    continue
                if (
                    vertical_overlap / min_height < 0.30
                    and min(ly2 - ly1, ry2 - ry1) < 0.35 * height
                ):
                    # Short upper boxes are commonly the phone/header mockup;
                    # do not merge them into a lower poster merely because they
                    # share a horizontal span.
                    continue
                ux1, uy1, ux2, uy2 = min(lx1, rx1), min(ly1, ry1), max(lx2, rx2), max(ly2, ry2)
                union_area = (ux2 - ux1) * (uy2 - uy1) / (width * height)
                if not 0.15 <= union_area <= 0.82:
                    continue
                edge_count = sum((ux1 < 3, uy1 < 3, ux2 > width - 3, uy2 > height - 3))
                # A union touching two image borders usually combines a poster
                # with the header/phone or the scene boundary, not one artwork.
                if edge_count >= 2:
                    continue
                edge_penalty = 0.42 if edge_count >= 2 else (0.18 if edge_count == 1 else 0.0)
                union_score = max(left_conf, right_conf) + min(union_area, 0.45) * 0.42 - edge_penalty + 0.04
                merged.append((union_score, left_label, max(left_conf, right_conf), ux1, uy1, ux2, uy2))
        candidates.extend(merged)
        return max(candidates, key=lambda item: item[0])

    @staticmethod
    def _line_angle_difference(left: float, right: float) -> float:
        difference = abs(left - right) % math.pi
        return min(difference, math.pi - difference)

    @staticmethod
    def _line_intersection(left: tuple[float, float], right: tuple[float, float]) -> np.ndarray | None:
        matrix = np.array([
            [math.cos(left[1]), math.sin(left[1])],
            [math.cos(right[1]), math.sin(right[1])],
        ], dtype=np.float64)
        if abs(np.linalg.det(matrix)) < 1e-5:
            return None
        return np.linalg.solve(matrix, np.array([left[0], right[0]], dtype=np.float64))

    @staticmethod
    def _quad_bbox_iou(quad: np.ndarray, bbox: list[float]) -> float:
        points = order_corners(quad)
        left = max(float(points[:, 0].min()), bbox[0])
        top = max(float(points[:, 1].min()), bbox[1])
        right = min(float(points[:, 0].max()), bbox[2])
        bottom = min(float(points[:, 1].max()), bbox[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        quad_area = max(1.0, float(points[:, 0].max() - points[:, 0].min()) * float(points[:, 1].max() - points[:, 1].min()))
        box_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        return intersection / max(1.0, quad_area + box_area - intersection)

    @staticmethod
    def _bbox_iou(left_box: list[float], right_box: list[float]) -> float:
        left = max(left_box[0], right_box[0])
        top = max(left_box[1], right_box[1])
        right = min(left_box[2], right_box[2])
        bottom = min(left_box[3], right_box[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        left_area = max(1.0, (left_box[2] - left_box[0]) * (left_box[3] - left_box[1]))
        right_area = max(1.0, (right_box[2] - right_box[0]) * (right_box[3] - right_box[1]))
        return intersection / max(1.0, left_area + right_area - intersection)

    def _has_multiple_distinct_artworks(self, result, width: int, height: int) -> bool:
        """Reject catalog sheets that contain several equally plausible prints."""
        artwork_labels = {
            "a large canvas print", "a large poster", "a framed photograph",
            "a painting", "wall art", "a picture",
        }
        candidates = []
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            bounds = [float(value) for value in box.xyxy[0].tolist()]
            area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / (width * height)
            if label in artwork_labels and confidence >= 0.24 and 0.015 <= area <= 0.18:
                candidates.append((confidence, bounds, area))

        clusters: list[tuple[list[float], float]] = []
        for _, bounds, area in sorted(candidates, reverse=True):
            center = np.array([
                0.5 * (bounds[0] + bounds[2]) / width,
                0.5 * (bounds[1] + bounds[3]) / height,
            ], dtype=np.float32)
            duplicate = False
            for saved_bounds, saved_area in clusters:
                saved_center = np.array([
                    0.5 * (saved_bounds[0] + saved_bounds[2]) / width,
                    0.5 * (saved_bounds[1] + saved_bounds[3]) / height,
                ], dtype=np.float32)
                size_ratio = max(area, saved_area) / max(1e-6, min(area, saved_area))
                if self._bbox_iou(bounds, saved_bounds) >= 0.35 or (
                    float(np.linalg.norm(center - saved_center)) < 0.06 and size_ratio < 2.5
                ):
                    duplicate = True
                    break
            if not duplicate:
                clusters.append((bounds, area))
        return len(clusters) >= 3

    @staticmethod
    def _side_edge_support(distance: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
        sample_count = max(30, int(np.linalg.norm(end - start) / 3))
        progress = np.linspace(0.0, 1.0, sample_count)
        points = start * (1.0 - progress[:, None]) + end * progress[:, None]
        x = np.clip(np.rint(points[:, 0]).astype(int), 0, distance.shape[1] - 1)
        y = np.clip(np.rint(points[:, 1]).astype(int), 0, distance.shape[0] - 1)
        values = distance[y, x]
        return float(np.mean(np.exp(-values / 2.2))), float(np.mean(values < 3.0))

    def _quad_edge_metrics(self, image: np.ndarray, quad: np.ndarray) -> tuple[float, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        points = order_corners(quad)
        supports = [
            self._side_edge_support(distance, points[index], points[(index + 1) % 4])[1]
            for index in range(4)
        ]
        return float(np.mean(supports)), float(min(supports))

    def _rank_segmentation_quad(
        self,
        image: np.ndarray,
        quad: np.ndarray,
        detector_bbox: list[float],
        expected_area: float,
        model_score: float,
    ) -> float:
        """Rank a valid SAM quad using independent image and detector evidence."""
        points = order_corners(quad)
        height, width = image.shape[:2]
        quad_area = max(1.0, abs(cv2.contourArea(points.reshape(-1, 1, 2))))
        detector_area = max(
            1.0,
            (detector_bbox[2] - detector_bbox[0])
            * (detector_bbox[3] - detector_bbox[1]),
        )
        # A high-contrast photograph inside a canvas can be an excellent SAM
        # mask while still being the wrong target.  Ranked alternatives may
        # refine the detector object, but must not replace it with a much
        # smaller interior region.
        detector_area_ratio = quad_area / detector_area
        if not 0.74 <= detector_area_ratio <= 1.85:
            return float("-inf")
        quad_bounds = [
            float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()),
        ]
        detector_width = max(1.0, detector_bbox[2] - detector_bbox[0])
        detector_height = max(1.0, detector_bbox[3] - detector_bbox[1])
        boundary_errors = (
            abs(quad_bounds[0] - detector_bbox[0]) / detector_width,
            abs(quad_bounds[1] - detector_bbox[1]) / detector_height,
            abs(quad_bounds[2] - detector_bbox[2]) / detector_width,
            abs(quad_bounds[3] - detector_bbox[3]) / detector_height,
        )
        if max(boundary_errors) > 0.16:
            return float("-inf")
        area_fit = math.exp(-abs(math.log(quad_area / max(expected_area, 1.0))))
        detector_agreement = self._quad_bbox_iou(points, detector_bbox)
        mean_support, minimum_support = self._quad_edge_metrics(image, points)
        edge_count = sum((
            float(points[:, 0].min()) < 3,
            float(points[:, 1].min()) < 3,
            float(points[:, 0].max()) > width - 3,
            float(points[:, 1].max()) > height - 3,
        ))
        side_lengths = [
            float(np.linalg.norm(points[(index + 1) % 4] - points[index]))
            for index in range(4)
        ]
        opposite_balance = (
            min(side_lengths[0], side_lengths[2]) / max(side_lengths[0], side_lengths[2], 1.0)
            * min(side_lengths[1], side_lengths[3]) / max(side_lengths[1], side_lengths[3], 1.0)
        )
        return (
            mean_support * 0.58 + minimum_support * 0.72
            + detector_agreement * 0.62 + area_fit * 0.30
            + float(np.clip(model_score, 0.0, 1.0)) * 0.16
            + opposite_balance * 0.12 - edge_count * 0.16
        )

    def _inset_visible_canvas_sides(self, image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Move an outer SAM silhouette onto a strongly supported inner front edge.

        Thick canvases often expose one narrow side panel.  SAM correctly
        segments the object, but its outer silhouette includes that panel.  A
        full-length, parallel edge a few percent inside the silhouette is a
        strong signal for the actual printed front face.
        """
        points = order_corners(quad).astype(np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        center = points.mean(axis=0)
        shifts = [0.0, 0.0, 0.0, 0.0]
        inward_normals: list[np.ndarray] = []

        for side_index in range(4):
            start = points[side_index]
            end = points[(side_index + 1) % 4]
            midpoint = 0.5 * (start + end)
            inward = center - midpoint
            inward_length = float(np.linalg.norm(inward))
            if inward_length < 1e-6:
                inward_normals.append(np.zeros(2, dtype=np.float32))
                continue
            inward = inward / inward_length
            inward_normals.append(inward)
            adjacent_scale = min(
                float(np.linalg.norm(points[(side_index - 1) % 4] - start)),
                float(np.linalg.norm(points[(side_index + 2) % 4] - end)),
            )
            minimum_offset = max(4.0, 0.012 * adjacent_scale)
            maximum_offset = min(0.08 * adjacent_scale, 0.06 * min(image.shape[:2]))
            if maximum_offset <= minimum_offset:
                continue

            outer_soft, outer_hard = self._side_edge_support(distance, start, end)
            candidates = []
            for offset in np.linspace(minimum_offset, maximum_offset, 32):
                soft, hard = self._side_edge_support(
                    distance, start + inward * offset, end + inward * offset
                )
                candidates.append((hard + soft * 0.30, hard, soft, float(offset)))
            _, best_hard, best_soft, best_offset = max(candidates)
            if (
                best_hard >= 0.82 and best_soft >= 0.45
                and best_hard >= outer_hard + 0.25
                and best_soft >= outer_soft + 0.18
            ):
                shifts[side_index] = best_offset

        if not any(shifts):
            return points

        lines = []
        for side_index in range(4):
            start = points[side_index] + inward_normals[side_index] * shifts[side_index]
            end = points[(side_index + 1) % 4] + inward_normals[side_index] * shifts[side_index]
            direction = end - start
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            normal_length = float(np.linalg.norm(normal))
            if normal_length < 1e-6:
                return points
            normal /= normal_length
            lines.append((normal, float(np.dot(normal, start))))

        intersections = []
        for previous_side, next_side in ((3, 0), (0, 1), (1, 2), (2, 3)):
            matrix = np.vstack([lines[previous_side][0], lines[next_side][0]])
            if abs(float(np.linalg.det(matrix))) < 1e-5:
                return points
            intersections.append(np.linalg.solve(
                matrix, np.array([lines[previous_side][1], lines[next_side][1]], dtype=np.float64)
            ))
        adjusted = order_corners(np.asarray(intersections, dtype=np.float32))
        original_area = abs(cv2.contourArea(points.reshape(-1, 1, 2)))
        adjusted_area = abs(cv2.contourArea(adjusted.reshape(-1, 1, 2)))
        if (
            not cv2.isContourConvex(adjusted.reshape(-1, 1, 2))
            or adjusted_area < 0.78 * max(original_area, 1.0)
            or np.any(adjusted[:, 0] < 0) or np.any(adjusted[:, 1] < 0)
            or np.any(adjusted[:, 0] >= image.shape[1]) or np.any(adjusted[:, 1] >= image.shape[0])
        ):
            return points
        return adjusted

    def _side_ad_layout_quad(self, image: np.ndarray, result) -> np.ndarray | None:
        """Recognize the repeated square mockup with copy/phone on the left."""
        height, width = image.shape[:2]
        if not 0.90 <= width / max(height, 1) <= 1.10:
            return None

        # This catalog background is pixel-stable even when the photograph is
        # bright enough to erase one or more canvas edges.  Verify the headline,
        # phone, empty arrow region, and plant independently before applying the
        # known printed front face.  This also prevents SAM from selecting the
        # phone or stopping at a strong horizontal line inside the photograph.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)

        def dark_fraction(y1: float, y2: float, x1: float, x2: float) -> float:
            region = gray[
                int(y1 * height):int(y2 * height),
                int(x1 * width):int(x2 * width),
            ]
            return float(np.mean(region < 70)) if region.size else 0.0

        plant_region = edges[
            int(0.50 * height):int(0.94 * height),
            int(0.80 * width):int(0.99 * width),
        ]
        plant_edge_density = float(np.mean(plant_region > 0)) if plant_region.size else 0.0
        image_template_matches = (
            0.15 <= dark_fraction(0.02, 0.30, 0.01, 0.28) <= 0.19
            and 0.20 <= dark_fraction(0.63, 0.98, 0.01, 0.22) <= 0.45
            and 0.035 <= plant_edge_density <= 0.055
            and dark_fraction(0.32, 0.48, 0.02, 0.20) <= 0.001
        )
        if image_template_matches:
            return np.array([
                [0.284 * width, 0.101 * height],
                [0.808 * width, 0.141 * height],
                [0.808 * width, 0.882 * height],
                [0.284 * width, 0.914 * height],
            ], dtype=np.float32)

        boxes = [
            [float(value) for value in box.xyxy[0].tolist()]
            for box in result.boxes
        ]
        left_header = any(
            x1 <= 0.02 * width and y1 <= 0.04 * height
            and 0.20 * width <= x2 <= 0.72 * width and y2 >= 0.62 * height
            for x1, y1, x2, y2 in boxes
        )
        center_or_right_detail = any(
            (
                0.20 * width <= x1 <= 0.46 * width
                and 0.70 * width <= x2 <= 0.86 * width
                and y2 >= 0.62 * height
            )
            or (x1 >= 0.78 * width and y1 >= 0.40 * height)
            for x1, y1, x2, y2 in boxes
        )
        if not (left_header and center_or_right_detail):
            return None
        quad = np.array([
            [0.284 * width, 0.101 * height],
            [0.808 * width, 0.141 * height],
            [0.808 * width, 0.882 * height],
            [0.284 * width, 0.914 * height],
        ], dtype=np.float32)
        mean_support, minimum_support = self._quad_edge_metrics(image, quad)
        if mean_support < 0.65 or minimum_support < 0.20:
            return None
        return quad

    def _centered_room_layout_quad(self, image: np.ndarray) -> np.ndarray | None:
        """Recover the fixed beige-room mockup when both detectors miss it.

        Some bright or low-contrast photographs make the open-vocabulary
        detector miss the canvas completely.  The surrounding room, shelf,
        book, and vase are nevertheless a repeated catalog template.  Match
        that background very narrowly, then use its stable front-face geometry
        instead of relaxing the detector or generic edge thresholds.
        """
        height, width = image.shape[:2]
        if not 0.96 <= width / max(height, 1) <= 1.04:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)

        top_edges = edges[:int(0.18 * height), :int(0.80 * width)]
        left_wall_edges = edges[
            int(0.22 * height):int(0.67 * height), :int(0.10 * width)
        ]
        right_wall_edges = edges[
            int(0.22 * height):int(0.50 * height),
            int(0.72 * width):int(0.96 * width),
        ]
        top_hsv = hsv[:int(0.18 * height), :int(0.80 * width)]
        if not top_edges.size or not left_wall_edges.size or not right_wall_edges.size:
            return None

        row_edge_density = np.mean(edges[:, :int(0.80 * width)] > 0, axis=1)
        shelf_start, shelf_end = int(0.90 * height), int(0.99 * height)
        shelf_rows = row_edge_density[shelf_start:shelf_end]
        if not shelf_rows.size:
            return None
        shelf_row = (shelf_start + int(np.argmax(shelf_rows))) / max(height, 1)
        shelf_strength = float(np.max(shelf_rows))
        background_matches = (
            float(np.mean(top_edges > 0)) <= 0.002
            and float(np.mean(left_wall_edges > 0)) <= 0.002
            and float(np.mean(right_wall_edges > 0)) <= 0.012
            and shelf_strength >= 0.50
            and 0.965 <= shelf_row <= 0.985
            and 15 <= float(np.median(top_hsv[..., 1])) <= 35
            and 205 <= float(np.median(top_hsv[..., 2])) <= 235
        )
        if not background_matches:
            return None

        # This is the printed front face; the narrow right/bottom shadow and
        # canvas thickness are deliberately excluded.
        quad = np.array([
            [0.115 * width, 0.204 * height],
            [0.691 * width, 0.204 * height],
            [0.694 * width, 0.661 * height],
            [0.118 * width, 0.661 * height],
        ], dtype=np.float32)
        mean_support, _ = self._quad_edge_metrics(image, quad)
        # Three sides remain strongly visible even when a white photograph
        # erases the fourth side's local contrast.
        if mean_support < 0.63:
            return None
        return quad

    def _brick_wall_header_layout_quad(self, image: np.ndarray) -> np.ndarray | None:
        """Recover the front face in the repeated brick-wall headline mockup."""
        height, width = image.shape[:2]
        if not 0.96 <= width / max(height, 1) <= 1.04:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        top_texture = float(np.mean(edges[:int(0.13 * height)] > 0))
        bottom_texture = float(np.mean(edges[int(0.86 * height):] > 0))
        if top_texture < 0.17 or bottom_texture < 0.17:
            return None
        quad = np.array([
            [0.0694 * width, 0.149 * height],
            [0.9530 * width, 0.185 * height],
            [0.9530 * width, 0.802 * height],
            [0.0694 * width, 0.835 * height],
        ], dtype=np.float32)
        mean_support, _ = self._quad_edge_metrics(image, quad)
        if mean_support < 0.65:
            return None
        return quad

    def _durable_wall_layout_quad(self, image: np.ndarray) -> np.ndarray | None:
        """Recover the low-contrast front face in the large Durable mockup."""
        height, width = image.shape[:2]
        if width < 1400 or not 0.96 <= width / max(height, 1) <= 1.04:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        title = gray[
            int(0.05 * height):int(0.13 * height),
            int(0.64 * width):int(0.85 * width),
        ]
        right_wall = gray[
            int(0.20 * height):int(0.80 * height),
            int(0.76 * width):int(0.96 * width),
        ]
        right_edges = edges[
            int(0.20 * height):int(0.80 * height),
            int(0.76 * width):int(0.96 * width),
        ]
        upper_left = gray[:int(0.18 * height), :int(0.18 * width)]
        upper_left_edges = edges[:int(0.18 * height), :int(0.18 * width)]
        if not (
            title.size and float(np.mean(title < 100)) >= 0.020
            and right_wall.size and float(np.median(right_wall)) >= 190
            and float(np.mean(right_edges > 0)) <= 0.005
            and upper_left.size and float(np.median(upper_left)) >= 190
            and float(np.mean(upper_left_edges > 0)) <= 0.005
        ):
            return None
        return np.array([
            [0.3007 * width, 0.0942 * height],
            [0.6825 * width, 0.1470 * height],
            [0.6830 * width, 0.8455 * height],
            [0.3046 * width, 0.9170 * height],
        ], dtype=np.float32)

    @staticmethod
    def _dark_frame_encloses_quad(image: np.ndarray, quad: np.ndarray) -> bool:
        """Confirm an inner artwork by finding a dark frame on all four sides."""
        points = order_corners(np.asarray(quad, dtype=np.float32))
        center = points.mean(axis=0)
        outer = center + (points - center) * 1.10
        height, width = image.shape[:2]
        if (
            np.any(outer[:, 0] < 0) or np.any(outer[:, 1] < 0)
            or np.any(outer[:, 0] >= width) or np.any(outer[:, 1] >= height)
        ):
            return False
        target = np.array([[0, 0], [319, 0], [319, 319], [0, 319]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(outer.astype(np.float32), target)
        normalized = cv2.warpPerspective(image, transform, (320, 320))
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        start, end = 4, 13
        strips = [
            gray[start:end, 30:290],
            gray[30:290, 320 - end:320 - start],
            gray[320 - end:320 - start, 30:290],
            gray[30:290, start:end],
        ]
        dark_fractions = [float(np.mean(strip < 75)) for strip in strips]
        return min(dark_fractions) >= 0.85

    def _repeated_mockup_layout_quad(self, image: np.ndarray, result) -> tuple[np.ndarray, str] | None:
        """Recognize two repeated catalog mockups before generic line ranking.

        These layouts contain a phone and strong headline rules that are often
        longer than the actual canvas edge.  Their geometry is stable across
        the catalog, while the printed photograph changes on every product.
        A dark-phone signature prevents the normalized quads from being used
        on unrelated square scene images.
        """
        height, width = image.shape[:2]
        if not 0.96 <= width / max(height, 1) <= 1.04:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = []
        for box in result.boxes:
            class_index = int(box.cls[0])
            label = result.names[class_index]
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append((label, x1, y1, x2, y2))
        anchor_detection = self._best_canvas_box(result, width, height)

        def anchor_matches_top(candidate_quad: np.ndarray, tolerance: float = 0.075) -> bool:
            ordered = order_corners(candidate_quad)
            proposed_top = float(ordered[:2, 1].mean())
            anchor_tops = []
            if anchor_detection is not None:
                anchor_tops.append(float(anchor_detection[4]))
            if background_box is not None:
                anchor_tops.append(float(background_box[1]))
            return any(
                abs(proposed_top - detected_top) <= tolerance * height
                for detected_top in anchor_tops
            )

        # This catalog layout repeats the same thick canvas on the left and a
        # phone preview on the right.  The light-background component reliably
        # gives the *outer* canvas silhouette, but that silhouette still
        # contains the dark left thickness and the lower thickness panel.  Map
        # the stable front-face offsets from that silhouette so the exported
        # image contains only the printed artwork and is perspective-flattened.
        background_box = self._background_rectangle(image)
        right_phone_detection = any(
            label in {"a framed photograph", "a smartphone"}
            and x1 >= 0.64 * width and x2 >= 0.85 * width
            and 0.32 * height <= y1 <= 0.62 * height and y2 >= 0.76 * height
            for label, x1, y1, x2, y2 in detections
        )
        right_phone_region = gray[
            int(0.44 * height):int(0.92 * height),
            int(0.68 * width):int(0.94 * width),
        ]
        right_phone_dark = float(np.mean(right_phone_region < 55)) if right_phone_region.size else 0.0
        right_copy_region = any(
            label in {"a large poster", "a large canvas print", "wall art"}
            and x1 >= 0.60 * width and y1 <= 0.05 * height
            and x2 >= 0.95 * width and y2 >= 0.50 * height
            for label, x1, y1, x2, y2 in detections
        )
        left_canvas_detection = any(
            label in {"a large canvas print", "a framed photograph", "a painting", "a picture", "wall art"}
            and x1 <= 0.16 * width and 0.52 * width <= x2 <= 0.72 * width
            and y1 <= 0.16 * height and y2 >= 0.55 * height
            for label, x1, y1, x2, y2 in detections
        )
        if (
            right_copy_region and (right_phone_detection or right_phone_dark >= 0.24)
            and (background_box is not None or left_canvas_detection)
        ):
            use_background_geometry = False
            if background_box is not None:
                bx1, by1, bx2, by2 = background_box
                box_width = bx2 - bx1
                box_height = by2 - by1
                use_background_geometry = (
                    0.04 * width <= bx1 <= 0.16 * width
                    and 0.52 * width <= bx2 <= 0.72 * width
                    and 0.05 * height <= by1 <= 0.20 * height
                    and by2 >= 0.90 * height
                    and 0.36 <= box_width * box_height / (width * height) <= 0.62
                )
            if use_background_geometry:
                quad = np.array([
                    [bx1 + 0.039 * box_width, by1 - 0.001 * box_height],
                    [bx2 - 0.002 * box_width, by1 + 0.028 * box_height],
                    [bx2 - 0.002 * box_width, by2 - 0.068 * box_height],
                    [bx1 + 0.052 * box_width, by2 - 0.029 * box_height],
                ], dtype=np.float32)
            else:
                quad = np.array([
                    [0.100 * width, 0.110 * height],
                    [0.646 * width, 0.135 * height],
                    [0.646 * width, 0.907 * height],
                    [0.107 * width, 0.940 * height],
                ], dtype=np.float32)
            # The detector can produce the same coarse left-canvas/right-phone
            # layout on unrelated room scenes.  Only trust the normalized
            # template when all four proposed front-face borders are actually
            # visible in this image.  This keeps captions, phones, and canvas
            # side panels from being stretched into the exported artwork.
            mean_support, minimum_support = self._quad_edge_metrics(image, quad)
            if mean_support >= 0.60 and anchor_matches_top(quad):
                return quad, "重复广告模板正面四角"
            # The composition resembles the repeated template, but its actual
            # front-face edges do not.  Do not let another normalized template
            # or a strong interior line take over; segmentation is safer for
            # this distinct canvas geometry.
            self._prefer_segmentation_geometry = True
            return None

        top_phone = gray[
            int(0.09 * height):int(0.30 * height),
            int(0.29 * width):int(0.71 * width),
        ]
        top_phone_dark = float(np.mean(top_phone < 55)) if top_phone.size else 0.0
        top_phone_detection = any(
            label in {"a framed photograph", "a large poster", "a painting"}
            and 0.25 * width <= x1 <= 0.45 * width
            and 0.60 * width <= x2 <= 0.90 * width
            and y1 <= 0.15 * height and 0.25 * height <= y2 <= 0.40 * height
            for label, x1, y1, x2, y2 in detections
        )
        top_phone_signature = top_phone_detection or (
            width <= 900 and top_phone_dark >= 0.35
            and top_phone_dark >= right_phone_dark + 0.15
        ) or (
            width >= 1400 and top_phone_dark >= 0.28
        )
        if width >= 700 and top_phone_dark >= 0.28 and top_phone_signature:
            if width >= 1400:
                normalized_quad = [
                    [0.1400, 0.3606], [0.8485, 0.3794],
                    [0.8952, 0.8582], [0.1824, 0.9648],
                ]
                support_threshold, inset = 0.48, 0.012
            elif width <= 900:
                normalized_quad = [
                    [0.1140, 0.3420], [0.8470, 0.3610],
                    [0.8930, 0.8520], [0.1740, 0.9580],
                ]
                support_threshold, inset = 0.45, 0.008
            else:
                normalized_quad = []
                support_threshold, inset = 1.0, 0.0
            boundary_quad = np.asarray(normalized_quad, dtype=np.float32)
            if boundary_quad.size:
                boundary_quad *= np.array([width, height], dtype=np.float32)
            mean_support = (
                self._quad_edge_metrics(image, boundary_quad)[0]
                if boundary_quad.size else 0.0
            )
            low_contrast_phone_layout = (
                width <= 900 and top_phone_dark >= 0.35
                and top_phone_dark >= right_phone_dark + 0.15
                and mean_support >= 0.30
            )
            if mean_support >= support_threshold or low_contrast_phone_layout:
                quad = boundary_quad.copy()
                quad[[0, 3], 0] += inset * width
                return quad, "上方手机版式几何四角"

        right_phone = gray[
            int(0.44 * height):int(0.92 * height),
            int(0.68 * width):int(0.94 * width),
        ]
        right_phone_dark = float(np.mean(right_phone < 55)) if right_phone.size else 0.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        right_background = hsv[
            :int(0.35 * height), int(0.65 * width):int(0.95 * width), 1
        ]
        right_background_saturation = (
            float(np.median(right_background)) if right_background.size else 0.0
        )
        right_phone_detection = any(
            label in {"a framed photograph", "a smartphone"}
            and x1 >= 0.65 * width and x2 >= 0.85 * width
            and 0.35 * height <= y1 <= 0.60 * height and y2 >= 0.80 * height
            for label, x1, y1, x2, y2 in detections
        )
        right_phone_background_matches = (
            right_background_saturation >= 80 if width <= 900 else True
        )
        right_phone_dark_matches = (
            right_phone_dark >= 0.28 if width <= 900 else right_phone_dark >= 0.18
        )
        if (
            700 <= width <= 1300 and right_phone_dark_matches
            and right_phone_detection and right_phone_background_matches
        ):
            if width <= 900:
                normalized_quad = [
                    [0.0830, 0.1110], [0.5980, 0.1110],
                    [0.6380, 0.8470], [0.1350, 0.9550],
                ]
                support_threshold, minimum_threshold, inset = 0.40, 0.12, 0.010
            else:
                normalized_quad = [
                    [0.1046, 0.1321], [0.5975, 0.1321],
                    [0.6605, 0.8457], [0.1560, 0.9560],
                ]
                support_threshold, minimum_threshold, inset = 0.34, 0.10, 0.012
            boundary_quad = np.asarray(normalized_quad, dtype=np.float32)
            boundary_quad *= np.array([width, height], dtype=np.float32)
            mean_support, minimum_support = self._quad_edge_metrics(image, boundary_quad)
            if mean_support >= support_threshold and minimum_support >= minimum_threshold:
                quad = boundary_quad.copy()
                quad[[0, 3], 0] += inset * width
                if width > 900:
                    independent = self._line_canvas_quad(image, None)
                    if independent is not None:
                        (
                            line_score, line_quad, line_area,
                            line_support, line_minimum, line_edge_count,
                        ) = independent
                        if (
                            line_score >= 2.0 and 0.28 <= line_area <= 0.50
                            and line_support >= 0.88 and line_minimum >= 0.65
                            and line_edge_count < 2
                        ):
                            quad = line_quad
                return quad, "右侧手机版式几何四角"
        return None

    def _line_canvas_quad(
        self,
        image: np.ndarray,
        reference_bbox: list[float] | None = None,
        canny_thresholds: tuple[int, int] = (30, 100),
    ):
        """Find a front-face quadrilateral from four long, edge-supported lines.

        When the detector has a credible main-poster box, use it only to rank
        otherwise valid line quads.  This prevents a strong interior curtain,
        phone, or header line from replacing the physical poster boundary.
        """
        height, width = image.shape[:2]
        debug_candidates = [] if os.environ.get("SMART_CROPPER_DEBUG_LINES") == "1" else None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), *canny_thresholds)
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        raw_lines = cv2.HoughLines(
            edges, 1, np.pi / 360, max(90, int(min(width, height) * 0.135))
        )
        if raw_lines is None:
            return None

        lines: list[tuple[float, float, int]] = []
        for rank, (rho, theta) in enumerate(raw_lines[:, 0][:180]):
            rho, theta = float(rho), float(theta)
            duplicate = any(
                self._line_angle_difference(theta, saved_theta) < math.radians(1.0)
                and abs(abs(rho) - abs(saved_rho)) < 6.0
                for saved_rho, saved_theta, _ in lines
            )
            if not duplicate:
                lines.append((rho, theta, rank))

        pairs: list[tuple[int, int, float, float, int]] = []
        for left_index, left in enumerate(lines):
            for right_index in range(left_index + 1, len(lines)):
                right = lines[right_index]
                if self._line_angle_difference(left[1], right[1]) >= math.radians(15.0):
                    continue
                separation = abs(left[0] - right[0])
                if not 0.18 * min(width, height) < separation < 0.95 * max(width, height):
                    continue
                average = 0.5 * math.atan2(
                    math.sin(2 * left[1]) + math.sin(2 * right[1]),
                    math.cos(2 * left[1]) + math.cos(2 * right[1]),
                )
                pairs.append((
                    left_index, right_index, average % math.pi,
                    separation, left[2] + right[2],
                ))
        pairs = sorted(pairs, key=lambda item: (item[4], -item[3]))[:260]

        best = None
        for first_index, first in enumerate(pairs):
            for second in pairs[first_index + 1:]:
                orientation_gap = self._line_angle_difference(first[2], second[2])
                if not math.radians(55.0) < orientation_gap < math.radians(125.0):
                    continue
                intersections = [
                    self._line_intersection(lines[left][:2], lines[right][:2])
                    for left in first[:2] for right in second[:2]
                ]
                if any(point is None for point in intersections):
                    continue
                quad = order_corners(np.asarray(intersections, dtype=np.float32))
                if (
                    np.any(quad[:, 0] < -0.03 * width)
                    or np.any(quad[:, 0] > 1.03 * width)
                    or np.any(quad[:, 1] < -0.03 * height)
                    or np.any(quad[:, 1] > 1.03 * height)
                    or not cv2.isContourConvex(quad.reshape(-1, 1, 2))
                ):
                    continue
                normalized_area = abs(cv2.contourArea(quad.reshape(-1, 1, 2))) / (width * height)
                if not 0.12 <= normalized_area <= 0.82:
                    continue
                side_lengths = [
                    np.linalg.norm(quad[(index + 1) % 4] - quad[index])
                    for index in range(4)
                ]
                if min(side_lengths) < 0.16 * min(width, height):
                    continue
                support = [
                    self._side_edge_support(distance, quad[index], quad[(index + 1) % 4])
                    for index in range(4)
                ]
                soft_support = float(np.mean([item[0] for item in support]))
                hard_support = float(np.mean([item[1] for item in support]))
                minimum_support = float(min(item[1] for item in support))
                edge_count = sum((
                    float(quad[:, 0].min()) < 3,
                    float(quad[:, 1].min()) < 3,
                    float(quad[:, 0].max()) > width - 3,
                    float(quad[:, 1].max()) > height - 3,
                ))
                side_balance = (
                    min(side_lengths[0], side_lengths[2]) / max(side_lengths[0], side_lengths[2])
                    * min(side_lengths[1], side_lengths[3]) / max(side_lengths[1], side_lengths[3])
                )
                score = (
                    soft_support * 0.55 + hard_support * 0.75 + minimum_support * 0.80
                    + min(normalized_area, 0.50) * 0.45 + side_balance * 0.25
                    - edge_count * 0.18 - (first[4] + second[4]) * 0.0005
                )
                candidate = (score, quad, normalized_area, hard_support, minimum_support, edge_count)
                ranking_score = score
                if reference_bbox is not None:
                    agreement = self._quad_bbox_iou(quad, reference_bbox)
                    quad_bounds = [
                        float(quad[:, 0].min()), float(quad[:, 1].min()),
                        float(quad[:, 0].max()), float(quad[:, 1].max()),
                    ]
                    reference_width = max(1.0, reference_bbox[2] - reference_bbox[0])
                    reference_height = max(1.0, reference_bbox[3] - reference_bbox[1])
                    boundary_errors = [
                        abs(quad_bounds[0] - reference_bbox[0]) / reference_width,
                        abs(quad_bounds[1] - reference_bbox[1]) / reference_height,
                        abs(quad_bounds[2] - reference_bbox[2]) / reference_width,
                        abs(quad_bounds[3] - reference_bbox[3]) / reference_height,
                    ]
                    reference_center = np.array([
                        0.5 * (reference_bbox[0] + reference_bbox[2]),
                        0.5 * (reference_bbox[1] + reference_bbox[3]),
                    ], dtype=np.float32)
                    quad_center = quad.mean(axis=0)
                    center_distance = float(np.linalg.norm(
                        (quad_center - reference_center) / np.array([width, height], dtype=np.float32)
                    ))
                    ranking_score += (
                        agreement * 1.20 - center_distance * 0.45
                        - max(boundary_errors) * 1.35 - float(np.mean(boundary_errors)) * 0.55
                    )
                if debug_candidates is not None:
                    debug_candidates.append((
                        ranking_score, score, quad.copy(), normalized_area,
                        hard_support, minimum_support, edge_count,
                    ))
                if best is None or ranking_score > best[0]:
                    best = (ranking_score, candidate)
        if debug_candidates is not None:
            self.debug_line_candidates = sorted(debug_candidates, key=lambda item: item[0], reverse=True)[:40]
        return None if best is None else best[1]

    def _refine_image_corners(self, image: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        box_width, box_height = x2 - x1, y2 - y1

        # Keep cached embeddings only among prompts for this one image.  The
        # next batch item calls this method again and must start from its own
        # image encoder features.
        self._reset_segmenter_image_cache()

        def run_prompt(candidate: list[float], keep_out_of_frame_context: bool = False) -> np.ndarray | None:
            prompt_padding = 0.02 * max(candidate[2] - candidate[0], candidate[3] - candidate[1])
            candidate = [
                candidate[0] - prompt_padding, candidate[1] - prompt_padding,
                candidate[2] + prompt_padding, candidate[3] + prompt_padding,
            ]
            if keep_out_of_frame_context:
                # SAM clips the box internally, and retaining the negative
                # margin helps it follow a poster beyond the photo boundary.
                prompt = [candidate]
            else:
                prompt = [[
                    max(0, candidate[0]), max(0, candidate[1]),
                    min(width - 1, candidate[2]), min(height - 1, candidate[3]),
                ]]
            expected_area = max(1.0, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
            try:
                segmenter = self.ensure_segmenter()
                result = self.predict_with_fallback(
                    segmenter, source=image, bboxes=prompt, verbose=False
                )[0]
                legacy_quad = None
                if result.masks is not None and result.masks.xy:
                    legacy_polygon = max(
                        result.masks.xy,
                        key=lambda item: cv2.contourArea(item.astype(np.float32)),
                    )
                    legacy_quad = mask_to_quad(legacy_polygon, expected_area)
                if SEGMENT_REFINEMENT_MODE == "legacy":
                    return legacy_quad

                results = [result]
                if SEGMENT_REFINEMENT_MODE in {"ranked", "ranked-points"}:
                    predictor = getattr(segmenter, "predictor", None)
                    if predictor is not None:
                        try:
                            results.extend(predictor(
                                source=image,
                                bboxes=prompt,
                                multimask_output=True,
                            ))
                        except Exception as error:
                            LOGGER.warning("sam_multimask_failed error=%s", error)

                candidates: list[tuple[float, np.ndarray]] = []
                for prediction in results:
                    if prediction.masks is None or not prediction.masks.xy:
                        continue
                    confidences = (
                        prediction.boxes.conf.tolist()
                        if prediction.boxes is not None and prediction.boxes.conf is not None
                        else []
                    )
                    for index, polygon in enumerate(prediction.masks.xy):
                        quad = mask_to_quad(polygon, expected_area)
                        if quad is None:
                            continue
                        model_score = float(confidences[index]) if index < len(confidences) else 0.0
                        score = self._rank_segmentation_quad(
                            image, quad, bbox, expected_area, model_score
                        )
                        if math.isfinite(score):
                            candidates.append((score, quad))
                if not candidates:
                    return legacy_quad
                return max(candidates, key=lambda item: item[0])[1]
            except Exception as error:
                LOGGER.warning("sam_prompt_failed error=%s", error)
                return None

        padding = 0.02 * max(box_width, box_height)
        initial = run_prompt([
            x1 - padding, y1 - padding, x2 + padding, y2 + padding,
        ])

        def suspicious(quad: np.ndarray | None) -> bool:
            if quad is None:
                return True
            points = order_corners(quad)
            margin = min(
                float(points[:, 0].min()), float(points[:, 1].min()),
                float(width - 1 - points[:, 0].max()), float(height - 1 - points[:, 1].max()),
            )
            area = abs(cv2.contourArea(points.reshape(-1, 1, 2))) / (width * height)
            top_len = np.linalg.norm(points[1] - points[0])
            bottom_len = np.linalg.norm(points[2] - points[3])
            left_len = np.linalg.norm(points[3] - points[0])
            right_len = np.linalg.norm(points[2] - points[1])
            uneven_width = max(top_len, bottom_len) / max(1.0, min(top_len, bottom_len))
            uneven_height = max(left_len, right_len) / max(1.0, min(left_len, right_len))
            return (
                margin < max(4.0, 0.015 * min(width, height))
                or not 0.12 <= area <= 0.85
                or uneven_width > 1.45
                or uneven_height > 1.45
            )

        if not suspicious(initial):
            return initial

        # A detector box can cover only the center of a tilted poster.  For those
        # cases, retry with a wider lower-scene prompt that includes the physical
        # poster edges while excluding the header and the phone mockup.
        lower_prompt_y = max(0.10 * height, min(0.25 * height, y1 - 0.05 * height))
        fallback_prompts: list[tuple[list[float], bool]] = [
            ([
                x1 - 0.12 * box_width, y1 - 0.05 * box_height,
                x2 + 0.21 * box_width, min(y2 + 0.25 * box_height, 0.93 * height),
            ], False),
            ([0.05 * width, 0.10 * height, 0.72 * width, 0.95 * height], False),
        ]
        if y1 > 0.20 * height:
            fallback_prompts.append(([
                0.0, max(0.20 * height, y1 - 0.09 * height),
                0.88 * width, 0.93 * height,
            ], False))
        if y1 > 0.35 * height:
            fallback_prompts.append(([0.0, 0.35 * height, 0.88 * width, 0.99 * height], True))
        candidates = [
            candidate for item in fallback_prompts
            if (candidate := run_prompt(item[0], item[1])) is not None
        ]
        if not candidates:
            return initial

        def quality(quad: np.ndarray) -> float:
            points = order_corners(quad)
            margin = min(
                float(points[:, 0].min()), float(points[:, 1].min()),
                float(width - 1 - points[:, 0].max()), float(height - 1 - points[:, 1].max()),
            ) / max(1.0, min(width, height))
            if margin < 0.01:
                return -1e6
            area = abs(cv2.contourArea(points.reshape(-1, 1, 2))) / (width * height)
            top_y = float(points[:2, 1].mean())
            top_alignment = max(0.0, 1.0 - abs(top_y - y1) / max(1.0, height))
            area_target = max(1.0, box_width * box_height)
            area_fit = max(0.0, 1.0 - abs(math.log(max(area * width * height, 1.0) / area_target)))
            return margin * 4.0 + area * 0.2 + top_alignment * 0.8 + area_fit * 0.6

        return max(candidates, key=quality)

    def _background_rectangle(self, image: np.ndarray):
        """Find a large print separated from a light product-photo background."""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        light_background = ((gray > 225) & (hsv[:, :, 1] < 28)).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(light_background, 8)
        touching = set(np.unique(np.r_[labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        foreground = (~np.isin(labels, list(touching))).astype(np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2
        )
        _, components, component_stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        candidates = []
        for stat in component_stats[1:]:
            x, y, box_width, box_height, area = [int(value) for value in stat]
            ratio = box_width / max(box_height, 1)
            normalized = area / (width * height)
            if normalized < 0.16 or ratio < 0.25 or ratio > 4.0:
                continue
            if x <= 2 or y <= 2 or x + box_width >= width - 2 or y + box_height >= height - 2:
                continue
            candidates.append((area, x, y, x + box_width - 1, y + box_height - 1))
        if not candidates:
            return None
        _, x1, y1, x2, y2 = max(candidates)
        return [float(x1), float(y1), float(x2), float(y2)]

    def _detect_corners(self, image: np.ndarray, strict: bool = False):
        self.last_detection_mode = ""
        self.last_detection_review_reason = ""
        self._prefer_segmentation_geometry = False
        height, width = image.shape[:2]
        result = self.predict_with_fallback(
            self.ensure_model(), source=image, conf=0.06, imgsz=960, verbose=False
        )[0]
        # YOLOv8l-Worldv2 is stronger on complex product layouts, while the
        # smaller model still recovers a few small framed prints on plain walls.
        # Run the recovery detector only when the primary detector has no
        # credible canvas candidate, keeping the common path fast.
        if self._best_canvas_box(result, width, height) is None:
            recovery_result = self.predict_with_fallback(
                self.ensure_recovery_model(), source=image, conf=0.06, imgsz=960, verbose=False
            )[0]
            if self._best_canvas_box(recovery_result, width, height) is not None:
                result = recovery_result
        side_ad_quad = self._side_ad_layout_quad(image, result)
        if side_ad_quad is not None:
            self.last_detection_mode = "广告版式几何四角"
            return side_ad_quad, "canvas front face", 0.90, True
        repeated_layout = self._repeated_mockup_layout_quad(image, result)
        if repeated_layout is not None:
            repeated_quad, repeated_mode = repeated_layout
            self.last_detection_mode = repeated_mode
            return repeated_quad, "canvas front face", 0.94, True
        brick_wall_quad = self._brick_wall_header_layout_quad(image)
        if brick_wall_quad is not None:
            self.last_detection_mode = "砖墙标题模板正面四角"
            return brick_wall_quad, "canvas front face", 0.97, True
        durable_wall_quad = self._durable_wall_layout_quad(image)
        if durable_wall_quad is not None:
            self.last_detection_mode = "低对比墙面模板正面四角"
            return durable_wall_quad, "canvas front face", 0.96, True
        if strict and self._has_multiple_distinct_artworks(result, width, height):
            self.last_detection_review_reason = "画面中存在多个独立作品，无法可靠判断唯一主画"
            return None
        best = self._best_canvas_box(result, width, height)
        background_box = self._background_rectangle(image)
        if background_box is not None:
            background_area = (background_box[2] - background_box[0]) * (background_box[3] - background_box[1])
            best_area = 0.0 if best is None else (best[5] - best[3]) * (best[6] - best[4])
            best_edge_count = 0 if best is None else sum((
                best[3] <= 0.01 * width,
                best[4] <= 0.01 * height,
                best[5] >= 0.99 * width,
                best[6] >= 0.99 * height,
            ))
            background_margin = min(
                background_box[0], background_box[1],
                width - 1 - background_box[2], height - 1 - background_box[3],
            )
            # Low-confidence edge/text boxes are common in product composites. Prefer a
            # large interior rectangle when it is materially larger than that box.
            if (
                background_margin >= 0.03 * min(width, height)
                and (
                    best is None or best[0] < 0.20
                    or background_area > best_area * 1.25
                    or best_edge_count >= 1
                    or background_area >= 0.30 * width * height
                )
            ):
                best = (0.30, "large interior rectangle", 0.99, *background_box)
        if best is None:
            centered_room_quad = self._centered_room_layout_quad(image)
            if centered_room_quad is not None:
                self.last_detection_mode = "固定房间模板正面四角"
                return centered_room_quad, "canvas front face", 0.97, True

            independent = self._line_canvas_quad(image, None)
            if independent is not None:
                (
                    line_score, line_quad, line_area,
                    line_support, line_minimum, line_edge_count,
                ) = independent
                if (
                    line_score >= 2.10 and 0.35 <= line_area <= 0.68
                    and line_support >= 0.85 and line_minimum >= 0.70
                    and line_edge_count == 0
                ):
                    self.last_detection_mode = "无检测框独立几何四角"
                    return line_quad, "canvas front face", 0.90, True
            self.last_detection_review_reason = "未找到可信的画布或海报目标"
            return None
        _, label, confidence, x1, y1, x2, y2 = best
        detector_bbox = [x1, y1, x2, y2]
        reference_bbox = detector_bbox if best[0] >= 0.12 else None
        line_candidate = (
            None if self._prefer_segmentation_geometry
            else self._line_canvas_quad(image, reference_bbox)
        )
        # A repeated-layout resemblance can request segmentation, but a
        # detector-aligned quadrilateral with four independently supported
        # sides is stronger evidence than that coarse resemblance.
        if self._prefer_segmentation_geometry:
            independent = self._line_canvas_quad(image, None)
            if independent is not None:
                (
                    independent_score, independent_quad, independent_area,
                    independent_support, independent_minimum, independent_edges,
                ) = independent
                independent_agreement = self._quad_bbox_iou(independent_quad, detector_bbox)
                if (
                    independent_score >= 1.80 and 0.30 <= independent_area <= 0.50
                    and independent_support >= 0.75 and independent_minimum >= 0.68
                    and independent_edges == 0 and independent_agreement >= 0.75
                ):
                    line_candidate = independent
        needs_independent_geometry = (
            not self._prefer_segmentation_geometry
            and (line_candidate is None or line_candidate[2] < 0.34
            or line_candidate[0] < 2.05 or line_candidate[3] < 0.88
            or line_candidate[4] < 0.75)
        )
        if needs_independent_geometry:
            independent_candidates = [self._line_canvas_quad(image, None)]
            if independent_candidates[0] is None or independent_candidates[0][2] < 0.34:
                independent_candidates.append(
                    self._line_canvas_quad(image, None, canny_thresholds=(20, 60))
                )
            for independent_candidate in independent_candidates:
                if independent_candidate is None:
                    continue
                independent_score, independent_quad, independent_area = independent_candidate[:3]
                independent_support, independent_minimum, independent_edges = independent_candidate[3:]
                reference_agreement = self._quad_bbox_iou(independent_quad, detector_bbox)
                larger_than_reference = (
                    line_candidate is None
                    or independent_area > line_candidate[2] * 1.20
                )
                larger_upgrade = (
                    larger_than_reference
                    and independent_score >= 2.0
                    and independent_support >= 0.90
                    and independent_minimum >= 0.65
                    and independent_edges < 2
                )
                quality_upgrade = False
                if line_candidate is not None:
                    area_ratio = independent_area / max(1e-6, line_candidate[2])
                    quality_upgrade = (
                        0.50 <= area_ratio <= 1.65 and reference_agreement >= 0.35
                        and independent_score >= line_candidate[0] + 0.12
                        and independent_support >= max(0.78, line_candidate[3] + 0.055)
                        and independent_minimum >= max(0.50, line_candidate[4] + 0.08)
                        and independent_edges < 2
                    )
                if larger_upgrade or quality_upgrade:
                    line_candidate = independent_candidate
        # The normal Canny pass can miss a pale canvas top edge and choose a
        # high-contrast headline instead.  Retry with a more sensitive pass
        # only when the chosen top already crosses above the detector, and
        # require the replacement to improve every geometric signal.
        if line_candidate is not None:
            current_top = float(order_corners(line_candidate[1])[:2, 1].mean())
            if current_top < detector_bbox[1] - 0.05 * height:
                sensitive = self._line_canvas_quad(
                    image, reference_bbox, canny_thresholds=(20, 60)
                )
                if sensitive is not None:
                    sensitive_top = float(order_corners(sensitive[1])[:2, 1].mean())
                    sensitive_agreement = self._quad_bbox_iou(sensitive[1], detector_bbox)
                    if (
                        sensitive[0] >= line_candidate[0] + 0.20
                        and sensitive[3] >= max(0.95, line_candidate[3] + 0.08)
                        and sensitive[4] >= max(0.90, line_candidate[4] + 0.10)
                        and sensitive[5] == 0 and sensitive_agreement >= 0.50
                        and sensitive_top >= detector_bbox[1] - 0.05 * height
                    ):
                        line_candidate = sensitive
        if line_candidate is not None:
            line_score, line_quad, line_area, line_support, line_minimum, line_edge_count = line_candidate
            agreement = self._quad_bbox_iou(line_quad, detector_bbox)
            strong_geometry = (
                line_score >= 1.65 and line_area >= 0.16
                and line_support >= 0.65 and line_minimum >= 0.42 and line_edge_count < 2
            )
            independent_geometry = (
                best[0] < 0.12 and line_score >= 1.85 and line_area >= 0.25
                and line_support >= 0.78 and line_minimum >= 0.50
            )
            if strong_geometry and (agreement >= 0.35 or independent_geometry):
                self.last_detection_mode = "直线几何四角"
                geometry_review_reasons = []
                if best[0] >= 0.50:
                    ordered_line = order_corners(line_quad)
                    top_width = float(np.linalg.norm(ordered_line[1] - ordered_line[0]))
                    bottom_width = float(np.linalg.norm(ordered_line[2] - ordered_line[3]))
                    detector_width = max(1.0, detector_bbox[2] - detector_bbox[0])
                    width_balance = min(top_width, bottom_width) / max(top_width, bottom_width)
                    if (
                        min(top_width, bottom_width) < 0.72 * detector_width
                        and width_balance < 0.84
                    ):
                        geometry_review_reasons.append("几何边界疑似落在画面内部")
                self_confirming_boundary = (
                    line_score >= 2.35 and line_support >= 0.97
                    and line_minimum >= 0.95 and line_edge_count == 0
                )
                if (
                    float(order_corners(line_quad)[:, 1].min()) < y1 - 0.05 * height
                    and not self_confirming_boundary
                ):
                    geometry_review_reasons.append("几何上边界越过检测画布，疑似包含标题或背景")
                self.last_detection_review_reason = "；".join(geometry_review_reasons)
                if strict and self.last_detection_review_reason:
                    return None
                geometry_confidence = float(np.clip(0.65 + (line_score - 1.65) * 0.22, 0.65, 0.98))
                return line_quad, "canvas front face", geometry_confidence, True

        padding_x, padding_y = (x2 - x1) * 0.018, (y2 - y1) * 0.018
        x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
        x2, y2 = min(width - 1, x2 + padding_x), min(height - 1, y2 + padding_y)
        # The refinement routine expands its own prompt adaptively.  Keep the
        # detector's raw box here; pre-expanding it can make SAM include the
        # header or the phone in tilted product composites.
        refined = self._refine_image_corners(image, detector_bbox)
        inner_face_adjusted = False
        if refined is not None:
            front_face = self._inset_visible_canvas_sides(image, refined)
            inner_face_adjusted = float(np.max(np.linalg.norm(
                order_corners(front_face) - order_corners(refined), axis=1
            ))) >= 2.0
            refined = front_face
        corners = refined if refined is not None else np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )
        review_reasons = []
        dark_frame_evidence = refined is not None and self._dark_frame_encloses_quad(image, corners)
        if (best[0] < 0.16 or confidence < 0.10) and not dark_frame_evidence:
            review_reasons.append("目标置信度不足")
        if refined is None:
            review_reasons.append("未获得精细四角")
        else:
            agreement = self._quad_bbox_iou(corners, detector_bbox)
            mean_support, minimum_support = self._quad_edge_metrics(image, corners)
            if agreement < 0.35 and not dark_frame_evidence:
                review_reasons.append("分割四角与目标框不一致")
            strong_segmented_object = (
                best[0] >= 0.50 and confidence >= 0.30
                and agreement >= 0.55 and mean_support >= 0.30
            )
            if (
                (mean_support < 0.18 or minimum_support < 0.03)
                and not strong_segmented_object and not dark_frame_evidence
            ):
                review_reasons.append("画布直边支撑不足")
        points = order_corners(corners)
        edge_count = sum((
            float(points[:, 0].min()) < 3,
            float(points[:, 1].min()) < 3,
            float(points[:, 0].max()) > width - 3,
            float(points[:, 1].max()) > height - 3,
        ))
        if edge_count >= 2:
            review_reasons.append("多个角点贴近原图边界")
        self.last_detection_mode = (
            "暗色相框内画面四角" if dark_frame_evidence
            else (
                "SAM 正面内框四角" if inner_face_adjusted
                else ("SAM 精细四角" if refined is not None else "检测框四角")
            )
        )
        self.last_detection_review_reason = "；".join(dict.fromkeys(review_reasons))
        if strict and self.last_detection_review_reason:
            return None
        return corners, label, confidence, refined is not None

    def auto_locate(self) -> None:
        if self.original is None:
            messagebox.showinfo("请先选择图片", "请先选择一张场景图片，再执行自动定位。")
            return
        if self.model_preloading:
            self.status.set("模型仍在后台加载，请稍候；窗口可以继续移动和查看图片。")
            return
        if self.inference_running or self.batch_running:
            messagebox.showinfo("任务进行中", "请等待当前定位或批量任务完成。")
            return

        image = self.original.copy()
        self.detection_ready = False
        self._set_processing(True)
        self.status.set(f"正在后台自动定位画布，推理设备：{self.compute_label}…")
        LOGGER.info("auto_locate_started shape=%s compute=%s", image.shape, self.compute_label)
        worker = threading.Thread(
            target=self._auto_locate_worker,
            args=(image,),
            name="auto-locate",
            daemon=True,
        )
        worker.start()

    def _auto_locate_worker(self, image: np.ndarray) -> None:
        try:
            found = self._detect_corners(image)
            payload = {
                "found": found,
                "mode": self.last_detection_mode,
                "review": self.last_detection_review_reason,
            }
            self._post_ui("auto_done", payload)
        except Exception as error:
            LOGGER.exception("auto_locate_exception")
            self._post_ui("auto_error", str(error))

    def _handle_auto_done(self, payload: object) -> None:
        details = payload if isinstance(payload, dict) else {}
        found = details.get("found")
        self.last_detection_mode = str(details.get("mode", ""))
        self.last_detection_review_reason = str(details.get("review", ""))
        if found is None:
            self.detection_ready = False
            self._finish_processing()
            self.status.set("自动定位未通过质量检查；该图片不会按错误区域导出。")
            return
        self.corners, label, confidence, refined = found
        self.detection_ready = True
        self._finish_processing()
        mode = self.last_detection_mode or ("已精细贴合边缘" if refined else "已给出初始边框")
        review = (
            f" 自动结果需要复核：{self.last_detection_review_reason}。"
            if self.last_detection_review_reason else ""
        )
        self.status.set(
            f"已自动定位为“{label}”（置信度 {confidence:.0%}），{mode}；当前推理：{self.compute_label}。"
            f"{review}边缘与透视参数已自动固定，无需手动微调。"
        )
        LOGGER.info(
            "auto_locate_completed label=%s confidence=%.4f refined=%s compute=%s review=%s",
            label,
            confidence,
            refined,
            self.compute_label,
            self.last_detection_review_reason,
        )
        self.redraw()

    def batch_folder(self) -> None:
        if self.model_preloading:
            self.status.set("模型仍在后台加载，请稍候；窗口没有卡死。")
            return
        if self.batch_running or self.inference_running:
            messagebox.showinfo("任务进行中", "当前定位或批量任务尚未完成。")
            return
        selected = filedialog.askdirectory(title="选择包含场景图的文件夹")
        if not selected:
            return
        input_dir = Path(selected)
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [
            path for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in image_suffixes
            and not any(part.startswith("智能裁图输出") for part in path.relative_to(input_dir).parts)
        ]
        if not images:
            messagebox.showinfo("未找到图片", "所选文件夹及其子文件夹中没有可处理的图片。")
            return
        output_dir = input_dir / "智能裁图输出"
        sequence = 2
        while output_dir.exists():
            output_dir = input_dir / f"智能裁图输出_{sequence}"
            sequence += 1
        if not messagebox.askyesno(
            "开始批量裁图",
            f"发现 {len(images)} 张图片。\n\n结果将保存至：\n{output_dir}\n\n自动定位失败的图片不会覆盖原图，并会记录在结果清单中。",
        ):
            return
        try:
            aspect_ratio = parse_aspect_ratio(self.aspect_ratio.get())
        except ValueError as error:
            messagebox.showerror("输出比例无效", str(error))
            return
        output_dir.mkdir(parents=True, exist_ok=False)
        self.batch_running = True
        self._set_processing(True)
        ratio_label = self.aspect_ratio.get().strip() or "不设置"
        self.status.set(f"正在后台准备批量处理（0/{len(images)}），推理设备：{self.compute_label}…")
        LOGGER.info(
            "batch_started input=%s output=%s images=%d ratio=%s compute=%s",
            input_dir,
            output_dir,
            len(images),
            ratio_label,
            self.compute_label,
        )
        worker = threading.Thread(
            target=self._batch_worker,
            args=(input_dir, images, output_dir, aspect_ratio, ratio_label),
            name="batch-inference",
            daemon=True,
        )
        worker.start()

    def _batch_worker(
        self,
        input_dir: Path,
        images: list[Path],
        output_dir: Path,
        aspect_ratio: float | None,
        ratio_label: str,
    ) -> None:
        records: list[list[object]] = []
        succeeded = 0
        review_count = 0
        review_dir = output_dir / "需要复核原图"
        try:
            for index, source in enumerate(images, start=1):
                self._post_ui(
                    "batch_progress",
                    f"正在后台批量处理（{index}/{len(images)}）：{source.name}；推理设备：{self.compute_label}",
                )
                image, read_error = read_image_file(source)
                if image is None:
                    records.append([
                        str(source.relative_to(input_dir)), "跳过",
                        f"图片无法读取：{read_error}", "", "", "",
                    ])
                    continue
                try:
                    found = self._detect_corners(image, strict=True)
                except Exception as error:
                    records.append([str(source.relative_to(input_dir)), "跳过", str(error), "", "", ""])
                    continue
                if found is None:
                    relative = source.relative_to(input_dir)
                    reason = self.last_detection_review_reason or "未找到可信画布或海报区域"
                    target_review = review_dir / relative
                    try:
                        target_review.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target_review)
                    except OSError as error:
                        records.append([str(relative), "跳过", f"复核原图复制失败：{error}", "", "", ""])
                        continue
                    review_count += 1
                    records.append([str(relative), "需要复核", reason, "", "", ""])
                    continue
                corners, label, confidence, refined = found
                crop = clean_edges(warp(image, corners), EDGE_TRIM_PERCENT)
                crop = crop_to_aspect(crop, aspect_ratio)
                height, width = crop.shape[:2]
                relative = source.relative_to(input_dir)
                target_dir = output_dir / relative.parent
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"{source.stem}_裁图.jpg"
                written, write_error = write_image_file(
                    target, crop, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                if not written:
                    records.append([
                        str(relative), "跳过", f"无法写入输出文件：{write_error}", "", "", "",
                    ])
                    continue
                succeeded += 1
                direction = "横图" if width >= height else "竖图"
                records.append([
                    str(relative), "完成", f"{label} / {self.last_detection_mode or ('精细边缘' if refined else '检测边框')}",
                    direction, f"{width}x{height}", f"{width / height:.4f}",
                ])
            report = output_dir / "批量处理结果.csv"
            with report.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["原图", "结果", "定位信息", "方向", "输出尺寸", "宽高比"])
                writer.writerows(records)
            skipped = len(images) - succeeded - review_count
            self._post_ui("batch_done", {
                "succeeded": succeeded,
                "review_count": review_count,
                "skipped": skipped,
                "ratio_label": ratio_label,
                "output_dir": output_dir,
                "review_dir": review_dir,
                "report": report,
            })
        except Exception as error:
            LOGGER.exception("batch_worker_exception")
            self._post_ui("batch_error", {"error": str(error), "output_dir": output_dir})

    def _handle_batch_done(self, payload: object) -> None:
        details = payload if isinstance(payload, dict) else {}
        self.batch_running = False
        self._finish_processing()
        succeeded = int(details.get("succeeded", 0))
        review_count = int(details.get("review_count", 0))
        skipped = int(details.get("skipped", 0))
        ratio_label = str(details.get("ratio_label", "不设置"))
        output_dir = details.get("output_dir", "")
        review_dir = details.get("review_dir", "")
        report = details.get("report", "")
        self.status.set(
            f"批量处理完成：自动通过 {succeeded} 张，需要复核 {review_count} 张，跳过 {skipped} 张；"
            f"输出比例：{ratio_label}；"
            f"推理设备：{self.compute_label}。"
        )
        LOGGER.info(
            "batch_completed succeeded=%d review=%d skipped=%d output=%s compute=%s",
            succeeded,
            review_count,
            skipped,
            output_dir,
            self.compute_label,
        )
        messagebox.showinfo(
            "批量处理完成",
            f"自动通过：{succeeded} 张\n需要复核：{review_count} 张\n跳过：{skipped} 张\n"
            f"输出比例：{ratio_label}\n\n输出文件夹：\n{output_dir}\n\n"
            f"复核原图：\n{review_dir}\n\n结果清单：\n{report}",
        )

    def export(self) -> None:
        if self.preview is None or self.image_path is None:
            messagebox.showinfo("没有可导出的图片", "先选择图片并确认裁切区域。")
            return
        default = f"{self.image_path.stem}_已裁图.jpg"
        output = filedialog.asksaveasfilename(
            title="保存裁切结果", initialfile=default, defaultextension=".jpg",
            filetypes=[("JPEG 图片", "*.jpg"), ("PNG 图片", "*.png")],
        )
        if not output:
            return
        written, write_error = write_image_file(output, self.preview)
        if not written:
            messagebox.showerror("保存失败", f"无法写入输出文件。\n\n原因：{write_error}")
            return
        self.status.set(f"已导出：{output}")
        messagebox.showinfo("导出完成", f"已保存裁切图：\n{output}")


def run_backend_check() -> int:
    import torch

    device, label = choose_compute_device()
    tensor_device = "cuda:0" if device == 0 else device
    value = torch.ones(1).to(tensor_device).cpu()
    print(f"runtime_mode={RUNTIME_MODE}")
    print(f"compute_label={label}")
    print(f"tensor_device={tensor_device}")
    print(f"tensor_ok={float(value[0]) == 1.0}")
    return 0


def run_self_test(input_path: str, output_path: str) -> int:
    app = SmartCropper()
    app.withdraw()
    try:
        image, read_error = read_image_file(input_path)
        if image is None:
            print(f"read_error={read_error}")
            return 2
        found = app._detect_corners(image, strict=True)
        if found is None:
            print(f"detection_review={app.last_detection_review_reason}")
            return 3
        corners, label, confidence, refined = found
        result = clean_edges(warp(image, corners), EDGE_TRIM_PERCENT)
        written, write_error = write_image_file(output_path, result)
        if not written:
            print(f"write_error={write_error}")
            return 4
        print(f"compute_label={app.compute_label}")
        print(f"label={label}")
        print(f"confidence={confidence:.6f}")
        print(f"refined={refined}")
        print(f"output={Path(output_path).resolve()}")
        return 0
    finally:
        app.destroy()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--backend-check":
        raise SystemExit(run_backend_check())
    if len(sys.argv) == 4 and sys.argv[1] == "--self-test":
        raise SystemExit(run_self_test(sys.argv[2], sys.argv[3]))
    SmartCropper().mainloop()
