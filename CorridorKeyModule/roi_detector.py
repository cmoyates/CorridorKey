"""Dynamic ROI (Region of Interest) subject detection using YOLOv11 on CPU.

Detects the subject bounding box in high-resolution frames without consuming
any GPU VRAM by running YOLOv11-Nano strictly on the CPU.
"""

from __future__ import annotations

import cv2
import numpy as np


# YOLO input resolution — the standard size YOLOv11 expects
YOLO_INPUT_SIZE = 640

# COCO class ID for "person" — the primary subject class for green screen work
PERSON_CLASS_ID = 0


class SubjectDetector:
    """Detects subject bounding boxes using YOLOv11-Nano on CPU.

    Downsamples the input to 640x640 for detection, then maps the resulting
    bounding box back to the original high-resolution coordinate space.
    """

    def __init__(self, model_name: str = "yolo11n.pt", confidence_threshold: float = 0.3) -> None:
        self.confidence_threshold = confidence_threshold
        self._model = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str):
        """Load YOLOv11-Nano and force CPU execution."""
        from ultralytics import YOLO

        model = YOLO(model_name)
        # Force CPU — zero GPU VRAM usage
        model.to("cpu")
        return model

    def detect(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Detect the primary subject in a high-resolution frame.

        Args:
            frame: Input image as [H, W, 3] numpy array (uint8 0-255 or float32 0-1).

        Returns:
            Bounding box (x1, y1, x2, y2) in original frame coordinates,
            or None if no subject detected.
        """
        original_height, original_width = frame.shape[:2]

        # Convert float32 [0-1] to uint8 [0-255] if needed (YOLO expects uint8)
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame_uint8 = frame

        # Downsample to 640x640 for YOLO
        scale_x = original_width / YOLO_INPUT_SIZE
        scale_y = original_height / YOLO_INPUT_SIZE
        small_frame = cv2.resize(frame_uint8, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

        # Run YOLO on CPU — verbose=False suppresses per-frame logging
        results = self._model.predict(small_frame, device="cpu", verbose=False, conf=self.confidence_threshold)

        if not results or len(results[0].boxes) == 0:
            return None

        # Find the best person detection (highest confidence)
        boxes = results[0].boxes
        best_box = None
        best_conf = 0.0

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            if class_id == PERSON_CLASS_ID and conf > best_conf:
                best_conf = conf
                best_box = boxes.xyxy[i]

        if best_box is None:
            # Fallback: use highest-confidence detection of any class
            best_idx = int(boxes.conf.argmax().item())
            best_box = boxes.xyxy[best_idx]

        # Map bounding box back to original resolution
        x1, y1, x2, y2 = best_box.cpu().numpy().flatten()
        x1 = int(round(x1 * scale_x))
        y1 = int(round(y1 * scale_y))
        x2 = int(round(x2 * scale_x))
        y2 = int(round(y2 * scale_y))

        # Clamp to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_width, x2)
        y2 = min(original_height, y2)

        # Sanity check: box must have positive area
        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)
