"""Tests for CorridorKeyModule.pipeline — async I/O triple-buffered pipeline.

Tests use a mock engine (no GPU/weights needed) and tiny temporary image
sequences to verify pipeline correctness: frame ordering, error handling,
and output file generation.
"""

from __future__ import annotations

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from unittest.mock import MagicMock

import cv2
import numpy as np
import torch

from CorridorKeyModule.pipeline import (
    AsyncFramePipeline,
    FrameSource,
    InferenceRingBuffer,
    _write_frame,
)

# ---------------------------------------------------------------------------
# InferenceRingBuffer
# ---------------------------------------------------------------------------


class TestInferenceRingBuffer:
    def test_creates_three_slots(self):
        ring = InferenceRingBuffer(img_size=32, device=torch.device("cpu"))
        assert len(ring.cpu_input) == 3
        assert len(ring.meta) == 3

    def test_buffer_shapes(self):
        ring = InferenceRingBuffer(img_size=64, device=torch.device("cpu"))
        for buf in ring.cpu_input:
            assert buf.shape == (1, 4, 64, 64)
            assert buf.dtype == torch.float32

    def test_cpu_buffers_not_pinned_on_cpu_device(self):
        ring = InferenceRingBuffer(img_size=32, device=torch.device("cpu"))
        for buf in ring.cpu_input:
            assert not buf.is_pinned()


# ---------------------------------------------------------------------------
# _write_frame
# ---------------------------------------------------------------------------


class TestWriteFrame:
    def test_writes_all_output_files(self, tmp_path):
        h, w = 8, 8
        res = {
            "fg": np.full((h, w, 3), 0.5, dtype=np.float32),
            "alpha": np.full((h, w, 1), 0.8, dtype=np.float32),
            "comp": np.full((h, w, 3), 0.4, dtype=np.float32),
            "processed": np.full((h, w, 4), 0.3, dtype=np.float32),
        }
        dirs = {
            "fg": str(tmp_path / "FG"),
            "matte": str(tmp_path / "Matte"),
            "comp": str(tmp_path / "Comp"),
            "proc": str(tmp_path / "Processed"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        exr_flags = [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_PXR24,
        ]

        _write_frame(res, dirs, "00000", exr_flags)

        assert os.path.exists(os.path.join(dirs["fg"], "00000.exr"))
        assert os.path.exists(os.path.join(dirs["matte"], "00000.exr"))
        assert os.path.exists(os.path.join(dirs["comp"], "00000.png"))
        assert os.path.exists(os.path.join(dirs["proc"], "00000.exr"))


# ---------------------------------------------------------------------------
# AsyncFramePipeline
# ---------------------------------------------------------------------------


def _make_tiny_sequence(tmp_path, name, num_frames, channels=3):
    """Create a directory of tiny PNG images for testing."""
    seq_dir = tmp_path / name
    seq_dir.mkdir(parents=True, exist_ok=True)
    fnames = []
    for i in range(num_frames):
        fname = f"frame_{i:04d}.png"
        if channels == 1:
            img = np.full((8, 8), 128, dtype=np.uint8)
        else:
            img = np.full((8, 8, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(seq_dir / fname), img)
        fnames.append(fname)
    return str(seq_dir), fnames


def _mock_engine():
    """Mock engine that returns deterministic results matching input shape."""

    def fake_process_frame(image, mask_linear, **kwargs):
        h, w = image.shape[:2]
        return {
            "fg": np.full((h, w, 3), 0.5, dtype=np.float32),
            "alpha": np.full((h, w, 1), 0.8, dtype=np.float32),
            "comp": np.full((h, w, 3), 0.4, dtype=np.float32),
            "processed": np.full((h, w, 4), 0.3, dtype=np.float32),
        }

    engine = MagicMock()
    engine.process_frame = MagicMock(side_effect=fake_process_frame)
    engine.img_size = 64
    engine.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    engine.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    return engine


class TestAsyncFramePipeline:
    def test_processes_all_frames(self, tmp_path):
        num_frames = 5
        input_dir, input_files = _make_tiny_sequence(tmp_path, "Input", num_frames)
        alpha_dir, alpha_files = _make_tiny_sequence(tmp_path, "Alpha", num_frames, channels=1)

        engine = _mock_engine()

        out_dirs = {
            "fg": str(tmp_path / "FG"),
            "matte": str(tmp_path / "Matte"),
            "comp": str(tmp_path / "Comp"),
            "proc": str(tmp_path / "Processed"),
        }
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

        exr_flags = [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_PXR24,
        ]

        input_src = FrameSource(type="sequence", path=input_dir, files=input_files)
        alpha_src = FrameSource(type="sequence", path=alpha_dir, files=alpha_files)

        pipeline = AsyncFramePipeline(
            engine=engine,
            img_size=64,
            device=torch.device("cpu"),
            process_frame_kwargs={
                "input_is_linear": False,
                "fg_is_straight": True,
                "despill_strength": 1.0,
                "auto_despeckle": False,
                "despeckle_size": 400,
                "refiner_scale": 1.0,
            },
        )
        pipeline.process_clip(
            input_source=input_src,
            alpha_source=alpha_src,
            output_dirs=out_dirs,
            num_frames=num_frames,
            input_is_linear=False,
            exr_flags=exr_flags,
            mean=engine.mean,
            std=engine.std,
        )

        assert engine.process_frame.call_count == num_frames

        # Verify output files exist for each frame
        for i in range(num_frames):
            stem = f"frame_{i:04d}"
            assert os.path.exists(os.path.join(out_dirs["fg"], f"{stem}.exr"))
            assert os.path.exists(os.path.join(out_dirs["matte"], f"{stem}.exr"))
            assert os.path.exists(os.path.join(out_dirs["comp"], f"{stem}.png"))

    def test_handles_single_frame(self, tmp_path):
        input_dir, input_files = _make_tiny_sequence(tmp_path, "Input", 1)
        alpha_dir, alpha_files = _make_tiny_sequence(tmp_path, "Alpha", 1, channels=1)

        engine = _mock_engine()

        out_dirs = {
            "fg": str(tmp_path / "FG"),
            "matte": str(tmp_path / "Matte"),
            "comp": str(tmp_path / "Comp"),
            "proc": str(tmp_path / "Processed"),
        }
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

        exr_flags = [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_PXR24,
        ]

        pipeline = AsyncFramePipeline(
            engine=engine,
            img_size=64,
            device=torch.device("cpu"),
            process_frame_kwargs={
                "input_is_linear": False,
                "fg_is_straight": True,
                "despill_strength": 1.0,
                "auto_despeckle": False,
                "despeckle_size": 400,
                "refiner_scale": 1.0,
            },
        )
        pipeline.process_clip(
            input_source=FrameSource(type="sequence", path=input_dir, files=input_files),
            alpha_source=FrameSource(type="sequence", path=alpha_dir, files=alpha_files),
            output_dirs=out_dirs,
            num_frames=1,
            input_is_linear=False,
            exr_flags=exr_flags,
            mean=engine.mean,
            std=engine.std,
        )

        assert engine.process_frame.call_count == 1

    def test_skips_corrupt_frames(self, tmp_path):
        """Pipeline should skip frames that fail to read (corrupt files)."""
        num_frames = 3
        input_dir, input_files = _make_tiny_sequence(tmp_path, "Input", num_frames)
        alpha_dir, alpha_files = _make_tiny_sequence(tmp_path, "Alpha", num_frames, channels=1)

        # Corrupt the second input frame
        with open(os.path.join(input_dir, input_files[1]), "w") as f:
            f.write("not an image")

        engine = _mock_engine()

        out_dirs = {
            "fg": str(tmp_path / "FG"),
            "matte": str(tmp_path / "Matte"),
            "comp": str(tmp_path / "Comp"),
            "proc": str(tmp_path / "Processed"),
        }
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

        exr_flags = [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_PXR24,
        ]

        pipeline = AsyncFramePipeline(
            engine=engine,
            img_size=64,
            device=torch.device("cpu"),
            process_frame_kwargs={
                "input_is_linear": False,
                "fg_is_straight": True,
                "despill_strength": 1.0,
                "auto_despeckle": False,
                "despeckle_size": 400,
                "refiner_scale": 1.0,
            },
        )
        # Should not raise — corrupt frame is skipped
        pipeline.process_clip(
            input_source=FrameSource(type="sequence", path=input_dir, files=input_files),
            alpha_source=FrameSource(type="sequence", path=alpha_dir, files=alpha_files),
            output_dirs=out_dirs,
            num_frames=num_frames,
            input_is_linear=False,
            exr_flags=exr_flags,
            mean=engine.mean,
            std=engine.std,
        )

        # Only 2 frames should have been processed (frame 1 corrupt)
        assert engine.process_frame.call_count == 2

    def test_frame_ordering_preserved(self, tmp_path):
        """Frames should be processed in sequential order."""
        num_frames = 4
        input_dir, input_files = _make_tiny_sequence(tmp_path, "Input", num_frames)
        alpha_dir, alpha_files = _make_tiny_sequence(tmp_path, "Alpha", num_frames, channels=1)

        call_order = []
        original_process = _mock_engine().process_frame.side_effect

        def tracking_process(image, mask_linear, **kwargs):
            call_order.append(len(call_order))
            return original_process(image, mask_linear, **kwargs)

        engine = _mock_engine()
        engine.process_frame = MagicMock(side_effect=tracking_process)

        out_dirs = {
            "fg": str(tmp_path / "FG"),
            "matte": str(tmp_path / "Matte"),
            "comp": str(tmp_path / "Comp"),
            "proc": str(tmp_path / "Processed"),
        }
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

        exr_flags = [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_PXR24,
        ]

        pipeline = AsyncFramePipeline(
            engine=engine,
            img_size=64,
            device=torch.device("cpu"),
            process_frame_kwargs={
                "input_is_linear": False,
                "fg_is_straight": True,
                "despill_strength": 1.0,
                "auto_despeckle": False,
                "despeckle_size": 400,
                "refiner_scale": 1.0,
            },
        )
        pipeline.process_clip(
            input_source=FrameSource(type="sequence", path=input_dir, files=input_files),
            alpha_source=FrameSource(type="sequence", path=alpha_dir, files=alpha_files),
            output_dirs=out_dirs,
            num_frames=num_frames,
            input_is_linear=False,
            exr_flags=exr_flags,
            mean=engine.mean,
            std=engine.std,
        )

        assert call_order == [0, 1, 2, 3]
