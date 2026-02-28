"""Scanner ULTRA — Video preprocessing.

Extracts frames, audio track, and metadata from video files.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_sec: float = 0.0
    codec: str = ""
    has_audio: bool = False


@dataclass
class VideoData:
    frames: list[np.ndarray] = field(default_factory=list)
    fps: float = 0.0
    audio_waveform: np.ndarray | None = None
    audio_sr: int = 0
    meta: VideoMeta = field(default_factory=VideoMeta)
    temp_path: str | None = None


class VideoProcessor:
    """Extract frames and audio from video files."""

    def __init__(self, max_frames: int = 32, target_fps: float = 8.0) -> None:
        self.max_frames = max_frames
        self.target_fps = target_fps

    async def process(self, content: bytes, filename: str) -> VideoData:
        suffix = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        meta = self._extract_meta(tmp_path)
        frames = self._extract_frames(tmp_path, meta)
        audio_waveform, audio_sr = self._extract_audio(tmp_path)

        return VideoData(
            frames=frames,
            fps=meta.fps,
            audio_waveform=audio_waveform,
            audio_sr=audio_sr,
            meta=meta,
            temp_path=tmp_path,
        )

    def _extract_meta(self, path: str) -> VideoMeta:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        try:
            meta = VideoMeta(
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                codec=self._fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC))),
            )
            meta.duration_sec = meta.frame_count / meta.fps if meta.fps > 0 else 0.0
            meta.has_audio = self._check_audio_stream(path)
            return meta
        finally:
            cap.release()

    def _extract_frames(self, path: str, meta: VideoMeta) -> list[np.ndarray]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        try:
            total = meta.frame_count
            if total <= 0:
                return []
            step = max(1, int(meta.fps / self.target_fps)) if meta.fps > self.target_fps else 1
            indices = list(range(0, total, step))[: self.max_frames]
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            logger.info("Extracted %d frames from %d total", len(frames), total)
            return frames
        finally:
            cap.release()

    @staticmethod
    def _extract_audio(path: str) -> tuple[np.ndarray | None, int]:
        sr = 16000
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", path, "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", "-f", "s16le", "-"],
                capture_output=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0 or not result.stdout:
                return None, 0
            waveform = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return waveform, sr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("ffmpeg not available — skipping audio extraction")
            return None, 0

    @staticmethod
    def _check_audio_stream(path: str) -> bool:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return "audio" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _fourcc_to_str(fourcc: int) -> str:
        return "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
