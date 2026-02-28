"""Scanner ULTRA — Image preprocessing.

Face detection, alignment, resizing, and normalization.
Uses MediaPipe Tasks API with OpenCV Haar cascade fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceRegion:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float = 0.0
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    aligned_face: np.ndarray | None = None


@dataclass
class ImageData:
    original: np.ndarray | None = None
    faces: list[FaceRegion] = field(default_factory=list)
    resized: np.ndarray | None = None
    normalized: np.ndarray | None = None
    width: int = 0
    height: int = 0


class ImageProcessor:
    def __init__(self, face_size: tuple[int, int] = (224, 224), min_confidence: float = 0.5) -> None:
        self.face_size = face_size
        self.min_confidence = min_confidence
        self._face_detector = None

    async def process(self, content: bytes, filename: str | None = None) -> ImageData:
        arr = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {filename}")
        return await self.process_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    async def process_image(self, image: np.ndarray) -> ImageData:
        h, w = image.shape[:2]
        faces = self._detect_faces(image)
        for face in faces:
            x, y, fw, fh = face.bbox
            x, y = max(0, x), max(0, y)
            crop = image[y : y + fh, x : x + fw]
            if crop.size > 0:
                face.aligned_face = cv2.resize(crop, self.face_size)
        resized = cv2.resize(image, self.face_size)
        return ImageData(
            original=image,
            faces=faces,
            resized=resized,
            normalized=resized.astype(np.float32) / 255.0,
            width=w,
            height=h,
        )

    def process_frame(self, frame: np.ndarray) -> ImageData:
        h, w = frame.shape[:2]
        faces = self._detect_faces(frame)
        for face in faces:
            x, y, fw, fh = face.bbox
            x, y = max(0, x), max(0, y)
            crop = frame[y : y + fh, x : x + fw]
            if crop.size > 0:
                face.aligned_face = cv2.resize(crop, self.face_size)
        resized = cv2.resize(frame, self.face_size)
        return ImageData(
            original=frame,
            faces=faces,
            resized=resized,
            normalized=resized.astype(np.float32) / 255.0,
            width=w,
            height=h,
        )

    def _detect_faces(self, image: np.ndarray) -> list[FaceRegion]:
        try:
            return self._detect_faces_mediapipe(image)
        except Exception:
            return self._detect_faces_opencv(image)

    def _detect_faces_mediapipe(self, image: np.ndarray) -> list[FaceRegion]:
        import mediapipe as mp

        if self._face_detector is None:
            base_options = mp.tasks.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=base_options, min_detection_confidence=self.min_confidence
            )
            self._face_detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self._face_detector.detect(mp_image)
        faces = []
        h, w = image.shape[:2]
        for detection in result.detections:
            bb = detection.bounding_box
            face = FaceRegion(
                bbox=(bb.origin_x, bb.origin_y, bb.width, bb.height),
                confidence=detection.categories[0].score if detection.categories else 0.0,
            )
            if detection.keypoints:
                face.landmarks = [(kp.x * w, kp.y * h) for kp in detection.keypoints]
            faces.append(face)
        return faces

    @staticmethod
    def _detect_faces_opencv(image: np.ndarray) -> list[FaceRegion]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return [FaceRegion(bbox=(int(x), int(y), int(w), int(h)), confidence=0.8) for (x, y, w, h) in detections]
