"""Scanner ULTRA — Scan orchestrator (10-step pipeline).

1. Media Router          → detect type
2. Preprocessing         → extract frames/audio/text
3. Quality Adaptation    → assess input quality
4. Hash DB Check         → cache / blocklist lookup
5. Core Detection        → run all enabled detectors (30s timeout)
6. Fusion                → cross-modal attention + trust score
6.5. PentaShield         → HYDRA + SENTINEL analysis (FAZ 2)
7. Explainability        → generate explanations
8. Cache Result          → store in hash DB
9. Report                → build ScanResult
10. Return
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import time
import uuid
from typing import Any

from scanner.core.base_detector import DetectorInput, DetectorResult
from scanner.core.defense.hash_database import HashDatabase
from scanner.core.defense.metadata_forensics import MetadataForensics
from scanner.core.defense.provenance_checker import ProvenanceChecker
from scanner.core.fusion.confidence_calibrator import ConfidenceCalibrator
from scanner.core.fusion.cross_modal_attention import CrossModalAttention
from scanner.core.fusion.trust_score_engine import TrustScoreEngine
from scanner.models.enums import (
    DetectorCapability,
    MediaType,
)
from scanner.models.registry import DetectorRegistry
from scanner.models.schemas import ScanResult
from scanner.pentashield.engine import PentaShieldEngine

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30  # seconds per detector


class ScanOrchestrator:
    """Main scan pipeline orchestrator."""

    def __init__(
        self,
        detector_timeout: int = DEFAULT_TIMEOUT,
        redis_url: str | None = None,
        cache_ttl: int = 86400,
    ) -> None:
        self.registry = DetectorRegistry()
        self.detector_timeout = detector_timeout
        self.hash_db = HashDatabase(redis_url=redis_url, ttl=cache_ttl)
        self.fusion = CrossModalAttention()
        self.calibrator = ConfidenceCalibrator()
        self.trust_engine = TrustScoreEngine()
        self.provenance = ProvenanceChecker()
        self.metadata_forensics = MetadataForensics()
        self.pentashield = PentaShieldEngine()

    async def scan(
        self,
        content: bytes,
        filename: str,
        media_type_hint: str | None = None,
    ) -> ScanResult:
        """Run the full 10-step detection pipeline."""
        start = time.perf_counter()
        scan_id = f"scn_{uuid.uuid4().hex[:12]}"
        correlation_id = uuid.uuid4().hex

        logger.info("[%s] Scan started — file=%s size=%d", scan_id, filename, len(content))

        # === Step 1: Media Router ===
        media_type = self._detect_media_type(filename, media_type_hint)
        logger.info("[%s] Media type: %s", scan_id, media_type.value)

        # === Step 2: Preprocessing ===
        detector_input = await self._preprocess(content, filename, media_type)

        # === Step 3: Quality Adaptation ===
        quality_weight = self._assess_quality(detector_input, media_type)

        # === Step 4: Hash DB Check ===
        content_hash = hashlib.sha256(content).hexdigest()
        cached = await self.hash_db.lookup(content_hash)
        if cached:
            logger.info("[%s] Cache HIT — returning cached result", scan_id)
            return ScanResult(**cached)

        known_fake = await self.hash_db.is_known_fake(content_hash)

        # === Step 5: Core Detection ===
        visual_results, audio_results, text_results = await self._run_detectors(detector_input, media_type)

        # Defense checks (parallel)
        defense_results = await self._run_defense(content, filename)

        # === Step 6: Fusion ===
        fused = self.fusion.fuse(visual_results, audio_results, text_results)
        fused_score = fused["fused_score"]
        fused_confidence = fused["confidence"]

        # Apply quality weight
        fused_confidence *= quality_weight

        # Known fake override
        if known_fake:
            fused_score = 0.95
            fused_confidence = 0.99

        # Trust score + verdict
        trust_result = self.trust_engine.compute(fused_score, fused_confidence)
        verdict = trust_result["verdict"]
        threat_level = trust_result["threat_level"]
        trust_score = trust_result["trust_score"]

        # Merge all detector results
        all_results: dict[str, dict[str, Any]] = {}
        for group in [visual_results, audio_results, text_results]:
            all_results.update(group)

        # === Step 6.5: PentaShield Analysis ===
        pentashield_result = await self.pentashield.analyze(
            detector_results=all_results,
            fused_score=fused_score,
            fused_confidence=fused_confidence,
            media_type=media_type,
            frames=detector_input.frames,
            fps=detector_input.fps if detector_input.fps else 30.0,
            defense_results=defense_results,
        )

        # Override verdict if PentaShield demands it
        pentashield_data = pentashield_result.model_dump()
        if pentashield_result.override_verdict is not None:
            verdict = pentashield_result.override_verdict
            trust_result = self.trust_engine.compute(fused_score, fused_confidence)
            threat_level = trust_result["threat_level"]
            trust_score = trust_result["trust_score"]

        # === Step 7: Explainability ===
        explanation = {
            **trust_result.get("explanation", {}),
            "content_hash": content_hash,
            "correlation_id": correlation_id,
            "fusion": fused,
            "defense": defense_results,
            "quality_weight": round(quality_weight, 4),
        }

        elapsed = (time.perf_counter() - start) * 1000

        # === Step 9: Report ===
        result = ScanResult(
            scan_id=scan_id,
            media_type=media_type,
            verdict=verdict,
            trust_score=trust_score,
            confidence=fused_confidence,
            threat_level=threat_level,
            detector_results=all_results,
            pentashield=pentashield_data,
            attribution=pentashield_data.get("forensic_dna", {}).get("attribution_report"),
            explanation=explanation,
            processing_time_ms=elapsed,
        )

        # === Step 8: Cache ===
        await self.hash_db.store(content_hash, result.model_dump(mode="json"))

        # Store for API results endpoint
        from scanner.api.v1.results import store_result

        store_result(scan_id, result.model_dump(mode="json"))

        logger.info("[%s] Scan complete — verdict=%s trust=%.2f (%.0fms)", scan_id, verdict.value, trust_score, elapsed)
        return result

    # ── Step 2: Preprocessing ──

    async def _preprocess(self, content: bytes, filename: str, media_type: MediaType) -> DetectorInput:
        """Run type-specific preprocessing."""
        frames = None
        fps = 0.0
        audio_waveform = None
        audio_sr = 0
        image = None
        text = None

        try:
            if media_type == MediaType.VIDEO:
                from scanner.preprocessing.video_processor import VideoProcessor

                vp = VideoProcessor()
                vdata = await vp.process(content, filename)
                frames = vdata.frames
                fps = vdata.fps
                audio_waveform = vdata.audio_waveform
                audio_sr = vdata.audio_sr

            elif media_type == MediaType.IMAGE:
                from scanner.preprocessing.image_processor import ImageProcessor

                ip = ImageProcessor()
                idata = await ip.process(content, filename)
                image = idata.original
                if idata.faces:
                    frames = [f.aligned_face for f in idata.faces if f.aligned_face is not None]

            elif media_type == MediaType.AUDIO:
                from scanner.preprocessing.audio_processor import AudioProcessor

                ap = AudioProcessor()
                adata = await ap.process(content, filename)
                audio_waveform = adata.waveform
                audio_sr = adata.sr

            elif media_type == MediaType.TEXT:
                from scanner.preprocessing.text_processor import TextProcessor

                tp = TextProcessor()
                tdata = await tp.process(content, filename)
                text = tdata.cleaned

        except Exception:
            logger.exception("Preprocessing failed for %s", filename)

        return DetectorInput(
            frames=frames,
            fps=fps,
            video_path=None,
            audio_waveform=audio_waveform,
            audio_sr=audio_sr,
            image=image,
            text=text,
            metadata={"filename": filename, "media_type": media_type.value},
        )

    # ── Step 3: Quality ──

    def _assess_quality(self, inp: DetectorInput, media_type: MediaType) -> float:
        """Assess input quality and return confidence weight."""
        try:
            from scanner.preprocessing.quality_adapter import QualityAdapter

            qa = QualityAdapter()
            if media_type == MediaType.VIDEO and inp.frames:
                report = qa.assess_frames(inp.frames)
            elif media_type == MediaType.IMAGE and inp.image is not None:
                report = qa.assess_image(inp.image)
            elif media_type == MediaType.AUDIO and inp.audio_waveform is not None:
                report = qa.assess_audio(inp.audio_waveform, inp.audio_sr)
            else:
                return 0.8
            return qa.get_confidence_weight(report)
        except Exception:
            return 0.8

    # ── Step 5: Core Detection ──

    async def _run_detectors(
        self, inp: DetectorInput, media_type: MediaType
    ) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
        """Run all enabled detectors grouped by type."""
        detectors = self.registry.get_enabled()
        if not detectors:
            logger.warning("No detectors registered")
            return {}, {}, {}

        # Select detectors by media type
        visual_dets = []
        audio_dets = []
        text_dets = []

        for det in detectors:
            caps = det.capabilities
            if media_type == MediaType.VIDEO:
                if DetectorCapability.VIDEO_FRAMES in caps:
                    visual_dets.append(det)
                if DetectorCapability.AUDIO_TRACK in caps:
                    audio_dets.append(det)
                if DetectorCapability.AV_SYNC in caps:
                    audio_dets.append(det)
            elif media_type == MediaType.IMAGE:
                if DetectorCapability.SINGLE_IMAGE in caps or DetectorCapability.VIDEO_FRAMES in caps:
                    visual_dets.append(det)
            elif media_type == MediaType.AUDIO:
                if DetectorCapability.AUDIO_TRACK in caps:
                    audio_dets.append(det)
            elif media_type == MediaType.TEXT:
                if DetectorCapability.TEXT_CONTENT in caps:
                    text_dets.append(det)

        # Run all detectors with timeout
        visual_results = await self._run_group(visual_dets, inp)
        audio_results = await self._run_group(audio_dets, inp)
        text_results = await self._run_group(text_dets, inp)

        # Calibrate
        visual_results = self.calibrator.calibrate_batch(visual_results)
        audio_results = self.calibrator.calibrate_batch(audio_results)
        text_results = self.calibrator.calibrate_batch(text_results)

        return visual_results, audio_results, text_results

    async def _run_group(self, detectors: list, inp: DetectorInput) -> dict[str, dict[str, Any]]:
        """Run a group of detectors concurrently with timeout."""
        if not detectors:
            return {}

        async def _run_one(det: Any) -> tuple[str, dict[str, Any]]:
            try:
                result: DetectorResult = await asyncio.wait_for(det.detect(inp), timeout=self.detector_timeout)
                return det.name, result.to_dict()
            except asyncio.TimeoutError:
                logger.warning("Detector %s timed out after %ds", det.name, self.detector_timeout)
                return det.name, {
                    "detector_name": det.name,
                    "score": 0.5,
                    "confidence": 0.0,
                    "method": f"{det.name}_timeout",
                    "status": "ERROR",
                    "details": {"error": "timeout"},
                }
            except Exception as exc:
                logger.error("Detector %s failed: %s", det.name, exc)
                return det.name, {
                    "detector_name": det.name,
                    "score": 0.5,
                    "confidence": 0.0,
                    "method": f"{det.name}_error",
                    "status": "ERROR",
                    "details": {"error": str(exc)},
                }

        tasks = [_run_one(d) for d in detectors]
        results = await asyncio.gather(*tasks)
        return dict(results)

    # ── Defense ──

    async def _run_defense(self, content: bytes, filename: str) -> dict[str, Any]:
        """Run defense checks."""
        try:
            prov_task = self.provenance.check(content)
            meta_task = self.metadata_forensics.analyze(content, filename)
            prov_result, meta_result = await asyncio.gather(prov_task, meta_task)
            return {"provenance": prov_result, "metadata_forensics": meta_result}
        except Exception as exc:
            logger.error("Defense checks failed: %s", exc)
            return {"error": str(exc)}

    # ── Step 1: Media Router ──

    @staticmethod
    def _detect_media_type(filename: str, hint: str | None) -> MediaType:
        """Auto-detect media type from filename or hint."""
        if hint:
            try:
                return MediaType(hint)
            except ValueError:
                pass

        mime, _ = mimetypes.guess_type(filename)
        if mime:
            if mime.startswith("video/"):
                return MediaType.VIDEO
            if mime.startswith("image/"):
                return MediaType.IMAGE
            if mime.startswith("audio/"):
                return MediaType.AUDIO
            if mime.startswith("text/"):
                return MediaType.TEXT

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        ext_map = {
            "mp4": MediaType.VIDEO,
            "avi": MediaType.VIDEO,
            "mov": MediaType.VIDEO,
            "mkv": MediaType.VIDEO,
            "webm": MediaType.VIDEO,
            "jpg": MediaType.IMAGE,
            "jpeg": MediaType.IMAGE,
            "png": MediaType.IMAGE,
            "webp": MediaType.IMAGE,
            "bmp": MediaType.IMAGE,
            "mp3": MediaType.AUDIO,
            "wav": MediaType.AUDIO,
            "flac": MediaType.AUDIO,
            "ogg": MediaType.AUDIO,
            "txt": MediaType.TEXT,
            "md": MediaType.TEXT,
        }
        return ext_map.get(ext, MediaType.IMAGE)
