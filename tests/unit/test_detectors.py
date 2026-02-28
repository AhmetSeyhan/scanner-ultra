"""Tests for detector modules (stub mode)."""

import numpy as np
import pytest

from scanner.core.base_detector import DetectorInput
from scanner.models.enums import DetectorStatus, DetectorType


class TestEfficientNetDetector:
    @pytest.mark.asyncio
    async def test_stub_mode(self):
        from scanner.core.visual.efficientnet_detector import EfficientNetDetector

        det = EfficientNetDetector(device="cpu")
        await det.ensure_loaded()
        assert det.name == "efficientnet_b0"
        assert det.detector_type == DetectorType.VISUAL

    @pytest.mark.asyncio
    async def test_no_frames_skip(self):
        from scanner.core.visual.efficientnet_detector import EfficientNetDetector

        det = EfficientNetDetector(device="cpu")
        result = await det.detect(DetectorInput())
        assert result.status == DetectorStatus.SKIPPED
        assert result.score == 0.5


class TestCLIPDetector:
    @pytest.mark.asyncio
    async def test_stub_mode(self):
        from scanner.core.visual.clip_detector import CLIPDetector

        det = CLIPDetector(device="cpu")
        await det.ensure_loaded()
        assert det.name == "clip_forensic"

    @pytest.mark.asyncio
    async def test_no_frames_skip(self):
        from scanner.core.visual.clip_detector import CLIPDetector

        det = CLIPDetector(device="cpu")
        result = await det.detect(DetectorInput())
        assert result.status == DetectorStatus.SKIPPED


class TestFrequencyDetector:
    @pytest.mark.asyncio
    async def test_detection(self):
        from scanner.core.visual.frequency_detector import FrequencyDetector

        det = FrequencyDetector(device="cpu")
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        inp = DetectorInput(frames=[frame])
        result = await det.detect(inp)
        assert result.status == DetectorStatus.PASS
        assert 0.0 <= result.score <= 1.0


class TestGANArtifactDetector:
    @pytest.mark.asyncio
    async def test_detection(self):
        from scanner.core.visual.gan_artifact_detector import GANArtifactDetector

        det = GANArtifactDetector(device="cpu")
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = await det.detect(DetectorInput(frames=[frame]))
        assert result.status == DetectorStatus.PASS


class TestPPGBioDetector:
    @pytest.mark.asyncio
    async def test_too_few_frames(self):
        from scanner.core.visual.ppg_bio_detector import PPGBioDetector

        det = PPGBioDetector(device="cpu")
        result = await det.detect(DetectorInput(frames=[np.zeros((64, 64, 3), dtype=np.uint8)]))
        assert result.status == DetectorStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_enough_frames(self):
        from scanner.core.visual.ppg_bio_detector import PPGBioDetector

        det = PPGBioDetector(device="cpu")
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(16)]
        result = await det.detect(DetectorInput(frames=frames, fps=30.0))
        assert result.status in (DetectorStatus.PASS, DetectorStatus.WARN)


class TestCQTDetector:
    @pytest.mark.asyncio
    async def test_detection(self):
        from scanner.core.audio.cqt_detector import CQTDetector

        det = CQTDetector(device="cpu")
        waveform = np.random.randn(16000 * 2).astype(np.float32)
        result = await det.detect(DetectorInput(audio_waveform=waveform, audio_sr=16000))
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_no_audio_skip(self):
        from scanner.core.audio.cqt_detector import CQTDetector

        det = CQTDetector(device="cpu")
        result = await det.detect(DetectorInput())
        assert result.status == DetectorStatus.SKIPPED


class TestVoiceCloneDetector:
    @pytest.mark.asyncio
    async def test_detection(self):
        from scanner.core.audio.voice_clone_detector import VoiceCloneDetector

        det = VoiceCloneDetector(device="cpu")
        waveform = np.random.randn(16000 * 3).astype(np.float32)
        result = await det.detect(DetectorInput(audio_waveform=waveform, audio_sr=16000))
        assert result.status == DetectorStatus.PASS


class TestAITextDetector:
    @pytest.mark.asyncio
    async def test_detection(self):
        from scanner.core.text.ai_text_detector import AITextDetector

        det = AITextDetector(device="cpu")
        text = "This is a test sentence. " * 20
        result = await det.detect(DetectorInput(text=text))
        assert result.status == DetectorStatus.PASS
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_short_text_skip(self):
        from scanner.core.text.ai_text_detector import AITextDetector

        det = AITextDetector(device="cpu")
        result = await det.detect(DetectorInput(text="hi"))
        assert result.status == DetectorStatus.SKIPPED
