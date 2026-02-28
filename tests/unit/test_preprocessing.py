"""Tests for preprocessing modules."""

import numpy as np
import pytest


class TestTextProcessor:
    @pytest.mark.asyncio
    async def test_process_string(self):
        from scanner.preprocessing.text_processor import TextProcessor

        tp = TextProcessor()
        result = await tp.process("Hello world. This is a test sentence.", "test.txt")
        assert result.word_count == 7
        assert result.sentence_count == 2
        assert result.cleaned == "Hello world. This is a test sentence."

    @pytest.mark.asyncio
    async def test_process_bytes(self):
        from scanner.preprocessing.text_processor import TextProcessor

        tp = TextProcessor()
        result = await tp.process(b"Hello world test.", "test.txt")
        assert result.word_count == 3

    @pytest.mark.asyncio
    async def test_chunking(self):
        from scanner.preprocessing.text_processor import TextProcessor

        tp = TextProcessor(chunk_size=5, chunk_overlap=1)
        text = " ".join(["word"] * 15)
        result = await tp.process(text, "test.txt")
        assert len(result.chunks) > 1

    @pytest.mark.asyncio
    async def test_empty_text(self):
        from scanner.preprocessing.text_processor import TextProcessor

        tp = TextProcessor()
        result = await tp.process("", "test.txt")
        assert result.word_count == 0


class TestQualityAdapter:
    def test_assess_image(self):
        from scanner.preprocessing.quality_adapter import QualityAdapter, QualityLevel

        qa = QualityAdapter()
        # High resolution image
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        report = qa.assess_image(img)
        assert report.resolution_score == 1.0
        assert report.level in (QualityLevel.HIGH, QualityLevel.MEDIUM)

    def test_assess_low_res(self):
        from scanner.preprocessing.quality_adapter import QualityAdapter

        qa = QualityAdapter()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        report = qa.assess_image(img)
        assert report.resolution_score < 0.5

    def test_assess_audio(self):
        from scanner.preprocessing.quality_adapter import QualityAdapter

        qa = QualityAdapter()
        waveform = np.random.randn(16000 * 5).astype(np.float32) * 0.05
        report = qa.assess_audio(waveform, 16000)
        assert 0.0 <= report.overall_score <= 1.0

    def test_assess_no_audio(self):
        from scanner.preprocessing.quality_adapter import QualityAdapter, QualityLevel

        qa = QualityAdapter()
        report = qa.assess_audio(None, 0)
        assert report.level == QualityLevel.VERY_LOW

    def test_confidence_weight(self):
        from scanner.preprocessing.quality_adapter import QualityAdapter, QualityLevel, QualityReport

        qa = QualityAdapter()
        assert qa.get_confidence_weight(QualityReport(level=QualityLevel.HIGH)) == 1.0
        assert qa.get_confidence_weight(QualityReport(level=QualityLevel.VERY_LOW)) == 0.3
