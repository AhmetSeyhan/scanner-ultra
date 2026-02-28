"""Tests for defense modules."""

import pytest


class TestHashDatabase:
    @pytest.mark.asyncio
    async def test_compute_hash(self):
        from scanner.core.defense.hash_database import HashDatabase

        db = HashDatabase()
        h = db.compute_hash(b"test content")
        assert len(h) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_store_and_lookup(self):
        from scanner.core.defense.hash_database import HashDatabase

        db = HashDatabase()
        h = db.compute_hash(b"test")
        await db.store(h, {"verdict": "fake"})
        result = await db.lookup(h)
        assert result is not None
        assert result["verdict"] == "fake"

    @pytest.mark.asyncio
    async def test_lookup_miss(self):
        from scanner.core.defense.hash_database import HashDatabase

        db = HashDatabase()
        result = await db.lookup("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_known_fake(self):
        from scanner.core.defense.hash_database import HashDatabase

        db = HashDatabase()
        assert not await db.is_known_fake("abc123")


class TestMetadataForensics:
    @pytest.mark.asyncio
    async def test_analyze_valid_jpeg(self):
        from scanner.core.defense.metadata_forensics import MetadataForensics

        mf = MetadataForensics()
        content = b"\xff\xd8\xff" + b"\x00" * 100
        result = await mf.analyze(content, "test.jpg")
        assert result["format_valid"]
        assert result["file_size"] > 0

    @pytest.mark.asyncio
    async def test_invalid_format(self):
        from scanner.core.defense.metadata_forensics import MetadataForensics

        mf = MetadataForensics()
        result = await mf.analyze(b"\x00\x00\x00\x00", "test.png")
        assert not result["format_valid"]

    @pytest.mark.asyncio
    async def test_ai_software_detection(self):
        from scanner.core.defense.metadata_forensics import MetadataForensics

        mf = MetadataForensics()
        content = b"header" + b"stable diffusion" + b"\x00" * 100
        result = await mf.analyze(content, "test.png")
        assert result["ai_software_detected"]
        assert result["ai_software_name"] == "stable diffusion"


class TestProvenanceChecker:
    @pytest.mark.asyncio
    async def test_no_exif(self):
        from scanner.core.defense.provenance_checker import ProvenanceChecker

        pc = ProvenanceChecker()
        result = await pc.check(b"\x00" * 100)
        assert not result["has_exif"]
        assert not result["has_c2pa"]

    @pytest.mark.asyncio
    async def test_c2pa_detection(self):
        from scanner.core.defense.provenance_checker import ProvenanceChecker

        pc = ProvenanceChecker()
        content = b"c2pa" + b"\x00" * 100
        result = await pc.check(content)
        assert result["has_c2pa"]
