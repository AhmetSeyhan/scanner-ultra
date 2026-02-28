"""Scanner ULTRA — Metadata forensics analyzer."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

AI_SOFTWARE = [
    "stable diffusion",
    "dall-e",
    "midjourney",
    "comfyui",
    "automatic1111",
    "novelai",
    "deepfacelab",
    "faceswap",
    "roop",
    "simswap",
]


class MetadataForensics:
    async def analyze(self, content: bytes, filename: str) -> dict[str, Any]:
        results: dict[str, Any] = {
            "file_size": len(content),
            "filename": filename,
            "format_valid": True,
            "ai_software_detected": False,
            "ai_software_name": None,
            "metadata_inconsistencies": [],
            "forensic_score": 0.5,
        }
        results["format_valid"] = self._validate_format(content, filename)
        ai = self._check_ai_software(content)
        results["ai_software_detected"] = ai["detected"]
        results["ai_software_name"] = ai["name"]
        results["metadata_inconsistencies"] = self._check_consistency(content, filename)
        results["forensic_score"] = self._score(results)
        return results

    @staticmethod
    def _validate_format(content: bytes, filename: str) -> bool:
        if len(content) < 4:
            return False
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        magic_map = {
            "jpg": [b"\xff\xd8\xff"],
            "jpeg": [b"\xff\xd8\xff"],
            "png": [b"\x89PNG"],
            "gif": [b"GIF8"],
            "webp": [b"RIFF"],
            "mp4": [b"\x00\x00\x00\x18", b"\x00\x00\x00\x1c", b"\x00\x00\x00\x20"],
        }
        expected = magic_map.get(ext, [])
        if not expected:
            return True
        return any(content[:4].startswith(m) for m in expected)

    @staticmethod
    def _check_ai_software(content: bytes) -> dict[str, Any]:
        header = content[:65536].lower()
        for sig in AI_SOFTWARE:
            if sig.encode() in header:
                return {"detected": True, "name": sig}
        return {"detected": False, "name": None}

    @staticmethod
    def _check_consistency(content: bytes, filename: str) -> list[str]:
        issues = []
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ("jpg", "jpeg", "png") and len(content) < 1000:
            issues.append("File suspiciously small for image format")
        if ext in ("mp4", "avi", "mov") and len(content) < 10000:
            issues.append("File suspiciously small for video format")
        return issues

    @staticmethod
    def _score(r: dict[str, Any]) -> float:
        s = 0.0
        if r["ai_software_detected"]:
            s += 0.5
        if not r["format_valid"]:
            s += 0.2
        s += len(r["metadata_inconsistencies"]) * 0.1
        return min(1.0, s)
