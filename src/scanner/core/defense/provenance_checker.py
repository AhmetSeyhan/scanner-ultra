"""Scanner ULTRA — Content provenance checker (C2PA/EXIF)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ProvenanceChecker:
    async def check(self, content: bytes, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {
            "has_c2pa": False,
            "has_exif": False,
            "camera_make": None,
            "camera_model": None,
            "creation_date": None,
            "editing_software": None,
            "gps_data": False,
            "provenance_score": 0.5,
        }
        exif = self._extract_exif(content)
        if exif:
            results["has_exif"] = True
            results["camera_make"] = exif.get("Make")
            results["camera_model"] = exif.get("Model")
            results["creation_date"] = exif.get("DateTimeOriginal")
            results["gps_data"] = "GPSInfo" in exif
            results["editing_software"] = exif.get("Software")
        results["has_c2pa"] = b"c2pa" in content[:4096] or b"jumbf" in content[:4096]
        results["provenance_score"] = self._score(results)
        return results

    @staticmethod
    def _extract_exif(content: bytes) -> dict[str, Any] | None:
        try:
            import io

            from PIL import Image
            from PIL.ExifTags import TAGS

            img = Image.open(io.BytesIO(content))
            exif = img._getexif()
            if not exif:
                return None
            return {TAGS.get(k, k): v for k, v in exif.items() if isinstance(v, (str, int, float))}
        except Exception:
            return None

    @staticmethod
    def _score(r: dict[str, Any]) -> float:
        s = 0.3
        if r["has_c2pa"]:
            s += 0.3
        if r["has_exif"]:
            s += 0.1
            if r["camera_make"]:
                s += 0.1
            if r["creation_date"]:
                s += 0.05
        if r["editing_software"]:
            s -= 0.05
        return min(1.0, max(0.0, s))
