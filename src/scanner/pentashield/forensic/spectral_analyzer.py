"""Scanner ULTRA — Spectral analysis for generator fingerprinting.

Analyzes FFT/DCT spectral signatures to identify GAN/diffusion generator fingerprints.
Each generator (StyleGAN, Stable Diffusion, etc.) leaves unique frequency-domain signatures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Generator spectral profiles (hardcoded heuristics)
GENERATOR_PROFILES = {
    "stylegan2": {
        "band_low": (0.25, 0.35),
        "band_mid": (0.15, 0.25),
        "band_high": (0.10, 0.20),
        "periodic_peaks": True,
        "centroid_range": (0.25, 0.40),
    },
    "stable_diffusion": {
        "band_low": (0.30, 0.45),
        "band_mid": (0.20, 0.30),
        "band_high": (0.05, 0.15),
        "periodic_peaks": False,
        "centroid_range": (0.20, 0.35),
    },
    "deepfacelab": {
        "band_low": (0.20, 0.30),
        "band_mid": (0.25, 0.35),
        "band_high": (0.15, 0.30),
        "periodic_peaks": False,
        "centroid_range": (0.30, 0.45),
    },
    "dalle": {
        "band_low": (0.35, 0.50),
        "band_mid": (0.15, 0.25),
        "band_high": (0.05, 0.12),
        "periodic_peaks": False,
        "centroid_range": (0.18, 0.32),
    },
}


@dataclass
class SpectralFingerprint:
    """Spectral signature extracted from a frame."""

    band_energies: dict[str, float]
    centroid: float
    has_periodic_peaks: bool
    azimuthal_profile: np.ndarray


@dataclass
class SpectralResult:
    """Result of spectral analysis."""

    fingerprints: list[SpectralFingerprint]
    best_match: str | None
    best_match_score: float
    all_matches: list[dict[str, Any]]


class SpectralAnalyzer:
    """Analyzes frequency-domain signatures for generator identification."""

    def __init__(self, profile_db: dict[str, dict] | None = None) -> None:
        """Initialize with optional custom generator profiles.

        Args:
            profile_db: Custom generator profiles. If None, uses GENERATOR_PROFILES.
        """
        self.profile_db = profile_db or GENERATOR_PROFILES

    def analyze(
        self,
        frames: list[np.ndarray] | None,
        freq_details: dict[str, Any] | None = None,
    ) -> SpectralResult:
        """Extract spectral fingerprints and match against known generators.

        Args:
            frames: Video/image frames (RGB). If None, returns empty result.
            freq_details: Optional pre-computed frequency analysis from FrequencyDetector.

        Returns:
            SpectralResult with fingerprints and generator matches.
        """
        if not frames:
            logger.warning("No frames provided for spectral analysis")
            return SpectralResult(
                fingerprints=[],
                best_match=None,
                best_match_score=0.0,
                all_matches=[],
            )

        # Extract fingerprints from frames (sample up to 4 frames)
        fingerprints = [self._extract_fingerprint(f) for f in frames[:4]]

        # Match against known profiles
        matches = self._match_profiles(fingerprints)

        # Best match
        best = matches[0] if matches else {"name": None, "score": 0.0}

        return SpectralResult(
            fingerprints=fingerprints,
            best_match=best["name"],
            best_match_score=best["score"],
            all_matches=matches,
        )

    def _extract_fingerprint(self, frame: np.ndarray) -> SpectralFingerprint:
        """Extract spectral fingerprint from a single frame.

        Args:
            frame: RGB frame (H,W,3)

        Returns:
            SpectralFingerprint with band energies, centroid, periodic peaks
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame

        # FFT
        dft = np.fft.fft2(gray.astype(np.float32))
        magnitude = np.abs(np.fft.fftshift(dft))
        magnitude = np.log1p(magnitude)  # Log scale

        # Azimuthal average (radial profile)
        radial_profile = self._azimuthal_average(magnitude)

        # Band energies
        band_energies = self._band_energies(radial_profile)

        # Spectral centroid
        centroid = self._spectral_centroid(radial_profile)

        # Periodic peaks detection
        has_peaks = self._detect_periodic_peaks(radial_profile)

        return SpectralFingerprint(
            band_energies=band_energies,
            centroid=centroid,
            has_periodic_peaks=has_peaks,
            azimuthal_profile=radial_profile,
        )

    @staticmethod
    def _azimuthal_average(magnitude: np.ndarray) -> np.ndarray:
        """Compute azimuthal (radial) average of 2D FFT magnitude.

        Args:
            magnitude: 2D FFT magnitude (H,W), fftshifted

        Returns:
            1D radial profile (distance from center → average magnitude)
        """
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        max_r = min(cy, cx)

        # Radial bins
        radial_profile = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            if mask.sum() > 0:
                radial_profile[i] = magnitude[mask].mean()

        return radial_profile

    @staticmethod
    def _band_energies(radial_profile: np.ndarray) -> dict[str, float]:
        """Compute energy in low/mid/high frequency bands.

        Args:
            radial_profile: 1D radial profile

        Returns:
            Dict with normalized band energies (low, mid, high)
        """
        n = len(radial_profile)
        if n < 3:
            return {"low": 1.0, "mid": 0.0, "high": 0.0}

        # Band ranges (0-33%, 33-66%, 66-100%)
        low_end = n // 3
        mid_end = 2 * n // 3

        low_energy = radial_profile[:low_end].sum()
        mid_energy = radial_profile[low_end:mid_end].sum()
        high_energy = radial_profile[mid_end:].sum()

        total = low_energy + mid_energy + high_energy + 1e-8

        return {
            "low": float(low_energy / total),
            "mid": float(mid_energy / total),
            "high": float(high_energy / total),
        }

    @staticmethod
    def _spectral_centroid(radial_profile: np.ndarray) -> float:
        """Compute spectral centroid (center of mass of frequency distribution).

        Args:
            radial_profile: 1D radial profile

        Returns:
            Normalized centroid position (0=DC, 1=Nyquist)
        """
        if len(radial_profile) == 0:
            return 0.5

        freqs = np.arange(len(radial_profile))
        centroid = (freqs * radial_profile).sum() / (radial_profile.sum() + 1e-8)
        return float(centroid / len(radial_profile))

    @staticmethod
    def _detect_periodic_peaks(radial_profile: np.ndarray, threshold: float = 0.3) -> bool:
        """Detect periodic peaks in radial profile (typical of GANs).

        Args:
            radial_profile: 1D radial profile
            threshold: Peak detection threshold

        Returns:
            True if periodic peaks detected
        """
        if len(radial_profile) < 10:
            return False

        # Autocorrelation to detect periodicity
        profile_norm = radial_profile - radial_profile.mean()
        autocorr = np.correlate(profile_norm, profile_norm, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        if autocorr[0] == 0:
            return False

        autocorr = autocorr / autocorr[0]

        # Count peaks above threshold (excluding lag=0)
        peaks = np.where(autocorr[2:] > threshold)[0]
        return len(peaks) >= 2  # At least 2 peaks → periodic

    def _match_profiles(self, fingerprints: list[SpectralFingerprint]) -> list[dict[str, Any]]:
        """Match extracted fingerprints against known generator profiles.

        Args:
            fingerprints: List of SpectralFingerprint objects

        Returns:
            List of matches sorted by score (descending)
        """
        if not fingerprints:
            return []

        # Average fingerprint across frames
        avg_bands = {
            "low": float(np.mean([fp.band_energies["low"] for fp in fingerprints])),
            "mid": float(np.mean([fp.band_energies["mid"] for fp in fingerprints])),
            "high": float(np.mean([fp.band_energies["high"] for fp in fingerprints])),
        }
        avg_centroid = float(np.mean([fp.centroid for fp in fingerprints]))
        has_peaks_majority = sum(fp.has_periodic_peaks for fp in fingerprints) >= len(fingerprints) / 2

        matches = []
        for gen_name, profile in self.profile_db.items():
            score = self._profile_similarity(avg_bands, avg_centroid, has_peaks_majority, profile)
            matches.append({"name": gen_name, "score": round(score, 4)})

        # Sort by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    @staticmethod
    def _profile_similarity(
        bands: dict[str, float],
        centroid: float,
        has_peaks: bool,
        profile: dict[str, Any],
    ) -> float:
        """Compute similarity between extracted fingerprint and a generator profile.

        Args:
            bands: Band energies {low, mid, high}
            centroid: Spectral centroid
            has_peaks: Periodic peaks detected
            profile: Generator profile dict

        Returns:
            Similarity score (0.0 - 1.0)
        """
        score = 0.0

        # Band energy match (60% weight)
        for band_name, energy in bands.items():
            expected_range = profile.get(f"band_{band_name}", (0, 1))
            if expected_range[0] <= energy <= expected_range[1]:
                score += 0.2  # Each band: 20%

        # Centroid match (20% weight)
        centroid_range = profile.get("centroid_range", (0, 1))
        if centroid_range[0] <= centroid <= centroid_range[1]:
            score += 0.2

        # Periodic peaks match (20% weight)
        expected_peaks = profile.get("periodic_peaks", False)
        if has_peaks == expected_peaks:
            score += 0.2

        return min(1.0, score)
