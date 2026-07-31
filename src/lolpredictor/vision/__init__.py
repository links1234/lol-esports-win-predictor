"""Versioned screenshot extraction for supported broadcast overlays."""

from lolpredictor.vision.parser import (
    DraftConfirmationRequiredError,
    build_confirmed_draft,
    parse_screenshot,
    predict_screenshot,
)
from lolpredictor.vision.schemas import (
    OverlayProfile,
    ScreenshotCorrections,
    ScreenshotDraftCandidate,
    ScreenshotPredictionContext,
    TemplateCatalog,
)

__all__ = [
    "DraftConfirmationRequiredError",
    "OverlayProfile",
    "ScreenshotCorrections",
    "ScreenshotDraftCandidate",
    "ScreenshotPredictionContext",
    "TemplateCatalog",
    "build_confirmed_draft",
    "parse_screenshot",
    "predict_screenshot",
]
