"""Process-start model warmup helpers."""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def warmup_core_models(
    context: str,
    *,
    include_mast3r: bool = False,
    include_legacy: bool = False,
) -> None:
    """Load process-start models once per process.

    Warmup failures are logged but do not stop process startup.
    """
    started_at = time.perf_counter()
    logger.info("Model warmup start: context=%s", context)

    if include_legacy:
        try:
            from app.models.openclip import openclip_model

            openclip_model._ensure_loaded()  # internal singleton warmup
            logger.info("Model warmup loaded: openclip context=%s", context)
        except Exception:
            logger.exception("Model warmup failed: openclip context=%s", context)

        try:
            from app.models.midas import midas_model

            midas_model._ensure_loaded()  # internal singleton warmup
            logger.info("Model warmup loaded: midas context=%s", context)
        except Exception:
            logger.exception("Model warmup failed: midas context=%s", context)

    if include_mast3r:
        try:
            from app.pipeline.phase1_analyze.mast3r_pipeline import warmup_mast3r

            warmup_mast3r()
            logger.info("Model warmup loaded: mast3r context=%s", context)
        except Exception:
            logger.exception("Model warmup failed: mast3r context=%s", context)

    logger.info(
        "Model warmup complete: context=%s include_mast3r=%s include_legacy=%s elapsed_ms=%.1f",
        context,
        include_mast3r,
        include_legacy,
        (time.perf_counter() - started_at) * 1000.0,
    )
