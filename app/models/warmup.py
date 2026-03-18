"""Process-start model warmup helpers."""
from __future__ import annotations

import logging
import time

from app.models.midas import midas_model
from app.models.openclip import openclip_model

logger = logging.getLogger(__name__)


def warmup_core_models(context: str, *, include_loftr: bool = False) -> None:
    """Load core semantic/depth models once per process.

    Warmup failures are logged but do not stop process startup.
    """
    started_at = time.perf_counter()
    logger.info("Model warmup start: context=%s", context)

    try:
        openclip_model._ensure_loaded()  # internal singleton warmup
        logger.info("Model warmup loaded: openclip context=%s", context)
    except Exception:
        logger.exception("Model warmup failed: openclip context=%s", context)

    try:
        from app.pipeline.phase1_analyze.learned_matching import _load_dinov2

        _load_dinov2()
        logger.info("Model warmup loaded: dinov3 context=%s", context)
    except Exception:
        logger.exception("Model warmup failed: dinov3 context=%s", context)

    try:
        midas_model._ensure_loaded()  # internal singleton warmup
        logger.info("Model warmup loaded: midas context=%s", context)
    except Exception:
        logger.exception("Model warmup failed: midas context=%s", context)

    if include_loftr:
        try:
            from app.pipeline.phase1_analyze.matcher_loaders import load_loftr_checkpoint

            load_loftr_checkpoint("indoor")
            logger.info("Model warmup loaded: loftr_indoor context=%s", context)
        except Exception:
            logger.exception("Model warmup failed: loftr_indoor context=%s", context)

    logger.info(
        "Model warmup complete: context=%s include_loftr=%s elapsed_ms=%.1f",
        context,
        include_loftr,
        (time.perf_counter() - started_at) * 1000.0,
    )
