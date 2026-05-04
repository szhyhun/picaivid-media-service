"""Process-start model warmup helpers."""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def warmup_core_models(
    context: str,
    *,
    include_vggt: bool = False,
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

    if include_vggt:
        try:
            from app.models.vggt import vggt_model

            vggt_model._ensure_loaded()
            logger.info("Model warmup loaded: vggt context=%s", context)
        except Exception:
            logger.exception("Model warmup failed: vggt context=%s", context)

    logger.info(
        "Model warmup complete: context=%s include_vggt=%s include_legacy=%s elapsed_ms=%.1f",
        context,
        include_vggt,
        include_legacy,
        (time.perf_counter() - started_at) * 1000.0,
    )
