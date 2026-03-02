"""MiDaS/DPT model for depth estimation using Hugging Face Transformers.

Uses Intel/dpt-large from Hugging Face Hub - more reliable than torch.hub.
Model is downloaded once and cached in ./ml_models/huggingface/

For production:
- Models are pre-baked into Docker image during build
- Set HF_HUB_OFFLINE=1 to use only cached models
- See scripts/download_models.py for pre-download during build
"""
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging as hf_logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Set HuggingFace cache directory
HF_CACHE_DIR = Path(settings.MODEL_CACHE_DIR) / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)

# Model identifier on Hugging Face Hub
MODEL_ID = "Intel/dpt-large"

# Check if running in offline mode (production with pre-baked models)
OFFLINE_MODE = os.environ.get("HF_HUB_OFFLINE", "0") == "1"


class MiDaSModel:
    """DPT depth estimation model wrapper (MiDaS architecture via HuggingFace).

    Uses Intel/dpt-large from Hugging Face Hub.
    Model is loaded once on first use and cached for subsequent calls.
    """

    _instance = None
    _model = None
    _processor = None
    _device = None
    _loaded = False

    def __new__(cls):
        """Singleton pattern - ensure only one model instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if MiDaSModel._loaded:
            return

        logger.info(f"Loading depth model: {MODEL_ID}")
        if OFFLINE_MODE:
            logger.info("Running in offline mode (using pre-cached models)")
        load_started_at = time.perf_counter()

        # Determine device
        if torch.backends.mps.is_available():
            MiDaSModel._device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon) for depth estimation")
        elif torch.cuda.is_available():
            MiDaSModel._device = torch.device("cuda")
            logger.info("Using CUDA for depth estimation")
        else:
            MiDaSModel._device = torch.device("cpu")
            logger.info("Using CPU for depth estimation")

        try:
            # Suppress verbose Hugging Face loading reports (missing/newly initialized keys)
            # that are expected for this checkpoint/runtime combo and create noisy logs.
            previous_hf_verbosity = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            # In production (OFFLINE_MODE), use only local cached models
            # In development, allow downloading if not cached
            MiDaSModel._processor = DPTImageProcessor.from_pretrained(
                MODEL_ID,
                cache_dir=str(HF_CACHE_DIR),
                local_files_only=OFFLINE_MODE,
            )
            MiDaSModel._model = DPTForDepthEstimation.from_pretrained(
                MODEL_ID,
                cache_dir=str(HF_CACHE_DIR),
                local_files_only=OFFLINE_MODE,
            )
            hf_logging.set_verbosity(previous_hf_verbosity)
            MiDaSModel._model = MiDaSModel._model.to(MiDaSModel._device)
            MiDaSModel._model.eval()
            MiDaSModel._loaded = True

            logger.info(
                "Depth model loaded successfully from %s on %s in %.2fs",
                MODEL_ID,
                MiDaSModel._device,
                (time.perf_counter() - load_started_at),
            )
        except Exception as e:
            # Restore verbosity before retry/raise path.
            try:
                hf_logging.set_verbosity(previous_hf_verbosity)
            except Exception:
                pass
            logger.error(f"Failed to load depth model: {e}")
            if not OFFLINE_MODE:
                # Try loading from local cache only as fallback
                try:
                    logger.info("Attempting to load from local cache...")
                    previous_hf_verbosity = hf_logging.get_verbosity()
                    hf_logging.set_verbosity_error()
                    MiDaSModel._processor = DPTImageProcessor.from_pretrained(
                        MODEL_ID,
                        cache_dir=str(HF_CACHE_DIR),
                        local_files_only=True,
                    )
                    MiDaSModel._model = DPTForDepthEstimation.from_pretrained(
                        MODEL_ID,
                        cache_dir=str(HF_CACHE_DIR),
                        local_files_only=True,
                    )
                    hf_logging.set_verbosity(previous_hf_verbosity)
                    MiDaSModel._model = MiDaSModel._model.to(MiDaSModel._device)
                    MiDaSModel._model.eval()
                    MiDaSModel._loaded = True
                    logger.info(
                        "Depth model loaded from local cache on %s in %.2fs",
                        MiDaSModel._device,
                        (time.perf_counter() - load_started_at),
                    )
                except Exception as cache_error:
                    try:
                        hf_logging.set_verbosity(previous_hf_verbosity)
                    except Exception:
                        pass
                    logger.error(f"Failed to load from cache: {cache_error}")
                    raise
            else:
                logger.error("Models not found in cache. Run 'python scripts/download_models.py' first.")
                raise

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Estimate depth map from image.

        Args:
            image: PIL Image

        Returns:
            Numpy array of depth values (higher = farther)
        """
        self._ensure_loaded()

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process image
        inputs = MiDaSModel._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(MiDaSModel._device) for k, v in inputs.items()}

        # Estimate depth
        with torch.no_grad():
            outputs = MiDaSModel._model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],  # PIL size is (width, height), need (height, width)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def analyze_depth(self, image: Image.Image) -> dict:
        """Analyze depth characteristics of an image.

        Args:
            image: PIL Image

        Returns:
            Dict with depth metrics
        """
        depth_map = self.estimate_depth(image)

        # Normalize depth map
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        # Compute metrics
        variance = float(np.var(depth_normalized))
        std = float(np.std(depth_normalized))

        # Estimate number of depth layers using histogram
        hist, _ = np.histogram(depth_normalized, bins=10)
        significant_bins = np.sum(hist > len(depth_normalized.flatten()) * 0.05)

        # Compute depth range
        depth_range = float(depth_map.max() - depth_map.min())

        return {
            "variance": variance,
            "std": std,
            "depth_layers": int(significant_bins),
            "depth_range": depth_range,
            "min_depth": float(depth_map.min()),
            "max_depth": float(depth_map.max()),
        }

    def get_confidence_tier(self, depth_metrics: dict) -> str:
        """Determine confidence tier based on depth metrics.

        Args:
            depth_metrics: Dict from analyze_depth()

        Returns:
            One of: "low", "medium", "high"
        """
        variance = depth_metrics["variance"]
        depth_layers = depth_metrics["depth_layers"]

        # High confidence: significant depth variation with multiple layers
        if variance > 0.1 and depth_layers >= 4:
            return "high"

        # Medium confidence: some depth variation
        if variance > 0.05 and depth_layers >= 2:
            return "medium"

        # Low confidence: flat scene
        return "low"


# Singleton instance
midas_model = MiDaSModel()
