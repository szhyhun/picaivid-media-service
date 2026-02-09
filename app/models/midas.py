"""MiDaS model for depth estimation."""
import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


class MiDaSModel:
    """MiDaS depth estimation model wrapper."""

    def __init__(self):
        self._model = None
        self._transform = None
        self._device = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        logger.info("Loading MiDaS model: DPT_Large")

        # Determine device
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon) for depth estimation")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info("Using CUDA for depth estimation")
        else:
            self._device = torch.device("cpu")
            logger.info("Using CPU for depth estimation")

        # Load MiDaS from torch hub
        self._model = torch.hub.load(
            "intel-isl/MiDaS",
            "DPT_Large",
            trust_repo=True,
        )
        self._model = self._model.to(self._device)
        self._model.eval()

        # Load transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._transform = midas_transforms.dpt_transform

        logger.info("MiDaS model loaded successfully")

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

        # Convert to numpy
        img_np = np.array(image)

        # Transform for model
        input_batch = self._transform(img_np).to(self._device)

        # Estimate depth
        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_np.shape[:2],
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
