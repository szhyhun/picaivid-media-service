"""OpenCLIP model for image embeddings and room classification."""
import logging
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

# Room type labels for classification
ROOM_TYPES = [
    "living room",
    "bedroom",
    "kitchen",
    "bathroom",
    "dining room",
    "office",
    "garage",
    "basement",
    "attic",
    "hallway",
    "entrance",
    "patio",
    "backyard",
    "front yard",
    "pool",
    "exterior front",
    "exterior back",
    "aerial view",
    "drone shot",
]


class OpenCLIPModel:
    """OpenCLIP model wrapper for embeddings and room classification."""

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None
        self._text_features = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        import open_clip

        logger.info(f"Loading OpenCLIP model: {settings.OPENCLIP_MODEL}")

        # Determine device
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon) for inference")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info("Using CUDA for inference")
        else:
            self._device = torch.device("cpu")
            logger.info("Using CPU for inference")

        # Load model
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            settings.OPENCLIP_MODEL,
            pretrained=settings.OPENCLIP_PRETRAINED,
            cache_dir=settings.MODEL_CACHE_DIR,
        )
        self._model = self._model.to(self._device)
        self._model.eval()

        self._tokenizer = open_clip.get_tokenizer(settings.OPENCLIP_MODEL)

        # Pre-compute text features for room types
        self._precompute_text_features()

        logger.info("OpenCLIP model loaded successfully")

    def _precompute_text_features(self) -> None:
        """Pre-compute text embeddings for room types."""
        prompts = [f"a photo of a {room}" for room in ROOM_TYPES]
        text_tokens = self._tokenizer(prompts).to(self._device)

        with torch.no_grad():
            self._text_features = self._model.encode_text(text_tokens)
            self._text_features = self._text_features / self._text_features.norm(dim=-1, keepdim=True)

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Get embedding vector for an image.

        Args:
            image: PIL Image

        Returns:
            Numpy array of shape (512,) for ViT-B-32
        """
        self._ensure_loaded()

        # Preprocess image
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)

        # Get embedding
        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten()

    def classify_room(self, image: Image.Image) -> tuple[str, float]:
        """Classify room type from image.

        Args:
            image: PIL Image

        Returns:
            Tuple of (room_type, confidence)
        """
        self._ensure_loaded()

        # Get image embedding
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity with room types
            similarity = (image_features @ self._text_features.T).squeeze(0)
            probs = similarity.softmax(dim=-1)

            # Get top prediction
            top_idx = probs.argmax().item()
            confidence = probs[top_idx].item()

        return ROOM_TYPES[top_idx], confidence

    def get_batch_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """Get embeddings for multiple images.

        Args:
            images: List of PIL Images

        Returns:
            Numpy array of shape (n_images, embedding_dim)
        """
        self._ensure_loaded()

        # Process all images
        image_tensors = torch.stack([self._preprocess(img) for img in images]).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()


# Singleton instance
openclip_model = OpenCLIPModel()
