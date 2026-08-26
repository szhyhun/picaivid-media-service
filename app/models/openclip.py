"""OpenCLIP model for image embeddings and lightweight region classification."""
import gc
import logging
from typing import Dict, List, Optional, Sequence

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
    "laundry room",
    "bathroom",
    "storage",
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

ROOM_PROMPTS = {
    # Every room type gets several descriptive prompts. Earlier only garage,
    # bathroom, storage and laundry were tuned while the rest fell back to a bare
    # "a photo of a {room}", so the tuned classes acted as attractors: measured
    # against human ground truth, 21 of 174 photos were wrongly called garage,
    # including living rooms, kitchens, front yards and patios.
    "living room": [
        "a photo of a living room",
        "a real estate photo of a living room with a sofa, coffee table and television",
        "an interior listing photo of a lounge or family room with seating",
    ],
    "bedroom": [
        "a photo of a bed in a bedroom",
        "a real estate photo of a bedroom with a made bed, pillows and a nightstand",
        "an interior photo of a bedroom showing a bed with a headboard",
    ],
    "kitchen": [
        "a photo of a kitchen",
        "a real estate photo of a kitchen with cabinets, countertops, sink and appliances",
        "an interior listing photo of a kitchen with a refrigerator and a stove",
    ],
    "laundry room": [
        "a photo of a laundry room",
        "a photo of a utility room with washer and dryer",
        "a photo of a washer and dryer in a laundry area",
    ],
    "bathroom": [
        "a photo of a bathroom",
        "a real estate photo of a bathroom with sink vanity mirror shower or toilet",
        "an interior listing photo of a bathroom",
    ],
    "storage": [
        "a photo of shelving stacked with boxes and stored items",
        "a photo of a storage closet packed with belongings",
    ],
    "dining room": [
        "a photo of a dining table surrounded by dining chairs",
        "a real estate photo of a dining area with a table set for a meal",
    ],
    "office": [
        "a photo of a home office",
        "a real estate photo of a home office with a desk, office chair and monitor",
        "an interior listing photo of a study or work room with a desk and bookshelf",
    ],
    "garage": [
        "a photo of the inside of a garage",
        "a photo of a garage interior with a concrete floor and a garage door",
        "an interior photo of a garage with tools, shelving and parked vehicles",
    ],
    "basement": [
        "a photo of a basement",
        "a photo of an unfinished basement with concrete walls and exposed ceiling",
        "an interior photo of a lower level rec room in a basement",
    ],
    "attic": [
        "a photo of an attic",
        "a photo of an attic loft with sloped ceilings and roof beams",
    ],
    "hallway": [
        "a photo of a hallway",
        "a real estate photo of an interior corridor with doors along it",
        "an interior listing photo of a landing or hallway with stairs",
    ],
    "entrance": [
        "a photo of an entryway with a closed front door",
        "a real estate photo of a foyer showing the inside of the front door",
    ],
    "patio": [
        "a photo of a patio",
        "a photo of a covered outdoor patio with outdoor furniture",
        "a real estate photo of a deck or terrace attached to a house",
    ],
    "backyard": [
        "a photo of a fenced backyard behind a house",
        "a real estate photo of a private rear garden enclosed by a fence",
        "an outdoor photo of a back garden with a lawn, fence and no street",
    ],
    "front yard": [
        "a photo of the front yard of a house seen from the street",
        "a real estate photo of a driveway and front walkway leading to a front door",
    ],
    "pool": [
        "a photo of a swimming pool",
        "a real estate photo of a backyard swimming pool with decking",
    ],
    "exterior front": [
        "a photo of the front exterior of a house",
        "a real estate photo of the front facade and entrance of a home from the street",
        "an exterior photo showing the front elevation of a house",
    ],
    "exterior back": [
        "a photo of the rear exterior of a house",
        "a real estate photo of the back facade of a home seen from the garden",
    ],
    "aerial view": [
        "an aerial photo of a house and its lot",
        "a drone photo looking down on a residential property and neighbourhood",
        "a high angle aerial view of houses and streets",
    ],
    "drone shot": [
        "a drone photograph of a property from above",
        "an aerial drone view of a house, roof and surrounding land",
    ],
}



class OpenCLIPModel:
    """OpenCLIP model wrapper for embeddings and room classification."""

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None
        self._text_features = None
        self._prompt_feature_cache: Dict[tuple[str, ...], torch.Tensor] = {}

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
        with torch.no_grad():
            prompts: list[str] = []
            prompt_counts: list[int] = []
            for room in ROOM_TYPES:
                room_prompts = ROOM_PROMPTS.get(room, [f"a photo of a {room}"])
                prompts.extend(room_prompts)
                prompt_counts.append(len(room_prompts))
            text_tokens = self._tokenizer(prompts).to(self._device)
            encoded = self._model.encode_text(text_tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)

            room_features = []
            offset = 0
            for count in prompt_counts:
                # Use the same per-room prompt centroid as before, but encode all
                # prompts in one model call instead of one call per room type.
                room_feature = encoded[offset : offset + count].mean(dim=0, keepdim=True)
                room_feature = room_feature / room_feature.norm(dim=-1, keepdim=True)
                room_features.append(room_feature)
                offset += count

            self._text_features = torch.cat(room_features, dim=0)

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
        return self._classify_features(image_features)[0]

    def classify_embeddings(self, embeddings: np.ndarray) -> list[tuple[str, float]]:
        """Classify normalized stored embeddings without re-encoding images."""
        self._ensure_loaded()
        features = torch.as_tensor(embeddings, dtype=torch.float32, device=self._device)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        features = features / features.norm(dim=-1, keepdim=True)
        return self._classify_features(features)

    def _classify_features(self, features: torch.Tensor) -> list[tuple[str, float]]:
        with torch.no_grad():
            similarity = features @ self._text_features.T
            logit_scale = self._model.logit_scale.exp() if hasattr(self._model, "logit_scale") else 1.0
            probabilities = (similarity * logit_scale).softmax(dim=-1)
            top_indices = probabilities.argmax(dim=-1)
        return [
            (ROOM_TYPES[int(index)], float(probabilities[row, index].item()))
            for row, index in enumerate(top_indices.tolist())
        ]

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

    def score_labels(
        self,
        image: Image.Image,
        labels: Sequence[str],
        prompt_templates: Sequence[str] | None = None,
    ) -> Dict[str, float]:
        """Score an image against arbitrary text labels with CLIP cosine similarity."""
        self._ensure_loaded()
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        unique_labels = tuple(str(label) for label in labels if str(label).strip())
        return self._score_features(image_features, unique_labels, prompt_templates)

    def score_embedding(
        self,
        embedding: Sequence[float] | np.ndarray,
        labels: Sequence[str],
        prompt_templates: Sequence[str] | None = None,
    ) -> Dict[str, float]:
        """Score arbitrary labels from a stored normalized image embedding."""
        self._ensure_loaded()
        unique_labels = tuple(str(label) for label in labels if str(label).strip())
        if not unique_labels:
            return {}
        image_features = torch.as_tensor(embedding, dtype=torch.float32, device=self._device).reshape(1, -1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return self._score_features(image_features, unique_labels, prompt_templates)

    def _score_features(
        self,
        image_features: torch.Tensor,
        unique_labels: tuple[str, ...],
        prompt_templates: Sequence[str] | None,
    ) -> Dict[str, float]:
        if not unique_labels:
            return {}
        text_features = self._features_for_labels(unique_labels, prompt_templates)
        with torch.no_grad():
            similarities = (image_features @ text_features.T).squeeze(0)

        return {
            label: float(similarities[idx].item())
            for idx, label in enumerate(unique_labels)
        }

    def _features_for_labels(
        self,
        unique_labels: tuple[str, ...],
        prompt_templates: Sequence[str] | None,
    ) -> torch.Tensor:
        cache_key = unique_labels + tuple(prompt_templates or ())
        cached = self._prompt_feature_cache.get(cache_key)
        if cached is not None:
            return cached
        prompts: list[str] = []
        for label in unique_labels:
            if prompt_templates:
                prompts.extend(template.format(label=label) for template in prompt_templates)
            else:
                prompts.append(f"a photo of a {label}")
        with torch.no_grad():
            text_tokens = self._tokenizer(prompts).to(self._device)
            encoded = self._model.encode_text(text_tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            if prompt_templates:
                grouped = []
                chunk = len(prompt_templates)
                for idx in range(0, encoded.shape[0], chunk):
                    feature = encoded[idx : idx + chunk].mean(dim=0, keepdim=True)
                    feature = feature / feature.norm(dim=-1, keepdim=True)
                    grouped.append(feature)
                encoded = torch.cat(grouped, dim=0)
        self._prompt_feature_cache[cache_key] = encoded
        return encoded

    def release(self) -> None:
        """Release accelerator-resident weights after the CLIP pipeline stage."""
        if self._model is None:
            return
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None
        self._text_features = None
        self._prompt_feature_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()


# Singleton instance
openclip_model = OpenCLIPModel()
