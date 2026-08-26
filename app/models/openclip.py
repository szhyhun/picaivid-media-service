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
            room_features = []
            for room in ROOM_TYPES:
                prompts = ROOM_PROMPTS.get(room, [f"a photo of a {room}"])
                text_tokens = self._tokenizer(prompts).to(self._device)
                text_features = self._model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Use centroid of prompts for a more robust room concept vector.
                room_feature = text_features.mean(dim=0, keepdim=True)
                room_feature = room_feature / room_feature.norm(dim=-1, keepdim=True)
                room_features.append(room_feature)

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

            # Compute similarity with room types
            similarity = (image_features @ self._text_features.T).squeeze(0)
            logit_scale = self._model.logit_scale.exp() if hasattr(self._model, "logit_scale") else 1.0
            similarity = similarity * logit_scale
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

    def score_labels(
        self,
        image: Image.Image,
        labels: Sequence[str],
        prompt_templates: Sequence[str] | None = None,
    ) -> Dict[str, float]:
        """Score an image against arbitrary text labels with CLIP cosine similarity."""
        self._ensure_loaded()
        unique_labels = tuple(str(label) for label in labels if str(label).strip())
        if not unique_labels:
            return {}

        cache_key = tuple(unique_labels) + tuple(prompt_templates or ())
        text_features = self._prompt_feature_cache.get(cache_key)
        if text_features is None:
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
                    text_features = torch.cat(grouped, dim=0)
                else:
                    text_features = encoded
            self._prompt_feature_cache[cache_key] = text_features

        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(0)

        return {
            label: float(similarities[idx].item())
            for idx, label in enumerate(unique_labels)
        }

    def release(self) -> None:
        """Release accelerator-resident weights after the CLIP pipeline stage."""
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
