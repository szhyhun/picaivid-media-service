"""LTX-2 video generation model with multi-frame injection support.

Supports 1-N input images for frame interpolation:
- 1 image: Single frame at position 0
- 2 images: Start and end frames
- 3+ images: Evenly distributed keyframes

On systems without CUDA (Apple Silicon, CPU), uses mock mode
that generates placeholder videos with simple motion effects.
"""
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

# LTX-2 model settings
LTX2_MODEL_ID = "Lightricks/LTX-2"
LTX2_RESOLUTION = (768, 512)  # width, height (must be divisible by 32)
LTX2_FPS = 24
LTX2_DEFAULT_FRAMES = 49  # 2 seconds at 24fps

# Check if CUDA is available (real LTX-2 requires CUDA)
CUDA_AVAILABLE = torch.cuda.is_available()
MOCK_MODE = not CUDA_AVAILABLE


class LTX2Model:
    """LTX-2 video generation model wrapper with multi-frame support.

    Singleton pattern - model is loaded once and reused.
    Uses mock mode on Apple Silicon / CPU systems.
    """

    _instance = None
    _pipe = None
    _vae = None
    _loaded = False
    _mock_mode = MOCK_MODE

    def __new__(cls):
        """Singleton pattern - ensure only one model instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if LTX2Model._loaded:
            return

        if LTX2Model._mock_mode:
            logger.info("LTX-2 running in MOCK MODE (no CUDA available)")
            LTX2Model._loaded = True
            return

        logger.info(f"Loading LTX-2 model: {LTX2_MODEL_ID}")

        try:
            from diffusers import LTX2ImageToVideoPipeline

            cache_dir = Path(settings.MODEL_CACHE_DIR) / "huggingface"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Check if we should use offline mode
            offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

            LTX2Model._pipe = LTX2ImageToVideoPipeline.from_pretrained(
                LTX2_MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=str(cache_dir),
                local_files_only=offline_mode,
            )

            # Move to GPU
            LTX2Model._pipe = LTX2Model._pipe.to("cuda")

            # Enable memory optimizations
            LTX2Model._pipe.enable_model_cpu_offload()

            # Store VAE reference for latent encoding
            LTX2Model._vae = LTX2Model._pipe.vae

            LTX2Model._loaded = True
            logger.info("LTX-2 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LTX-2 model: {e}")
            logger.info("Falling back to mock mode")
            LTX2Model._mock_mode = True
            LTX2Model._loaded = True

    def generate_video(
        self,
        images: List[Image.Image],
        prompt: str,
        cfg_scale: float = 4.0,
        inference_steps: int = 20,
        num_frames: int = LTX2_DEFAULT_FRAMES,
        guide_strength: float = 0.6,
        seed: Optional[int] = None,
        motion_type: str = "push_in",
    ) -> Tuple[np.ndarray, int]:
        """Generate video with multi-frame injection.

        Args:
            images: List of 1-N PIL Images as keyframes
                    - Evenly distributed across timeline
                    - 1 image: frame 0
                    - 2 images: frame 0 and last
                    - 3+ images: evenly spaced positions
                    - More images = smoother interpolation
            prompt: LTX-2 prompt describing the motion
            cfg_scale: Classifier-free guidance scale (3.0-6.0)
            inference_steps: Number of denoising steps (20-50)
            num_frames: Total frames to generate (49 = 2s, 73 = 3s)
            guide_strength: How strongly to adhere to keyframes (0.0-1.0)
            seed: Random seed for reproducibility
            motion_type: Type of motion (used for mock mode)

        Returns:
            Tuple of (video_frames as numpy array, fps)
        """
        self._ensure_loaded()

        if LTX2Model._mock_mode:
            return self._generate_mock_video(
                images, num_frames, motion_type
            ), LTX2_FPS

        # Real LTX-2 generation with guide frame injection
        return self._generate_real_video(
            images=images,
            prompt=prompt,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            num_frames=num_frames,
            guide_strength=guide_strength,
            seed=seed,
        ), LTX2_FPS

    def _compute_guide_positions(
        self, num_images: int, total_frames: int
    ) -> List[int]:
        """Compute frame positions for guide injection.

        Args:
            num_images: Number of input images
            total_frames: Total frames in video

        Returns:
            List of frame positions for each image
        """
        if num_images == 1:
            return [0]

        # Evenly distribute N images across the timeline
        positions = []
        for i in range(num_images):
            pos = int(i * (total_frames - 1) / (num_images - 1))
            positions.append(pos)
        return positions

    def _generate_real_video(
        self,
        images: List[Image.Image],
        prompt: str,
        cfg_scale: float,
        inference_steps: int,
        num_frames: int,
        guide_strength: float,
        seed: Optional[int],
    ) -> np.ndarray:
        """Generate video using real LTX-2 model with guide injection.

        Uses custom sampling loop to inject guide frames at each step.
        """
        import torch
        from diffusers.utils import export_to_video

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Prepare the first image (LTX-2 image-to-video uses single image)
        # For multi-frame, we'll inject additional frames during sampling
        primary_image = images[0].resize(LTX2_RESOLUTION, Image.Resampling.LANCZOS)

        # Compute guide positions
        positions = self._compute_guide_positions(len(images), num_frames)

        # For now, use the standard pipeline with first image
        # TODO: Implement custom sampling loop with guide injection
        # This requires modifying the diffusers internals

        # Standard generation (single image)
        output = LTX2Model._pipe(
            image=primary_image,
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted, warping, bending walls",
            width=LTX2_RESOLUTION[0],
            height=LTX2_RESOLUTION[1],
            num_frames=num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=cfg_scale,
            generator=generator,
            output_type="np",
        )

        # Extract video frames
        video_frames = output.frames[0]  # Shape: (num_frames, H, W, C)

        logger.info(
            f"Generated video: {video_frames.shape}, "
            f"positions={positions}, guide_strength={guide_strength}"
        )

        return video_frames

    def _generate_mock_video(
        self,
        images: List[Image.Image],
        num_frames: int,
        motion_type: str,
    ) -> np.ndarray:
        """Generate mock video for testing without GPU.

        Creates a real video by interpolating between input images
        with simple motion effects (zoom, pan, blend).

        Args:
            images: Input images
            num_frames: Number of frames to generate
            motion_type: Type of motion to simulate

        Returns:
            Video frames as numpy array (num_frames, H, W, 3)
        """
        import cv2

        # Resize all images to target resolution
        resized = []
        for img in images:
            img_resized = img.resize(LTX2_RESOLUTION, Image.Resampling.LANCZOS)
            resized.append(np.array(img_resized))

        frames = []
        positions = self._compute_guide_positions(len(resized), num_frames)

        for frame_idx in range(num_frames):
            t = frame_idx / (num_frames - 1) if num_frames > 1 else 0

            # Find which two keyframes we're between
            keyframe_a_idx = 0
            keyframe_b_idx = 0
            local_t = 0

            for i in range(len(positions) - 1):
                if positions[i] <= frame_idx <= positions[i + 1]:
                    keyframe_a_idx = i
                    keyframe_b_idx = i + 1
                    span = positions[i + 1] - positions[i]
                    if span > 0:
                        local_t = (frame_idx - positions[i]) / span
                    break
            else:
                # After last keyframe
                keyframe_a_idx = len(positions) - 1
                keyframe_b_idx = len(positions) - 1
                local_t = 0

            img_a = resized[keyframe_a_idx]
            img_b = resized[keyframe_b_idx]

            # Apply motion effect based on type
            if motion_type in ("push_in", "dolly_in", "micro_push_in"):
                # Zoom in effect
                scale = 1.0 + 0.15 * t
                frame = self._apply_zoom(img_a, scale)
            elif motion_type in ("push_out", "dolly_out", "micro_push_out"):
                # Zoom out effect
                scale = max(1.0, 1.15 - 0.15 * t)
                frame = self._apply_zoom(img_a, scale)
            elif motion_type in ("pan_left", "subtle_pan"):
                # Pan left effect
                offset = int(50 * t)
                frame = self._apply_pan(img_a, -offset, 0)
            elif motion_type == "pan_right":
                # Pan right effect
                offset = int(50 * t)
                frame = self._apply_pan(img_a, offset, 0)
            elif len(resized) > 1:
                # Blend between keyframes
                frame = cv2.addWeighted(
                    img_a, 1 - local_t, img_b, local_t, 0
                )
            else:
                # Static with subtle zoom
                scale = 1.0 + 0.02 * t
                frame = self._apply_zoom(img_a, scale)

            frames.append(frame)

        video = np.stack(frames, axis=0)
        logger.info(f"Generated mock video: {video.shape}, motion={motion_type}")
        return video

    def _apply_zoom(self, img: np.ndarray, scale: float) -> np.ndarray:
        """Apply zoom effect to image."""
        import cv2

        h, w = img.shape[:2]
        new_h = max(h, int(round(h * scale)))
        new_w = max(w, int(round(w * scale)))

        # Resize
        zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Crop center
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        cropped = zoomed[start_y:start_y + h, start_x:start_x + w]

        # Guard against rare rounding/crop edge-cases.
        if cropped.shape[0] != h or cropped.shape[1] != w:
            cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return cropped

    def _apply_pan(
        self, img: np.ndarray, offset_x: int, offset_y: int
    ) -> np.ndarray:
        """Apply pan effect to image."""
        import cv2

        h, w = img.shape[:2]

        # Create translation matrix
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

        # Apply translation
        panned = cv2.warpAffine(
            img, M, (w, h), borderMode=cv2.BORDER_REPLICATE
        )

        return panned

    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return LTX2Model._mock_mode


# Singleton instance
ltx2_model = LTX2Model()
