import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Tuple

import torch

from app.core.config import settings

logger = logging.getLogger(__name__)

_ZJU_DEBUG_ENV_BY_VARIANT: Dict[str, str] = {
    "indoor_ds": "LOFTR_ZJU_INDOOR_DS_CKPT",
    "indoor_ot": "LOFTR_ZJU_INDOOR_OT_CKPT",
}
_MATCHFORMER_DEBUG_ENV_BY_VARIANT: Dict[str, str] = {
    "indoor": "MATCHFORMER_INDOOR_CKPT",
    "outdoor": "MATCHFORMER_OUTDOOR_CKPT",
}


def _preferred_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and bool(mps_backend.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def _same_torch_device(a: torch.device, b: torch.device) -> bool:
    if a.type != b.type:
        return False
    if a.type in {"cpu", "mps"}:
        return True
    if a.type == "cuda":
        if a.index is None or b.index is None:
            return True
        return a.index == b.index
    return a == b


def _env_or_settings(name: str, default: str = "") -> str:
    raw_env = str(os.getenv(name, "")).strip()
    if raw_env:
        return raw_env
    raw_settings = getattr(settings, name, None)
    if raw_settings is None:
        return str(default)
    return str(raw_settings).strip() or str(default)


def _cfg_to_lower_dict(config: Any) -> Any:
    if isinstance(config, dict):
        return {str(k).lower(): _cfg_to_lower_dict(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        converted = [_cfg_to_lower_dict(v) for v in config]
        return type(config)(converted) if isinstance(config, tuple) else converted
    if hasattr(config, "items"):
        return {str(k).lower(): _cfg_to_lower_dict(v) for k, v in config.items()}
    return config


def _matchformer_config_for_variant(cfg: Any, variant: str) -> Any:
    if variant == "indoor":
        cfg.MATCHFORMER.BACKBONE_TYPE = "largesea"
        cfg.MATCHFORMER.SCENS = "indoor"
    elif variant == "outdoor":
        cfg.MATCHFORMER.BACKBONE_TYPE = "largela"
        cfg.MATCHFORMER.SCENS = "outdoor"
    else:
        raise RuntimeError(f"Unsupported MatchFormer debug variant '{variant}'")

    cfg.MATCHFORMER.RESOLUTION = (8, 2)
    cfg.MATCHFORMER.COARSE.D_MODEL = 256
    cfg.MATCHFORMER.COARSE.D_FFN = 256
    return cfg


class _BaseMatcherLoader(ABC):
    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self._meta_cache: Dict[str, Dict[str, str]] = {}

    def is_cached(self, key: str) -> bool:
        return str(key) in self._cache

    def load(self, key: str) -> Tuple[Any, Dict[str, str]]:
        cache_key = str(key)
        preferred = self._resolve_device(cache_key)
        if cache_key in self._cache:
            matcher = self._cache[cache_key]
            matcher = self._move_cached_matcher(cache_key, matcher, preferred)
            return matcher, dict(self._meta_cache.get(cache_key, {}))

        matcher, meta = self._build(cache_key, preferred)
        matcher = matcher.to(preferred)
        matcher.eval()
        self._cache[cache_key] = matcher
        self._meta_cache[cache_key] = dict(meta)
        self._log_loaded(cache_key, preferred)
        return matcher, dict(self._meta_cache[cache_key])

    def _move_cached_matcher(self, key: str, matcher: Any, preferred: torch.device) -> Any:
        current = self._matcher_device(matcher)
        if not _same_torch_device(current, preferred):
            matcher = matcher.to(preferred)
            matcher.eval()
            self._cache[key] = matcher
            self._log_moved(key, current, preferred)
        return matcher

    def _matcher_device(self, matcher: Any) -> torch.device:
        try:
            return next(matcher.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _resolve_device(self, key: str) -> torch.device:
        return _preferred_torch_device()

    @abstractmethod
    def _build(self, key: str, preferred: torch.device) -> Tuple[Any, Dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def _log_loaded(self, key: str, preferred: torch.device) -> None:
        raise NotImplementedError

    @abstractmethod
    def _log_moved(self, key: str, current: torch.device, preferred: torch.device) -> None:
        raise NotImplementedError


class _KorniaLoFTRLoader(_BaseMatcherLoader):
    def _build(self, key: str, preferred: torch.device) -> Tuple[Any, Dict[str, str]]:
        from kornia.feature import LoFTR

        matcher = LoFTR(pretrained=key)
        return matcher, {"checkpoint": key}

    def _log_loaded(self, key: str, preferred: torch.device) -> None:
        logger.info("Loaded LoFTR matcher (%s) on %s", key, preferred)

    def _log_moved(self, key: str, current: torch.device, preferred: torch.device) -> None:
        logger.info("Moved cached LoFTR matcher (%s) from %s to %s", key, current, preferred)


class _ZJUDebugMatcherLoader(_BaseMatcherLoader):
    def _repo_dir(self) -> str:
        repo_dir = _env_or_settings("LOFTR_ZJU_REPO_DIR", "")
        if not repo_dir:
            raise RuntimeError("ZJU debug matcher is missing env LOFTR_ZJU_REPO_DIR")
        if not os.path.isdir(repo_dir):
            raise RuntimeError(f"ZJU debug matcher repo path not found: {repo_dir}")
        return repo_dir

    def _checkpoint_path(self, variant: str) -> Tuple[str, str]:
        env_key = _ZJU_DEBUG_ENV_BY_VARIANT.get(str(variant))
        if not env_key:
            raise RuntimeError(f"Unsupported ZJU debug variant '{variant}'")
        ckpt_path = _env_or_settings(env_key, "")
        if not ckpt_path:
            raise RuntimeError(f"ZJU debug matcher is missing env {env_key}")
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"ZJU debug checkpoint not found for {variant}: {ckpt_path}")
        return ckpt_path, env_key

    def _build(self, key: str, preferred: torch.device) -> Tuple[Any, Dict[str, str]]:
        repo_dir = self._repo_dir()
        ckpt_path, env_key = self._checkpoint_path(key)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        try:
            from src.loftr import LoFTR as ZJULoFTR, default_cfg as zju_default_cfg
        except Exception as exc:
            raise RuntimeError(
                f"ZJU debug matcher failed importing repo modules from {repo_dir}: {exc}"
            ) from exc

        cfg = deepcopy(zju_default_cfg)
        if isinstance(cfg.get("match_coarse"), dict):
            cfg["match_coarse"].setdefault("sparse_spvs", False)

        matcher = ZJULoFTR(config=cfg)
        try:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        except Exception as exc:
            raise RuntimeError(
                f"ZJU debug matcher failed loading checkpoint '{ckpt_path}': {exc}"
            ) from exc

        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"ZJU debug matcher checkpoint has invalid state_dict format: {ckpt_path}"
            )

        try:
            incompatible = matcher.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            raise RuntimeError(
                f"ZJU debug matcher failed loading state_dict for {key}: {exc}"
            ) from exc

        return matcher, {
            "variant": key,
            "repo_dir": repo_dir,
            "ckpt_path": ckpt_path,
            "env_key": env_key,
            "model_class": matcher.__class__.__name__,
            "missing_keys": str(len(getattr(incompatible, "missing_keys", []) or [])),
            "unexpected_keys": str(len(getattr(incompatible, "unexpected_keys", []) or [])),
        }

    def _log_loaded(self, key: str, preferred: torch.device) -> None:
        meta = self._meta_cache[key]
        logger.info(
            "Loaded ZJU LoFTR debug matcher (%s) from %s on %s [missing=%s unexpected=%s]",
            key,
            meta["ckpt_path"],
            preferred,
            meta["missing_keys"],
            meta["unexpected_keys"],
        )

    def _log_moved(self, key: str, current: torch.device, preferred: torch.device) -> None:
        logger.info(
            "Moved cached ZJU LoFTR debug matcher (%s) from %s to %s",
            key,
            current,
            preferred,
        )


class _MatchFormerDebugMatcherLoader(_BaseMatcherLoader):
    def _repo_dir(self) -> str:
        repo_dir = _env_or_settings("MATCHFORMER_REPO_DIR", "")
        if not repo_dir:
            raise RuntimeError("MatchFormer debug matcher is missing env MATCHFORMER_REPO_DIR")
        if not os.path.isdir(repo_dir):
            raise RuntimeError(f"MatchFormer debug matcher repo path not found: {repo_dir}")
        return repo_dir

    def _checkpoint_path(self, variant: str) -> Tuple[str, str]:
        env_key = _MATCHFORMER_DEBUG_ENV_BY_VARIANT.get(str(variant))
        if not env_key:
            raise RuntimeError(f"Unsupported MatchFormer debug variant '{variant}'")
        ckpt_path = _env_or_settings(env_key, "")
        if not ckpt_path:
            raise RuntimeError(f"MatchFormer debug matcher is missing env {env_key}")
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(
                f"MatchFormer debug checkpoint not found for {variant}: {ckpt_path}"
            )
        return ckpt_path, env_key

    def _resolve_device(self, key: str) -> torch.device:
        requested_device = _env_or_settings("MATCHFORMER_DEBUG_DEVICE", "auto").lower()
        preferred = _preferred_torch_device()
        if requested_device in {"", "auto"}:
            return torch.device("cpu") if preferred.type == "mps" else preferred
        if requested_device == "cpu":
            return torch.device("cpu")
        if requested_device == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not bool(mps_backend.is_available()):
                raise RuntimeError(
                    "MatchFormer debug matcher requested MATCHFORMER_DEBUG_DEVICE=mps but MPS is unavailable"
                )
            return torch.device("mps")
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "MatchFormer debug matcher requested MATCHFORMER_DEBUG_DEVICE=cuda but CUDA is unavailable"
                )
            return torch.device("cuda")
        raise RuntimeError(
            f"MatchFormer debug matcher has invalid MATCHFORMER_DEBUG_DEVICE='{requested_device}'. "
            "Use one of: auto, cpu, mps, cuda."
        )

    def _build(self, key: str, preferred: torch.device) -> Tuple[Any, Dict[str, str]]:
        repo_dir = self._repo_dir()
        ckpt_path, env_key = self._checkpoint_path(key)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
                    category=FutureWarning,
                )
                from config.defaultmf import get_cfg_defaults
                from model.matchformer import Matchformer
        except Exception as exc:
            raise RuntimeError(
                f"MatchFormer debug matcher failed importing repo modules from {repo_dir}: {exc}"
            ) from exc

        try:
            cfg = get_cfg_defaults()
        except Exception as exc:
            raise RuntimeError(
                f"MatchFormer debug matcher failed building default config: {exc}"
            ) from exc
        cfg = _matchformer_config_for_variant(cfg, key)
        cfg_lower = _cfg_to_lower_dict(cfg)
        model_cfg = cfg_lower.get("matchformer")
        if not isinstance(model_cfg, dict):
            raise RuntimeError("MatchFormer debug matcher config is invalid: missing 'matchformer' block")

        matcher = Matchformer(config=model_cfg)
        try:
            checkpoint = torch.load(
                ckpt_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        except Exception as exc:
            raise RuntimeError(
                f"MatchFormer debug matcher failed loading checkpoint '{ckpt_path}': {exc}"
            ) from exc

        if isinstance(checkpoint, dict):
            nested_state = checkpoint.get("state_dict")
            state_dict = nested_state if isinstance(nested_state, dict) else checkpoint
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"MatchFormer debug matcher checkpoint has invalid state_dict format: {ckpt_path}"
            )

        model_state_dict: Dict[str, Any] = {}
        for raw_key, value in state_dict.items():
            normalized_key = str(raw_key)
            if normalized_key.startswith("matcher."):
                normalized_key = normalized_key[len("matcher."):]
            elif normalized_key.startswith("model.matcher."):
                normalized_key = normalized_key[len("model.matcher."):]
            elif normalized_key.startswith("model."):
                normalized_key = normalized_key[len("model."):]
            model_state_dict[normalized_key] = value

        try:
            incompatible = matcher.load_state_dict(model_state_dict, strict=False)
        except Exception as exc:
            raise RuntimeError(
                f"MatchFormer debug matcher failed loading state_dict for {key}: {exc}"
            ) from exc

        requested_device = _env_or_settings("MATCHFORMER_DEBUG_DEVICE", "auto").lower() or "auto"
        return matcher, {
            "variant": key,
            "repo_dir": repo_dir,
            "ckpt_path": ckpt_path,
            "env_key": env_key,
            "model_class": matcher.__class__.__name__,
            "requested_device": requested_device,
            "missing_keys": str(len(getattr(incompatible, "missing_keys", []) or [])),
            "unexpected_keys": str(len(getattr(incompatible, "unexpected_keys", []) or [])),
        }

    def _log_loaded(self, key: str, preferred: torch.device) -> None:
        meta = self._meta_cache[key]
        logger.info(
            "Loaded MatchFormer debug matcher (%s) from %s on %s (requested=%s) [missing=%s unexpected=%s]",
            key,
            meta["ckpt_path"],
            preferred,
            meta["requested_device"],
            meta["missing_keys"],
            meta["unexpected_keys"],
        )

    def _log_moved(self, key: str, current: torch.device, preferred: torch.device) -> None:
        logger.info(
            "Moved cached MatchFormer debug matcher (%s) from %s to %s",
            key,
            current,
            preferred,
        )


class _RomaDebugMatcherLoader(_BaseMatcherLoader):
    def _resolve_device(self, key: str) -> torch.device:
        requested_device = _env_or_settings("ROMA_DEBUG_DEVICE", "auto").lower()
        preferred_device = _preferred_torch_device()
        if requested_device in {"", "auto"}:
            return torch.device("cpu") if preferred_device.type == "mps" else preferred_device
        if requested_device == "cpu":
            return torch.device("cpu")
        if requested_device == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not bool(mps_backend.is_available()):
                raise RuntimeError("RoMa-v2 debug matcher requested ROMA_DEBUG_DEVICE=mps but MPS is unavailable")
            return torch.device("mps")
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("RoMa-v2 debug matcher requested ROMA_DEBUG_DEVICE=cuda but CUDA is unavailable")
            return torch.device("cuda")
        raise RuntimeError(
            f"RoMa-v2 debug matcher has invalid ROMA_DEBUG_DEVICE='{requested_device}'. "
            "Use one of: auto, cpu, mps, cuda."
        )

    def _build(self, key: str, preferred: torch.device) -> Tuple[Any, Dict[str, str]]:
        try:
            from romatch import roma_outdoor
        except Exception as exc:
            raise RuntimeError(
                "RoMa-v2 debug matcher requires package 'romatch'. Install it in media-service venv first."
            ) from exc

        try:
            matcher = roma_outdoor(device=str(preferred))
        except TypeError:
            matcher = roma_outdoor(device=preferred)
        except Exception as exc:
            raise RuntimeError(f"RoMa-v2 debug matcher failed to initialize: {exc}") from exc

        requested_device = _env_or_settings("ROMA_DEBUG_DEVICE", "auto").lower() or "auto"
        return matcher, {
            "variant": "outdoor",
            "model_class": matcher.__class__.__name__,
            "device": str(preferred),
            "requested_device": requested_device,
        }

    def _log_loaded(self, key: str, preferred: torch.device) -> None:
        meta = self._meta_cache[key]
        logger.info(
            "Loaded RoMa-v2 debug matcher (outdoor) on %s (requested=%s)",
            preferred,
            meta["requested_device"],
        )

    def _log_moved(self, key: str, current: torch.device, preferred: torch.device) -> None:
        logger.info(
            "Moved cached RoMa-v2 debug matcher (%s) from %s to %s",
            key,
            current,
            preferred,
        )


_kornia_loftr_loader = _KorniaLoFTRLoader()
_zju_debug_loader = _ZJUDebugMatcherLoader()
_matchformer_debug_loader = _MatchFormerDebugMatcherLoader()
_roma_debug_loader = _RomaDebugMatcherLoader()


def kornia_loftr_cache_hit(checkpoint: str) -> bool:
    return _kornia_loftr_loader.is_cached(str(checkpoint))


def load_loftr_checkpoint(checkpoint: str) -> Any:
    matcher, _ = _kornia_loftr_loader.load(str(checkpoint))
    return matcher


def load_zju_loftr_debug_variant(variant: str) -> Tuple[Any, Dict[str, str]]:
    return _zju_debug_loader.load(str(variant))


def matchformer_debug_cache_hit(variant: str) -> bool:
    return _matchformer_debug_loader.is_cached(str(variant))


def load_matchformer_debug_variant(variant: str) -> Tuple[Any, Dict[str, str]]:
    return _matchformer_debug_loader.load(str(variant))


def roma_debug_cache_hit() -> bool:
    return _roma_debug_loader.is_cached("outdoor")


def load_roma_v2_debug_matcher() -> Tuple[Any, Dict[str, str]]:
    return _roma_debug_loader.load("outdoor")
