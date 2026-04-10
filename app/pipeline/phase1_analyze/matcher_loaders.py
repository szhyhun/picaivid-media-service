"""Local/S3 artifact hydration helpers for phase-1 model assets."""
from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from typing import Dict, Tuple
from urllib.parse import urlparse

import boto3

from app.core.config import settings

logger = logging.getLogger(__name__)

_PLACEHOLDER_CREDENTIALS = {
    "",
    "use-instance-role-or-set-if-needed",
    "unset",
    "none",
    "null",
}


def _optional_setting(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _aws_credentials_kwargs() -> Dict[str, str]:
    access_key = _optional_setting(settings.AWS_ACCESS_KEY_ID)
    secret_key = _optional_setting(settings.AWS_SECRET_ACCESS_KEY)
    if (
        not access_key
        or not secret_key
        or access_key.lower() in _PLACEHOLDER_CREDENTIALS
        or secret_key.lower() in _PLACEHOLDER_CREDENTIALS
    ):
        return {}
    return {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }


def _env_or_settings(name: str, default: str = "") -> str:
    raw_env = str(os.getenv(name, "")).strip()
    if raw_env:
        return raw_env
    raw_settings = getattr(settings, name, None)
    if raw_settings is None:
        return str(default)
    return str(raw_settings).strip() or str(default)


def _s3_client():
    kwargs = {
        "region_name": settings.AWS_REGION,
        **_aws_credentials_kwargs(),
    }
    endpoint_url = _optional_setting(settings.S3_ENDPOINT)
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **kwargs)


def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    parsed = urlparse(str(s3_uri))
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise RuntimeError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _download_file_from_s3(target_path: str, s3_uri: str) -> str:
    bucket, key = _parse_s3_uri(s3_uri)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    tmp_path = f"{target_path}.tmp"
    client = _s3_client()
    client.download_file(bucket, key, tmp_path)
    os.replace(tmp_path, target_path)
    return target_path


def _extract_archive(archive_path: str, destination_dir: str) -> None:
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(destination_dir)
        return
    if archive_path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(destination_dir)
        return
    raise RuntimeError(f"Unsupported matcher asset archive format: {archive_path}")


def _ensure_local_repo(path_value: str, s3_uri_env: str) -> str:
    if path_value and os.path.isdir(path_value):
        return path_value
    s3_uri = _env_or_settings(s3_uri_env, "")
    if not path_value:
        raise RuntimeError(f"Missing local repo path and no target path configured for {s3_uri_env}")
    if not s3_uri:
        raise RuntimeError(f"Required repo not found locally and no S3 fallback configured: {path_value}")

    logger.info("Hydrating model repo from S3: %s -> %s", s3_uri, path_value)
    os.makedirs(os.path.dirname(path_value), exist_ok=True)
    tmp_root = tempfile.mkdtemp(prefix="model_repo_")
    try:
        archive_name = os.path.basename(urlparse(str(s3_uri)).path) or "model_repo.zip"
        archive_path = os.path.join(tmp_root, archive_name)
        _download_file_from_s3(archive_path, s3_uri)
        extracted_root = os.path.join(tmp_root, "extracted")
        os.makedirs(extracted_root, exist_ok=True)
        _extract_archive(archive_path, extracted_root)

        entries = [os.path.join(extracted_root, entry) for entry in os.listdir(extracted_root)]
        if len(entries) == 1 and os.path.isdir(entries[0]):
            source_dir = entries[0]
        else:
            source_dir = extracted_root

        tmp_target = f"{path_value}.tmp"
        if os.path.isdir(tmp_target):
            shutil.rmtree(tmp_target)
        shutil.copytree(source_dir, tmp_target)
        if os.path.isdir(path_value):
            shutil.rmtree(path_value)
        os.replace(tmp_target, path_value)
        return path_value
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _ensure_local_file(path_value: str, s3_uri_env: str) -> str:
    if path_value and os.path.isfile(path_value):
        return path_value
    s3_uri = _env_or_settings(s3_uri_env, "")
    if not path_value:
        raise RuntimeError(f"Missing local path and no target path configured for {s3_uri_env}")
    if not s3_uri:
        raise RuntimeError(f"Required file not found locally and no S3 fallback configured: {path_value}")
    logger.info("Hydrating model file from S3: %s -> %s", s3_uri, path_value)
    return _download_file_from_s3(path_value, s3_uri)
