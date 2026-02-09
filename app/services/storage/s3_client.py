"""S3 client for accessing photos and uploading artifacts."""
import logging
from io import BytesIO
from typing import Optional

import boto3
from botocore.config import Config
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


class S3Client:
    """S3 client wrapper for MinIO/S3 compatible storage."""

    def __init__(self):
        self._client = None
        self._resource = None

    @property
    def client(self):
        """Lazy-load S3 client."""
        if self._client is None:
            config = Config(
                signature_version='s3v4',
                s3={'addressing_style': 'path'}
            )
            self._client = boto3.client(
                's3',
                endpoint_url=settings.S3_ENDPOINT,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
                config=config,
            )
        return self._client

    def download_image(self, s3_uri: str) -> Image.Image:
        """Download image from S3 and return as PIL Image.

        Args:
            s3_uri: S3 URI (s3://bucket/key) or just the key

        Returns:
            PIL Image object
        """
        key = self._parse_key(s3_uri)
        logger.info(f"Downloading image from S3: {key}")

        response = self.client.get_object(Bucket=settings.S3_BUCKET, Key=key)
        image_data = response['Body'].read()
        return Image.open(BytesIO(image_data))

    def download_bytes(self, s3_uri: str) -> bytes:
        """Download raw bytes from S3.

        Args:
            s3_uri: S3 URI or key

        Returns:
            Raw bytes
        """
        key = self._parse_key(s3_uri)
        response = self.client.get_object(Bucket=settings.S3_BUCKET, Key=key)
        return response['Body'].read()

    def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload bytes to S3.

        Args:
            key: S3 object key
            data: Bytes to upload
            content_type: Content type

        Returns:
            S3 URI
        """
        logger.info(f"Uploading to S3: {key}")
        self.client.put_object(
            Bucket=settings.S3_BUCKET,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return f"s3://{settings.S3_BUCKET}/{key}"

    def file_exists(self, s3_uri: str) -> bool:
        """Check if file exists in S3."""
        key = self._parse_key(s3_uri)
        try:
            self.client.head_object(Bucket=settings.S3_BUCKET, Key=key)
            return True
        except Exception:
            return False

    def _parse_key(self, s3_uri: str) -> str:
        """Parse S3 URI to extract key."""
        if s3_uri.startswith("s3://"):
            # s3://bucket/key format
            parts = s3_uri[5:].split("/", 1)
            if len(parts) == 2:
                return parts[1]
            return ""
        # Assume it's already a key
        return s3_uri


# Singleton instance
s3_client = S3Client()
