"""SQS consumer for processing job messages."""
import json
import logging
import time
from typing import Callable, Optional

import boto3

from app.core.config import settings

logger = logging.getLogger(__name__)


class SQSConsumer:
    """SQS message consumer for job processing."""

    def __init__(
        self,
        queue_url: str,
        handler: Callable[[dict], None],
        visibility_timeout: int = 3600,
        wait_time_seconds: int = 20,
    ):
        self.queue_url = queue_url
        self.handler = handler
        self.visibility_timeout = visibility_timeout
        self.wait_time_seconds = wait_time_seconds
        self._running = False
        self._client = None

    @property
    def client(self):
        """Lazy-load SQS client."""
        if self._client is None:
            self._client = boto3.client(
                'sqs',
                endpoint_url=settings.SQS_ENDPOINT,
                aws_access_key_id=settings.SQS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.SQS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
            )
        return self._client

    def start(self) -> None:
        """Start consuming messages from SQS."""
        self._running = True
        logger.info(f"Starting SQS consumer for queue: {self.queue_url}")

        while self._running:
            try:
                self._poll_messages()
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.stop()
            except Exception as e:
                logger.error(f"Error polling messages: {e}")
                time.sleep(5)  # Back off on error

    def stop(self) -> None:
        """Stop consuming messages."""
        self._running = False
        logger.info("Stopping SQS consumer")

    def _poll_messages(self) -> None:
        """Poll for messages from SQS."""
        response = self.client.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=self.wait_time_seconds,
            VisibilityTimeout=self.visibility_timeout,
        )

        messages = response.get('Messages', [])
        if not messages:
            return

        for message in messages:
            self._process_message(message)

    def _process_message(self, message: dict) -> None:
        """Process a single SQS message."""
        receipt_handle = message['ReceiptHandle']
        message_id = message['MessageId']

        try:
            body = json.loads(message['Body'])
            logger.info(f"Processing message {message_id}: {body}")

            self.handler(body)

            # Delete message on success
            self.client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
            )
            logger.info(f"Successfully processed and deleted message {message_id}")

        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
            # Message will become visible again after visibility timeout
            raise


def send_message(message: dict, queue_url: Optional[str] = None) -> str:
    """Send a message to SQS.

    Args:
        message: Message body as dict
        queue_url: Optional queue URL (defaults to settings)

    Returns:
        Message ID
    """
    client = boto3.client(
        'sqs',
        endpoint_url=settings.SQS_ENDPOINT,
        aws_access_key_id=settings.SQS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.SQS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )

    url = queue_url or settings.SQS_QUEUE_URL
    response = client.send_message(
        QueueUrl=url,
        MessageBody=json.dumps(message),
    )

    logger.info(f"Sent message to SQS: {response['MessageId']}")
    return response['MessageId']
