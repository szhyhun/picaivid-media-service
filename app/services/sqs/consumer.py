"""SQS consumer for processing job messages."""
import json
import logging
import threading
import time
from typing import Callable, Optional

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


def _aws_credentials_kwargs() -> dict[str, str]:
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


def _sqs_client_kwargs() -> dict:
    kwargs = {
        "region_name": settings.AWS_REGION,
        **_aws_credentials_kwargs(),
    }
    endpoint_url = _optional_setting(settings.SQS_ENDPOINT)
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return kwargs


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
            self._client = boto3.client('sqs', **_sqs_client_kwargs())
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

    def _extend_visibility(self, receipt_handle: str, message_id: str,
                           stop: threading.Event) -> None:
        """Keep one in-flight message invisible for as long as work continues.

        A fixed visibility timeout is a bet that the work finishes inside it.
        Generation does not honour that bet: shots are capped individually, so
        a run of slow ones outlives any constant. When the bet is lost SQS
        hands the same message to a second worker, which then repeats work the
        first is still doing.

        Renewing on a timer removes the bet. The heartbeat stops when the
        handler returns, so a worker that dies still releases its message and
        the job is genuinely retried rather than lost.
        """
        interval = max(10.0, self.visibility_timeout / 3.0)
        while not stop.wait(interval):
            try:
                self.client.change_message_visibility(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=self.visibility_timeout,
                )
            except Exception as error:
                # Losing the extension is not fatal on its own: the billing
                # guard still stops a redelivered message from buying anything
                # twice. Log it and let the work finish.
                logger.warning("VISIBILITY_EXTEND_FAILED message=%s error=%s",
                               message_id, error)
                return

    def _process_message(self, message: dict) -> None:
        """Process a single SQS message."""
        receipt_handle = message['ReceiptHandle']
        message_id = message['MessageId']

        stop = threading.Event()
        keepalive = threading.Thread(
            target=self._extend_visibility,
            args=(receipt_handle, message_id, stop),
            name=f"sqs-visibility-{message_id[:8]}",
            daemon=True,
        )
        keepalive.start()

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
        finally:
            stop.set()
            keepalive.join(timeout=5.0)


def send_message(message: dict, queue_url: Optional[str] = None) -> str:
    """Send a message to SQS.

    Args:
        message: Message body as dict
        queue_url: Optional queue URL (defaults to settings)

    Returns:
        Message ID
    """
    client = boto3.client('sqs', **_sqs_client_kwargs())

    url = queue_url or settings.SQS_QUEUE_URL
    response = client.send_message(
        QueueUrl=url,
        MessageBody=json.dumps(message),
    )

    logger.info(f"Sent message to SQS: {response['MessageId']}")
    return response['MessageId']
