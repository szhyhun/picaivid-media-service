"""SQS commands are handled by the worker, not the HTTP API."""
import unittest
from unittest.mock import MagicMock, patch

from app import worker
from app.workers import cpu


class WorkerCommandTests(unittest.TestCase):
    def test_cpu_worker_uses_the_cpu_queue(self):
        with patch.object(worker.settings, "WORKER_TYPE", "cpu"), patch.object(
            worker.settings, "SQS_CPU_QUEUE_URL", "cpu-queue"
        ), patch.object(worker.settings, "SQS_QUEUE_URL", "gpu-queue"):
            self.assertEqual(worker._queue_url(), "cpu-queue")

    def test_gpu_worker_uses_the_analysis_queue(self):
        with patch.object(worker.settings, "WORKER_TYPE", "gpu"), patch.object(
            worker.settings, "SQS_CPU_QUEUE_URL", "cpu-queue"
        ), patch.object(worker.settings, "SQS_QUEUE_URL", "gpu-queue"):
            self.assertEqual(worker._queue_url(), "gpu-queue")

    def test_timeline_generation_command_runs_on_cpu_worker(self):
        db = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = db
        context.__exit__.return_value = False

        with patch.object(worker.settings, "WORKER_TYPE", "cpu"), patch.object(
            cpu, "get_db_context", return_value=context
        ), patch.object(cpu, "run_timeline_generation") as run_generation:
            worker.process_message(
                {
                    "action": "timeline_generate",
                    "project_id": "project-1",
                    "shot_uids": ["shot-1"],
                    "resolution": "720p",
                }
            )

        run_generation.assert_called_once_with(
            db,
            "project-1",
            shot_uids=["shot-1"],
            resolution="720p",
        )

    def test_timeline_commands_are_not_run_by_gpu_worker(self):
        with patch.object(worker.settings, "WORKER_TYPE", "gpu"):
            with self.assertRaises(RuntimeError):
                worker.process_message(
                    {"action": "timeline_assemble", "project_id": "project-1"}
                )


if __name__ == "__main__":
    unittest.main()
