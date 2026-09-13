"""Music selection and beat-grid primitives."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.core.config import settings
from app.pipeline.phase3_timeline.music import (
    MusicAnalysisError,
    _suffix,
    analyze_audio,
    download_music,
)


class MusicUriTests(unittest.TestCase):
    def test_suffix_is_taken_from_the_source_path(self):
        self.assertEqual(_suffix("s3://bucket/music/track.mp3"), ".mp3")
        self.assertEqual(_suffix("https://cdn.example.com/track"), ".audio")

    def test_local_http_is_allowed_only_in_development(self):
        destination = Path("/tmp/picaivid-music-test.audio")
        with patch.object(settings, "ENVIRONMENT", "production"):
            with self.assertRaises(MusicAnalysisError):
                download_music("http://localhost:9000/music.mp3", destination)

    def test_unsupported_uri_scheme_is_rejected(self):
        with self.assertRaises(MusicAnalysisError):
            download_music("file:///tmp/music.mp3", "/tmp/music.audio")


class BeatAnalysisTests(unittest.TestCase):
    def test_short_audio_is_rejected(self):
        with patch("app.pipeline.phase3_timeline.music.subprocess.run"), \
             patch("app.pipeline.phase3_timeline.music.np.fromfile",
                   return_value=np.zeros(100, dtype=np.float32)):
            with self.assertRaises(MusicAnalysisError):
                analyze_audio("/tmp/music.mp3")


if __name__ == "__main__":
    unittest.main()
