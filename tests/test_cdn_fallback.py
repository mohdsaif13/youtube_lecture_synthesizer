import unittest
from unittest.mock import patch, MagicMock
import inspect

import app


class CDNFallbackTests(unittest.TestCase):
    def test_extract_youtube_id_formats(self):
        self.assertEqual(app.extract_youtube_id("https://www.youtube.com/watch?v=kqtD5dpn9C8"), "kqtD5dpn9C8")
        self.assertEqual(app.extract_youtube_id("https://youtu.be/kqtD5dpn9C8"), "kqtD5dpn9C8")
        self.assertEqual(app.extract_youtube_id("https://www.youtube.com/embed/kqtD5dpn9C8"), "kqtD5dpn9C8")
        self.assertEqual(app.extract_youtube_id("https://www.youtube.com/shorts/kqtD5dpn9C8"), "kqtD5dpn9C8")

    def test_rate_limiter_defaults_and_caps(self):
        sig = inspect.signature(app.extract_slides)
        self.assertEqual(sig.parameters["max_slides_per_minute"].default, 2)
        self.assertEqual(sig.parameters["max_total_slides"].default, 35)
        self.assertEqual(sig.parameters["min_slide_duration"].default, 30.0)

    @patch("app.extract_slides_from_youtube_cdn")
    def test_bot_challenge_triggers_cdn_storyboard_fallback(self, mock_cdn):
        mock_ydl_module = MagicMock()
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = Exception("ERROR: [youtube] Sign in to confirm you're not a bot")
        mock_ydl_module.YoutubeDL.return_value.__enter__.return_value = mock_instance

        mock_cdn.return_value = ([{"slide_index": 1, "image_path": "slide.jpg"}], {"title": "Test Lecture"})

        with patch.object(app, "HAS_YTDLP", True), patch.object(app, "yt_dlp", mock_ydl_module):
            slides, meta = app.extract_slides(
                "https://www.youtube.com/watch?v=kqtD5dpn9C8",
                "/tmp/fake_dir"
            )

        mock_cdn.assert_called_once()
        self.assertEqual(len(slides), 1)
        self.assertEqual(meta["title"], "Test Lecture")

    @patch("app.extract_slides_from_youtube_cdn")
    def test_http_403_triggers_cdn_storyboard_fallback(self, mock_cdn):
        mock_ydl_module = MagicMock()
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = Exception("HTTP Error 403: Forbidden")
        mock_ydl_module.YoutubeDL.return_value.__enter__.return_value = mock_instance

        mock_cdn.return_value = ([{"slide_index": 1, "image_path": "slide.jpg"}], {"title": "Test Lecture"})

        with patch.object(app, "HAS_YTDLP", True), patch.object(app, "yt_dlp", mock_ydl_module):
            slides, meta = app.extract_slides(
                "https://www.youtube.com/watch?v=kqtD5dpn9C8",
                "/tmp/fake_dir"
            )

        mock_cdn.assert_called_once()
        self.assertEqual(len(slides), 1)


if __name__ == "__main__":
    unittest.main()
