import unittest

from youtube_errors import classify_youtube_download_error


class YouTubeDownloadErrorTests(unittest.TestCase):
    def test_bot_verification_error_points_to_authorized_cookie_or_upload_path(self):
        message = classify_youtube_download_error(
            "ERROR: [youtube] t4jKBFuFZGA: Sign in to confirm you're not a bot"
        )
        self.assertIn("YOUTUBE_COOKIES", message)
        self.assertIn("Upload Lecture Video File", message)
        self.assertIn("not an invalid lecture URL", message)

    def test_forbidden_error_is_distinguished_from_bot_check(self):
        message = classify_youtube_download_error("HTTP Error 403: Forbidden")
        self.assertIn("HTTP 403", message)
        self.assertNotIn("Sign in to confirm", message)

    def test_private_video_error_mentions_permission(self):
        message = classify_youtube_download_error("This is a private video")
        self.assertIn("requires account access or permission", message)

    def test_unknown_error_preserves_original_message(self):
        message = classify_youtube_download_error("unexpected extractor failure")
        self.assertIn("unexpected extractor failure", message)


if __name__ == "__main__":
    unittest.main()
