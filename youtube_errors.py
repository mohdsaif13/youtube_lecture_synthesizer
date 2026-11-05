"""User-facing error classification for YouTube downloads."""


def classify_youtube_download_error(error):
    """Convert common yt-dlp failures into clear, actionable guidance."""
    message = str(error or "Unknown yt-dlp error")
    lowered = message.lower()

    if "sign in to confirm" in lowered or "confirm you're not a bot" in lowered:
        return (
            "YouTube requested account verification: sign in to confirm you're not a bot. "
            "This is a YouTube access challenge, not an invalid lecture URL; changing yt-dlp "
            "player clients cannot reliably remove it. For a video you are authorized to view, "
            "export a fresh YouTube cookies.txt file in Netscape format and add its complete "
            "contents to Streamlit Cloud → App settings → Secrets as YOUTUBE_COOKIES. Keep "
            "cookies private, never commit them, and remove/refresh them if they expire. "
            "Alternatively, download the lecture on your own device while signed in and use "
            "the Upload Lecture Video File tab. YouTube may still restrict downloads from "
            "cloud-hosted IP addresses even when cookies are supplied."
        )

    if "403" in lowered or "forbidden" in lowered:
        return (
            "YouTube refused this download from the hosting server (HTTP 403). This is an "
            "upstream access restriction. Try again later, supply a valid cookies.txt through "
            "Streamlit Secrets for an account authorized to view the video, or download the "
            "video locally and upload it through the app."
        )

    if "private video" in lowered or "members-only" in lowered or "age-restricted" in lowered:
        return (
            "This video requires account access or permission. Confirm that your account is "
            "authorized to view it, then provide a fresh cookies.txt through Streamlit Secrets "
            "or upload a local copy you are permitted to process."
        )

    return f"Unable to download the YouTube video: {message}"
