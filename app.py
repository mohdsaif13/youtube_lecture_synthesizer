"""
YouTube Lecture Synthesizer — Complete Production Streamlit Application
Extracts presentation slides, transcribes audio, runs OCR, synthesizes notes via LLM,
and compiles comprehensive lecture notes into a publication-quality PDF.
"""

import os
import re
import sys
import time
import shutil
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Optional & Conditional Imports with Safe Fallbacks
# ---------------------------------------------------------------------------
# SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# OCR
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# YouTube Transcript API
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
    HAS_YT_TRANSCRIPT = True
except ImportError:
    HAS_YT_TRANSCRIPT = False

# yt-dlp
try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False

# Faster-Whisper / Whisper Fallback
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

try:
    import whisper
    HAS_OPENAI_WHISPER = True
except ImportError:
    HAS_OPENAI_WHISPER = False

# ReportLab for PDF Compilation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, PageBreak, KeepTogether, HRFlowable
    )
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# LLM Providers (Google GenAI & OpenAI)
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GOOGLE_GENAI_NEW = True
except ImportError:
    HAS_GOOGLE_GENAI_NEW = False

try:
    import google.generativeai as legacy_genai
    HAS_GOOGLE_GENAI_LEGACY = True
except ImportError:
    HAS_GOOGLE_GENAI_LEGACY = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# Helper: Tesseract Binary Auto-Detection
# ---------------------------------------------------------------------------
def detect_tesseract_path(user_path: Optional[str] = None) -> Optional[str]:
    """Locates the tesseract executable across PATH and standard OS installation paths."""
    if user_path and os.path.exists(user_path):
        return user_path

    # Check PATH
    which_path = shutil.which("tesseract")
    if which_path and os.path.exists(which_path):
        return which_path

    # Check common Windows paths
    common_windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in common_windows_paths:
        if os.path.exists(p):
            return p

    # Common Unix/macOS paths
    common_unix_paths = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    for p in common_unix_paths:
        if os.path.exists(p):
            return p

    return None


def escape_xml(text: Any) -> str:
    """Escapes XML entities for safe ReportLab Paragraph rendering."""
    if not text:
        return ""
    # & must always be replaced first to prevent double-escaping
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def format_markdown_for_reportlab(text: str) -> str:
    """Safely converts common Markdown tokens (bold, italic, inline code) to ReportLab XML tags."""
    if not text:
        return ""
    escaped = escape_xml(text)
    # Inline code: `code` -> <font name="Courier">code</font>
    escaped = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', escaped)
    # Bold: **text** or __text__ -> <b>text</b>
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
    escaped = re.sub(r'__(.+?)__', r'<b>\1</b>', escaped)
    # Italic: *text* or _text_ -> <i>text</i>
    escaped = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', escaped)
    escaped = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'<i>\1</i>', escaped)
    return escaped


def extract_youtube_id(url: str) -> Optional[str]:
    """Extracts the 11-character YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',
    ]
    for pat in patterns:
        match = re.search(pat, url)
        if match:
            return match.group(1)
    return None


def format_timestamp(seconds: float) -> str:
    """Formats seconds into MM:SS or HH:MM:SS format."""
    secs = int(seconds)
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    remaining_secs = secs % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{remaining_secs:02d}"
    return f"{minutes:02d}:{remaining_secs:02d}"


def get_yt_dlp_options(
    output_path: str,
    is_audio: bool = False,
    client: str = "web"
) -> Dict[str, Any]:
    """
    Build conservative yt-dlp options for YouTube.

    A player-client switch is not an authentication bypass. Use a valid,
    user-supplied cookies file only when the user is authorized to access
    the video. Never place browser cookies or credentials in source control.
    """
    opts: Dict[str, Any] = {
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "source_address": "0.0.0.0",
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 20,
        "extractor_args": {
            "youtube": {
                "player_client": [client],
            }
        },
    }

    if is_audio:
        opts["format"] = "bestaudio[ext=m4a]/bestaudio/best"
    else:
        opts["format"] = "best[height<=720][ext=mp4]/best[height<=720]/best"

    # Cookie file may be supplied locally or through Streamlit Secrets.
    cookie_file = os.environ.get("YOUTUBE_COOKIES_FILE")
    if cookie_file and os.path.isfile(cookie_file):
        opts["cookiefile"] = cookie_file
    elif os.path.isfile("cookies.txt"):
        opts["cookiefile"] = "cookies.txt"
    else:
        try:
            cookie_text = st.secrets.get("YOUTUBE_COOKIES", "")
        except Exception:
            cookie_text = ""
        if cookie_text:
            cookie_path = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
            with open(cookie_path, "w", encoding="utf-8") as cookie_handle:
                cookie_handle.write(str(cookie_text).replace("\\r\\n", "\\n"))
            try:
                os.chmod(cookie_path, 0o600)
            except OSError:
                pass
            opts["cookiefile"] = cookie_path

    return opts


def classify_youtube_download_error(error: Any) -> str:
    """Return actionable, user-safe guidance for common YouTube download failures."""
    message = str(error or "")
    lower = message.lower()

    if "sign in to confirm" in lower or "confirm you're not a bot" in lower:
        return (
            "YouTube requires an authenticated session for this request. "
            "The Streamlit Cloud server cannot use your browser session automatically. "
            "For a video you are authorized to access, export a fresh YouTube cookies.txt "
            "file in Netscape format and add its complete contents to Streamlit Secrets "
            "as YOUTUBE_COOKIES. Treat that file like a password and never commit it. "
            "Alternatively, download the video on your own computer while signed in and "
            "use the Upload Lecture Video File tab. Captions may still be available even "
            "when video download is blocked."
        )
    if "403" in lower or "forbidden" in lower:
        return (
            "YouTube refused the download from this hosting server (HTTP 403). "
            "This is an upstream access restriction, not a problem with the lecture URL. "
            "Try again later, use an authorized cookies.txt through Streamlit Secrets, "
            "or download the video locally and upload the file."
        )
    if "private video" in lower or "members-only" in lower or "age-restricted" in lower:
        return (
            "This video requires an account or permission. Confirm that your account is "
            "authorized to watch it, then provide a fresh cookies.txt through Streamlit "
            "Secrets, or upload a local copy you are permitted to use."
        )
    return f"Unable to download the YouTube video: {message}"




# ---------------------------------------------------------------------------
# Core Modular Function 1: Slide Detection & Extraction (extract_slides)
# ---------------------------------------------------------------------------
def extract_slides(
    video_path_or_url: str,
    output_dir: str,
    sample_interval: float = 2.5,
    ssim_threshold: float = 0.82,
    min_slide_duration: float = 3.0,
    progress_callback=None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Downloads or opens video stream, samples frames at sample_interval seconds,
    and applies Structural Similarity Index (SSIM) to detect and extract unique slide keyframes.

    Returns:
        Tuple of (slides_list, video_metadata)
    """
    slides_dir = os.path.join(output_dir, "slides")
    os.makedirs(slides_dir, exist_ok=True)

    video_meta = {
        "title": "Extracted Lecture",
        "uploader": "Unknown Lecturer",
        "duration": 0,
        "thumbnail": None,
        "source_url": video_path_or_url,
    }

    local_video_path = video_path_or_url

    # Check if input is a YouTube URL
    is_url = video_path_or_url.startswith("http://") or video_path_or_url.startswith("https://")
    if is_url:
        if not HAS_YTDLP:
            raise RuntimeError("yt-dlp is required to download YouTube videos. Run: pip install yt-dlp")

        # Avoid rapid client-rotation retries: YouTube login challenges are
        # authorization/risk decisions and cannot be reliably fixed by changing clients.
        video_filename = os.path.join(output_dir, "source_lecture.mp4")
        last_exception = None
        download_success = False
        try:
            if progress_callback:
                progress_callback(0.06, "Requesting YouTube video stream...")
            ydl_opts = get_yt_dlp_options(video_filename, is_audio=False)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_path_or_url, download=True)
                video_meta["title"] = info.get("title", "Lecture Video")
                video_meta["uploader"] = info.get("uploader", "Academic Instructor")
                video_meta["duration"] = info.get("duration", 0)
                video_meta["thumbnail"] = info.get("thumbnail")
            download_success = os.path.isfile(video_filename) and os.path.getsize(video_filename) > 0
        except Exception as exc:
            last_exception = exc
            if os.path.exists(video_filename):
                try:
                    os.remove(video_filename)
                except OSError:
                    pass

        if not download_success:
            raise RuntimeError(classify_youtube_download_error(last_exception))

