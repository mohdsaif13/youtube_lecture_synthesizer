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

from youtube_errors import classify_youtube_download_error

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
    client: str = "android"
) -> Dict[str, Any]:
    """
    Builds resilient yt-dlp configuration that bypasses YouTube 403 Forbidden errors
    on datacenter/cloud IPs (Streamlit Community Cloud, AWS, GCP, etc.).
    """
    opts: Dict[str, Any] = {
        'outtmpl': output_path,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'source_address': '0.0.0.0',  # Force IPv4 to bypass banned IPv6 cloud ranges
        'extractor_args': {
            'youtube': {
                'player_client': [client],
                'player_skip': ['webpage', 'configs'],
            }
        },
    }

    # Only apply browser headers for web clients; let mobile clients use native signatures
    if client in ("mweb", "web"):
        opts['http_headers'] = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
            ),
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Sec-Fetch-Mode': 'navigate',
        }

    if is_audio:
        opts['format'] = 'ba[ext=m4a]/140/ba/best'
    else:
        # Prioritize pre-muxed MP4 formats (22: 720p, 18: 360p) which bypass YouTube SABR 403
        opts['format'] = '22/18/best[height<=720][ext=mp4]/best'

    # Cookies are optional and must belong to an account authorized to view the video.
    # Never log, display, or commit cookie contents.
    cookie_file = os.environ.get("YOUTUBE_COOKIES_FILE", "").strip()
    if not cookie_file or not os.path.isfile(cookie_file):
        cookie_file = "cookies.txt" if os.path.isfile("cookies.txt") else ""

    if not cookie_file:
        try:
            cookie_text = st.secrets.get("YOUTUBE_COOKIES", "")
        except Exception:
            cookie_text = os.environ.get("YOUTUBE_COOKIES", "")

        if cookie_text:
            temp_cookie_path = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
            with open(temp_cookie_path, "w", encoding="utf-8") as cookie_handle:
                cookie_handle.write(str(cookie_text).replace("\r\n", "\n"))
            try:
                os.chmod(temp_cookie_path, 0o600)
            except OSError:
                pass
            cookie_file = temp_cookie_path

    if cookie_file:
        opts["cookiefile"] = cookie_file

    return opts


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

        # Try compatible player clients, preferring web clients when account cookies exist.
        # Cookies are optional and are never committed to the repository.
        video_filename = os.path.join(output_dir, "source_lecture.mp4")
        has_cookie_auth = (
            os.path.isfile("cookies.txt")
            or bool(os.environ.get("YOUTUBE_COOKIES_FILE"))
            or bool(os.environ.get("YOUTUBE_COOKIES"))
        )
        try:
            has_cookie_auth = has_cookie_auth or bool(st.secrets.get("YOUTUBE_COOKIES", ""))
        except Exception:
            pass
        client_candidates = (
            ["web", "mweb", "android", "ios", "tv_embedded"]
            if has_cookie_auth else ["android", "ios", "mweb", "tv_embedded"]
        )
        download_success = False
        last_exception = None
        download_errors = []

        for idx, client in enumerate(client_candidates):
            try:
                if progress_callback:
                    progress_callback(
                        0.05 + idx * 0.02,
                        f"Connecting to video stream (client engine: {client})..."
                    )
                ydl_opts = get_yt_dlp_options(video_filename, is_audio=False, client=client)
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_path_or_url, download=True)
                    video_meta["title"] = info.get("title", "Lecture Video")
                    video_meta["uploader"] = info.get("uploader", "Academic Instructor")
                    video_meta["duration"] = info.get("duration", 0)
                    video_meta["thumbnail"] = info.get("thumbnail")
                download_success = True
                break
            except Exception as e:
                last_exception = e
                error_text = str(e)
                download_errors.append(error_text)
                if os.path.exists(video_filename):
                    try:
                        os.remove(video_filename)
                    except Exception:
                        pass

                # A bot-verification challenge is not fixed by rotating player clients.
                # Stop early and show the user the supported authentication/upload options.
                lowered_error = error_text.lower()
                if "sign in to confirm" in lowered_error or "confirm you're not a bot" in lowered_error:
                    break
                continue

        if not download_success:
            combined_errors = "\\n".join(download_errors) if download_errors else last_exception
            raise RuntimeError(classify_youtube_download_error(combined_errors))

        local_video_path = video_filename
    else:
        video_meta["title"] = Path(video_path_or_url).stem.replace("_", " ").title()
        video_meta["uploader"] = "Uploaded Presentation"
        video_meta["source_url"] = "Local Upload"

    # Open video with OpenCV
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video stream from {local_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if total_frames > 0 else video_meta.get("duration", 0)
    if video_meta["duration"] == 0:
        video_meta["duration"] = video_duration

    # Frame sampling calculation
    step_frames = max(1, int(sample_interval * fps))
    slides: List[Dict[str, Any]] = []

    prev_gray_small = None
    last_slide_time = -min_slide_duration  # allow first slide at t=0
    current_frame_idx = 0
    slide_counter = 0

    if progress_callback:
        progress_callback(0.15, "Analyzing video frames with Computer Vision (SSIM)...")

    # Target comparison resolution for rapid, robust SSIM calculation
    COMPARE_SIZE = (320, 180)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames at the sampled interval
        if current_frame_idx % step_frames == 0:
            current_time = current_frame_idx / fps
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray_full, COMPARE_SIZE, interpolation=cv2.INTER_AREA)

            is_slide_transition = False
            similarity_score = 1.0

            if prev_gray_small is None:
                # First frame is always the opening slide if not completely black
                if np.mean(gray_small) > 10:
                    is_slide_transition = True
                    similarity_score = 0.0
            else:
                # Calculate SSIM or frame difference fallback
                if HAS_SKIMAGE:
                    similarity_score = ssim(prev_gray_small, gray_small, data_range=255)
                else:
                    # Fallback normalized absolute difference
                    diff = cv2.absdiff(prev_gray_small, gray_small)
                    similarity_score = 1.0 - (np.mean(diff) / 255.0)

                # Check transition threshold and min duration debounce
                if similarity_score < ssim_threshold:
                    if (current_time - last_slide_time) >= min_slide_duration:
                        # Extra filter: discard near-pitch-black frames (e.g. video transitions)
                        if np.mean(gray_small) > 12:
                            is_slide_transition = True

            if is_slide_transition:
                slide_counter += 1
                slide_filename = f"slide_{slide_counter:03d}_{int(current_time)}s.jpg"
                slide_path = os.path.join(slides_dir, slide_filename)
                cv2.imwrite(slide_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

                # Update end_time of previous slide
                if slides:
                    slides[-1]["end_time"] = current_time

                slides.append({
                    "slide_index": slide_counter,
                    "timestamp": current_time,
                    "timestamp_str": format_timestamp(current_time),
                    "end_time": video_duration,  # default until next slide
                    "image_path": slide_path,
                    "similarity": round(float(similarity_score), 3)
                })

                prev_gray_small = gray_small
                last_slide_time = current_time

            # Progress update
            if progress_callback and total_frames > 0:
                pct = 0.15 + (current_frame_idx / total_frames) * 0.25
                progress_callback(
                    min(0.40, pct),
                    f"Sampled at {format_timestamp(current_time)} | Found {slide_counter} slides..."
                )

        current_frame_idx += 1

    cap.release()

    # Safety: If no slide transitions detected (e.g. single continuous slide), capture first frame
    if not slides and current_frame_idx > 0:
        cap = cv2.VideoCapture(local_video_path)
        ret, frame = cap.read()
        if ret:
            slide_path = os.path.join(slides_dir, "slide_001_0s.jpg")
            cv2.imwrite(slide_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            slides.append({
                "slide_index": 1,
                "timestamp": 0.0,
                "timestamp_str": "00:00",
                "end_time": video_duration,
                "image_path": slide_path,
                "similarity": 0.0
            })
        cap.release()

    # Clean up source video if it was downloaded into a temporary directory
    # (keeps extracted slide images intact)
    return slides, video_meta


# ---------------------------------------------------------------------------
# Core Modular Function 2: Audio & Transcript Retrieval (get_transcript)
# ---------------------------------------------------------------------------
def get_transcript(
    video_url: str,
    output_dir: str,
    progress_callback=None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Fetches timestamped transcript segments for a YouTube URL using youtube-transcript-api.
    Fallback: Downloads audio stream with yt-dlp and transcribes using faster-whisper or whisper.

    Returns:
        Tuple of (transcript_segments_list, method_used_string)
    """
    video_id = extract_youtube_id(video_url)
    segments: List[Dict[str, Any]] = []

    # Method 1: Try youtube-transcript-api first
    if HAS_YT_TRANSCRIPT and video_id:
        if progress_callback:
            progress_callback(0.42, "Checking for official YouTube subtitles & auto-captions...")

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            target_transcript = None

            # 1. Prefer manual English
            try:
                target_transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
            except Exception:
                pass

            # 2. Next, auto-generated English
            if not target_transcript:
                try:
                    target_transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                except Exception:
                    pass

            # 3. Otherwise, get first available and translate to English
            if not target_transcript:
                try:
                    for t in transcript_list:
                        target_transcript = t.translate('en') if t.is_translatable else t
                        break
                except Exception:
                    pass

            if target_transcript:
                raw_data = target_transcript.fetch()
                for item in raw_data:
                    start_t = float(item.get("start", 0.0))
                    duration = float(item.get("duration", 0.0))
                    text = item.get("text", "").replace("\n", " ").strip()
                    if text:
                        segments.append({
                            "text": text,
                            "start": start_t,
                            "duration": duration,
                            "end": start_t + duration
                        })

                if segments:
                    return segments, "YouTube Captions API"
        except Exception as e:
            # Captions disabled or unavailable, move to fallback
            pass

    # If video is already a local video file (e.g. uploaded video or local file), transcribe directly
    if os.path.exists(video_url):
        if progress_callback:
            progress_callback(0.45, "Transcribing local audio with Whisper model...")
        if HAS_FASTER_WHISPER:
            try:
                model = WhisperModel("base", device="cpu", compute_type="int8")
                whisper_segments, _ = model.transcribe(video_url, beam_size=3)
                for s in whisper_segments:
                    txt = s.text.strip()
                    if txt:
                        segments.append({
                            "text": txt,
                            "start": float(s.start),
                            "duration": float(s.end - s.start),
                            "end": float(s.end)
                        })
                return segments, "Local Faster-Whisper (Direct Audio)"
            except Exception:
                pass
        if HAS_OPENAI_WHISPER:
            try:
                model = whisper.load_model("base")
                result = model.transcribe(video_url)
                for s in result.get("segments", []):
                    txt = s.get("text", "").strip()
                    if txt:
                        start_t = float(s.get("start", 0.0))
                        end_t = float(s.get("end", 0.0))
                        segments.append({
                            "text": txt,
                            "start": start_t,
                            "duration": end_t - start_t,
                            "end": end_t
                        })
                return segments, "Local OpenAI Whisper (Direct Audio)"
            except Exception:
                pass
        return [], "Spoken Audio Unavailable (Whisper model required for local video)"

    # Method 2: Local Whisper Audio Fallback with multi-client retry
    if progress_callback:
        progress_callback(0.45, "Captions not found. Downloading audio stream for Whisper fallback...")

    if not HAS_YTDLP:
        return [], "No Transcript Available (yt-dlp missing)"

    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, "lecture_audio.m4a")

    try:
        audio_download_success = False

        # Optimization: Reuse already downloaded source video if available from extract_slides()
        existing_video = os.path.join(output_dir, "source_lecture.mp4")
        if os.path.exists(existing_video) and os.path.getsize(existing_video) > 1024:
            audio_path = existing_video
            audio_download_success = True
        else:
            for client in ["android", "ios", "mweb"]:
                try:
                    ydl_audio_opts = get_yt_dlp_options(audio_path, is_audio=True, client=client)
                    with yt_dlp.YoutubeDL(ydl_audio_opts) as ydl:
                        ydl.download([video_url])
                    audio_download_success = True
                    break
                except Exception:
                    if os.path.exists(audio_path):
                        try:
                            os.remove(audio_path)
                        except Exception:
                            pass
                    continue

            if not os.path.exists(audio_path):
                downloaded = list(Path(audio_dir).glob("lecture_audio*"))
                if downloaded:
                    audio_path = str(downloaded[0])
                    audio_download_success = True

        if not audio_download_success or not os.path.exists(audio_path):
            return [], "Captions Unavailable (YouTube cloud stream restricted)"

        if progress_callback:
            progress_callback(0.50, "Transcribing spoken audio with local Whisper model...")

        # Try faster-whisper first
        if HAS_FASTER_WHISPER:
            model = WhisperModel("base", device="cpu", compute_type="int8")
            whisper_segments, _ = model.transcribe(audio_path, beam_size=3)
            for s in whisper_segments:
                txt = s.text.strip()
                if txt:
                    segments.append({
                        "text": txt,
                        "start": float(s.start),
                        "duration": float(s.end - s.start),
                        "end": float(s.end)
                    })
            return segments, "Local Faster-Whisper (Audio Fallback)"

        # Try standard openai-whisper
        if HAS_OPENAI_WHISPER:
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            for s in result.get("segments", []):
                txt = s.get("text", "").strip()
                if txt:
                    start_t = float(s.get("start", 0.0))
                    end_t = float(s.get("end", 0.0))
                    segments.append({
                        "text": txt,
                        "start": start_t,
                        "duration": end_t - start_t,
                        "end": end_t
                    })
            return segments, "Local OpenAI Whisper (Audio Fallback)"

        return [], "Captions Unavailable (Whisper not installed for audio transcription)"

    except Exception as e:
        return [], f"Audio Fallback Error: {str(e)[:80]}"


# ---------------------------------------------------------------------------
# Core Modular Function 3: Slide Optical Character Recognition (run_ocr)
# ---------------------------------------------------------------------------
def run_ocr(
    slides_data: List[Dict[str, Any]],
    tesseract_cmd: Optional[str] = None,
    ocr_engine: str = "pytesseract",
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Executes OCR on each extracted slide image.
    Extracts text, bullet points, code, and headers, and determines a probable slide title.

    Returns:
        Enriched slides_data with 'ocr_text' and 'slide_title'.
    """
    detected_tesseract = detect_tesseract_path(tesseract_cmd)
    if HAS_PYTESSERACT and detected_tesseract:
        pytesseract.pytesseract.tesseract_cmd = detected_tesseract

    easyocr_reader = None
    if ocr_engine == "easyocr" and HAS_EASYOCR:
        try:
            easyocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception:
            pass

    total_slides = len(slides_data)
    for idx, slide in enumerate(slides_data):
        img_path = slide["image_path"]
        raw_text = ""

        if progress_callback:
            pct = 0.55 + (idx / total_slides) * 0.15
            progress_callback(pct, f"Running OCR on Slide {idx+1}/{total_slides}...")

        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                # Preprocessing for presentation slide text
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # If dark mode slide (mean brightness < 128), invert so text is dark on light
                if np.mean(gray) < 120:
                    gray = cv2.bitwise_not(gray)

                # Mild contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # Attempt pytesseract
                if HAS_PYTESSERACT and detected_tesseract and ocr_engine == "pytesseract":
                    try:
                        raw_text = pytesseract.image_to_string(enhanced, config="--psm 3 --oem 3")
                    except Exception as e:
                        raw_text = f"[OCR Error: {str(e)}]"

                # Attempt easyocr
                elif easyocr_reader and ocr_engine == "easyocr":
                    try:
                        results = easyocr_reader.readtext(enhanced, detail=0)
                        raw_text = "\n".join(results)
                    except Exception as e:
                        raw_text = f"[EasyOCR Error: {str(e)}]"
                else:
                    raw_text = "[Tesseract OCR executable not found. Install Tesseract OCR to extract on-slide text]"

        cleaned_text = "\n".join([line.strip() for line in raw_text.splitlines() if line.strip()])
        slide["ocr_text"] = cleaned_text or "(No text detected on this slide)"

        # Extract probable slide title from the first prominent non-empty line
        title = f"Slide {slide['slide_index']}"
        lines = [line for line in cleaned_text.splitlines() if len(line) > 3]
        if lines:
            first_line = lines[0].strip(" -#*:•")
            if len(first_line) > 3 and len(first_line) < 80:
                title = first_line

        slide["slide_title"] = title

    return slides_data


# ---------------------------------------------------------------------------
# Alignment Helper: Synchronize Slide Timestamps with Audio Transcripts
# ---------------------------------------------------------------------------
def align_transcript_with_slides(
    slides_data: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Maps each spoken transcript segment to its corresponding slide interval [T_k, T_{k+1}).
    Attaches 'verbal_explanation' to each slide dictionary.
    """
    for i, slide in enumerate(slides_data):
        start_t = slide["timestamp"]
        end_t = slide.get("end_time", start_t + 60.0)

        # Collect transcript snippets falling into this slide's timeframe
        matching_texts = []
        for seg in transcript_segments:
            seg_start = seg["start"]
            # Segment belongs to this slide if its start timestamp falls within this interval
            if start_t <= seg_start < end_t:
                matching_texts.append(seg["text"])

        # If strict window returned empty, check with small 2-second tolerance
        if not matching_texts:
            for seg in transcript_segments:
                seg_start = seg["start"]
                if (start_t - 2.0) <= seg_start <= (end_t + 2.0):
                    matching_texts.append(seg["text"])

        verbal_text = " ".join(matching_texts).strip()
        slide["verbal_explanation"] = verbal_text if verbal_text else "(No spoken audio detected for this slide)"

    return slides_data


# ---------------------------------------------------------------------------
# Core Modular Function 4: LLM Note Synthesis (synthesize_notes)
# ---------------------------------------------------------------------------
def synthesize_notes(
    slides_data: List[Dict[str, Any]],
    transcript_data: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    provider: str = "gemini",
    model_name: str = "gemini-2.5-flash",
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Synthesizes slide visual content (OCR) with spoken transcript for each slide.
    Calls Google Gemini or OpenAI API to produce academic, publication-ready lecture notes.
    """
    # First, ensure timestamp alignment is complete
    slides_data = align_transcript_with_slides(slides_data, transcript_data)

    total_slides = len(slides_data)

    # Initialize client if API key is provided
    gemini_client = None
    openai_client = None

    if api_key and api_key.strip():
        if provider == "gemini":
            if HAS_GOOGLE_GENAI_NEW:
                try:
                    gemini_client = genai.Client(api_key=api_key.strip())
                except Exception:
                    pass
            elif HAS_GOOGLE_GENAI_LEGACY:
                try:
                    legacy_genai.configure(api_key=api_key.strip())
                except Exception:
                    pass
        elif provider == "openai" and HAS_OPENAI:
            try:
                openai_client = OpenAI(api_key=api_key.strip())
            except Exception:
                pass

    for idx, slide in enumerate(slides_data):
        if progress_callback:
            pct = 0.70 + (idx / total_slides) * 0.18
            progress_callback(pct, f"Synthesizing Notes for Slide {idx+1}/{total_slides}...")

        slide_num = slide["slide_index"]
        timestamp_str = slide["timestamp_str"]
        slide_title = slide["slide_title"]
        ocr_text = slide["ocr_text"]
        verbal = slide["verbal_explanation"]

        prompt = f"""You are an elite academic scribe and university teaching assistant.
Your task is to synthesize what is written on the slide (OCR) with what the teacher explained verbally (transcript) at this timestamp.

Slide #{slide_num}: "{slide_title}" (Timestamp: {timestamp_str})

[On-Slide Text (OCR)]
{ocr_text}

[Teacher's Spoken Audio (Transcript)]
{verbal}

Deliverable:
Synthesize what is written on the slide (OCR) with what the teacher explained verbally (transcript) at this timestamp. Create structured lecture notes including:
1. **Key Concepts**: Core terminology, definitions, and main themes.
2. **Detailed Explanations**: Deep-dive synthesis combining visual bullets with the teacher's oral elaboration, analogies, and examples.
3. **Code/Formulas**: Any code snippets, mathematical formulas, algorithms, or equations mentioned or displayed. If none, summarize key structural steps.
4. **Lecturer's Remarks**: Important tips, exam hints, practical caveats, or intuition emphasized verbally by the speaker.

Formatting rules:
- Format cleanly in Markdown with bold section headers and clean bullet points.
- Be thorough, academically precise, and clear."""

        notes_result = ""

        # Google Gemini Call
        if provider == "gemini" and api_key:
            try:
                if gemini_client and HAS_GOOGLE_GENAI_NEW:
                    resp = gemini_client.models.generate_content(
                        model=model_name or "gemini-2.5-flash",
                        contents=prompt
                    )
                    notes_result = resp.text
                elif HAS_GOOGLE_GENAI_LEGACY:
                    m = legacy_genai.GenerativeModel(model_name or "gemini-1.5-flash")
                    resp = m.generate_content(prompt)
                    notes_result = resp.text
            except Exception as e:
                notes_result = f"*(LLM API Notice: {str(e)[:120]})*\n\n"

        # OpenAI Call
        elif provider == "openai" and openai_client:
            try:
                resp = openai_client.chat.completions.create(
                    model=model_name or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional university scribe synthesizing lecture slides and verbal audio."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                notes_result = resp.choices[0].message.content
            except Exception as e:
                notes_result = f"*(OpenAI API Notice: {str(e)[:120]})*\n\n"

        # Offline / Simulated Synthesis Fallback (Guarantees app works even without active API keys)
        if not notes_result:
            notes_result = f"""### Key Concepts
- **Core Subject**: {slide_title}
- **Visual Points**: {', '.join([l.strip() for l in ocr_text.splitlines()[:3] if l.strip()]) or 'Fundamental lecture progression'}

### Detailed Explanations
The instructor addressed this slide at timestamp **{timestamp_str}**. 
Visual content on the slide outlines:
{ocr_text[:350]}...

The lecturer explained:
> "{verbal[:400]}..."

### Code/Formulas
- *Formulas/Code Structure*: Synthesized from on-screen diagrams and verbal definitions. Key principles apply directly to the topic context.

### Lecturer's Remarks
- Pay close attention to how this slide sets the foundation for subsequent concepts.
- Review the relationship between the visual bullets and the verbal explanation above."""

        slide["synthesized_notes"] = notes_result.strip()

    return slides_data


# ---------------------------------------------------------------------------
# ReportLab Canvas for Dynamic "Page X of Y" Numbering & Running Headers
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that dynamically calculates the total page count
    and renders professional running headers and footers on every page after the cover.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self.doc_title = "Lecture Notes"

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Do not print header/footer on cover page (Page 1)
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))

            # Running Header
            header_title = (self.doc_title[:65] + "...") if len(self.doc_title) > 65 else self.doc_title
            self.drawString(54, self._pagesize[1] - 36, header_title)
            self.drawRightString(self._pagesize[0] - 54, self._pagesize[1] - 36, "AI Lecture Synthesizer")

            # Header separator line
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, self._pagesize[1] - 42, self._pagesize[0] - 54, self._pagesize[1] - 42)

            # Running Footer
            self.line(54, 45, self._pagesize[0] - 54, 45)
            self.drawString(54, 32, "Confidential / Educational Use — Synthesized from Slide OCR & Audio Transcript")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(self._pagesize[0] - 54, 32, page_text)

            self.restoreState()


# ---------------------------------------------------------------------------
# Core Modular Function 5: Publication-Quality PDF Compilation (generate_pdf)
# ---------------------------------------------------------------------------
def generate_pdf(
    slides_data: List[Dict[str, Any]],
    video_meta: Dict[str, Any],
    output_path: str,
    page_format: str = "A4",
    progress_callback=None
) -> str:
    """
    Compiles lecture notes into a downloadable, publication-quality PDF using ReportLab.
    Features a Title Page, Table of Contents, High-Res Slide Images, Verbatim Slide OCR text,
    and Synthesized Teacher's Notes with clean typography.
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab is required for PDF generation. Run: pip install reportlab")

    if progress_callback:
        progress_callback(0.90, "Compiling publication-quality PDF with ReportLab...")

    page_size = A4 if page_format.upper() == "A4" else letter
    page_width, page_height = page_size
    usable_width = page_width - 108  # 54pt (0.75 in) margins on each side

    doc = SimpleDocTemplate(
        output_path,
        pagesize=page_size,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Clean Palette
    COLOR_PRIMARY = colors.HexColor("#0F172A")    # Deep Slate
    COLOR_BRAND = colors.HexColor("#1D4ED8")      # Blue
    COLOR_ACCENT = colors.HexColor("#0D9488")     # Teal
    COLOR_BG_CARD = colors.HexColor("#F8FAFC")    # Very light slate
    COLOR_BORDER = colors.HexColor("#E2E8F0")     # Light gray border
    COLOR_TEXT = colors.HexColor("#1E293B")       # Dark charcoal
    COLOR_MUTED = colors.HexColor("#64748B")      # Muted slate

    # Typographic Styles
    style_cover_super = ParagraphStyle(
        "CoverSuper",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=COLOR_BRAND,
        spaceAfter=12
    )

    style_cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=COLOR_PRIMARY,
        alignment=0,
        spaceAfter=14
    )

    style_cover_sub = ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=COLOR_MUTED,
        spaceAfter=24
    )

    style_section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=COLOR_PRIMARY,
        spaceBefore=14,
        spaceAfter=8
    )

    style_slide_banner = ParagraphStyle(
        "SlideBanner",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=COLOR_PRIMARY,
        spaceAfter=6
    )

    style_subheading = ParagraphStyle(
        "SubHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=COLOR_BRAND,
        spaceBefore=8,
        spaceAfter=4
    )

    style_body = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=COLOR_TEXT,
        spaceAfter=6
    )

    style_ocr_code = ParagraphStyle(
        "OCRCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # =======================================================================
    # 1. TITLE / COVER PAGE
    # =======================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("LECTURE SUMMARY &amp; MULTIMODAL SLIDE SYNTHESIS", style_cover_super))
    safe_video_title = escape_xml(video_meta.get("title", "Lecture Notes"))
    safe_uploader = escape_xml(video_meta.get("uploader", "Lecturer"))
    story.append(Paragraph(safe_video_title, style_cover_title))
    story.append(Paragraph(f"Presented by: <b>{safe_uploader}</b>", style_cover_sub))
    story.append(HRFlowable(width="100%", thickness=2, color=COLOR_BRAND, spaceAfter=24))

    # Metadata Summary Box
    duration_str = format_timestamp(video_meta.get("duration", 0))
    safe_url = escape_xml(video_meta.get('source_url', '#'))
    meta_table_data = [
        [
            Paragraph("<b>Lecture Duration:</b>", style_body),
            Paragraph(escape_xml(duration_str), style_body),
            Paragraph("<b>Total Slides Detected:</b>", style_body),
            Paragraph(str(len(slides_data)), style_body),
        ],
        [
            Paragraph("<b>Compilation Date:</b>", style_body),
            Paragraph(datetime.now().strftime("%B %d, %Y"), style_body),
            Paragraph("<b>Source URL:</b>", style_body),
            Paragraph(f"<a href='{safe_url}'>YouTube Link</a>", style_body),
        ]
    ]

    meta_table = Table(meta_table_data, colWidths=[100, usable_width/2 - 100, 110, usable_width/2 - 110])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_CARD),
        ('BOX', (0, 0), (-1, -1), 0.75, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)

    story.append(Spacer(1, 24))
    overview_text = (
        "This document integrates on-screen presentation slide keyframes extracted via Computer Vision (SSIM), "
        "optical character recognition (OCR) of slide texts and equations, and synthesized verbal explanations "
        "transcribed from the lecture audio stream. Each section matches the teacher's verbal delivery "
        "to the corresponding visual slide."
    )
    story.append(Paragraph(overview_text, style_body))
    story.append(PageBreak())

    # =======================================================================
    # 2. TABLE OF CONTENTS
    # =======================================================================
    story.append(Paragraph("Table of Contents", style_section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_BORDER, spaceAfter=14))

    toc_data = [[
        Paragraph("<b>Slide #</b>", style_body),
        Paragraph("<b>Timestamp</b>", style_body),
        Paragraph("<b>Slide Subject / Extracted Title</b>", style_body)
    ]]

    for slide in slides_data:
        title_snippet = slide["slide_title"]
        if len(title_snippet) > 60:
            title_snippet = title_snippet[:57] + "..."
        toc_data.append([
            Paragraph(f"Slide {slide['slide_index']}", style_body),
            Paragraph(f"[{escape_xml(slide['timestamp_str'])}]", style_body),
            Paragraph(escape_xml(title_snippet), style_body)
        ])

    toc_table = Table(toc_data, colWidths=[65, 85, usable_width - 150])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('LINEBELOW', (0, 0), (-1, 0), 1, COLOR_BRAND),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_BG_CARD]),
        ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # =======================================================================
    # 3. INDIVIDUAL SLIDE MODULES
    # =======================================================================
    for slide in slides_data:
        slide_flowables = []

        slide_idx = slide["slide_index"]
        t_str = slide["timestamp_str"]
        slide_title = slide["slide_title"]

        # Slide Header Banner
        header_text = f"Slide {slide_idx:02d} — [{escape_xml(t_str)}] : {escape_xml(slide_title)}"
        slide_flowables.append(Paragraph(header_text, style_slide_banner))
        slide_flowables.append(HRFlowable(width="100%", thickness=1, color=COLOR_BRAND, spaceAfter=8))

        # Slide Image
        img_path = slide["image_path"]
        if os.path.exists(img_path):
            try:
                # Maintain aspect ratio (assume 16:9 or measure image)
                with Image.open(img_path) as pil_img:
                    orig_w, orig_h = pil_img.size
                aspect = orig_h / orig_w
                target_w = usable_width
                target_h = usable_width * aspect

                # Cap height so notes have ample room on the page
                if target_h > 210:
                    target_h = 210
                    target_w = target_h / aspect

                rl_img = RLImage(img_path, width=target_w, height=target_h)
                img_table = Table([[rl_img]], colWidths=[usable_width])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#000000")),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ]))
                slide_flowables.append(img_table)
            except Exception as e:
                slide_flowables.append(Paragraph(f"[Image Render Error: {escape_xml(str(e))}]", style_body))

        slide_flowables.append(Spacer(1, 8))

        # Slide Text (OCR Box)
        slide_flowables.append(Paragraph("<b>Slide Text (OCR Extracted):</b>", style_subheading))
        ocr_snippet = slide.get("ocr_text", "(No text detected)").strip()
        # Truncate if unreasonably massive
        if len(ocr_snippet) > 800:
            ocr_snippet = ocr_snippet[:797] + "..."
        safe_ocr = escape_xml(ocr_snippet).replace("\n", "<br/>")

        ocr_table = Table([[Paragraph(safe_ocr, style_ocr_code)]], colWidths=[usable_width])
        ocr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        slide_flowables.append(ocr_table)

        slide_flowables.append(Spacer(1, 8))

        # Teacher's Explanation & Synthesized Notes
        slide_flowables.append(Paragraph("<b>Teacher's Explanation &amp; Synthesized Notes:</b>", style_subheading))

        notes_markdown = slide.get("synthesized_notes", "").strip()
        # Convert markdown headers/bullets into ReportLab styled paragraphs
        for line in notes_markdown.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("### "):
                header_title = format_markdown_for_reportlab(line[4:].strip())
                slide_flowables.append(Paragraph(f"<b>{header_title}</b>", style_subheading))
            elif line.startswith("## "):
                header_title = format_markdown_for_reportlab(line[3:].strip())
                slide_flowables.append(Paragraph(f"<b>{header_title}</b>", style_subheading))
            elif line.startswith("# "):
                header_title = format_markdown_for_reportlab(line[2:].strip())
                slide_flowables.append(Paragraph(f"<b>{header_title}</b>", style_section_heading))
            elif line.startswith("- ") or line.startswith("* "):
                bullet_content = format_markdown_for_reportlab(line[2:].strip())
                slide_flowables.append(Paragraph(f"• {bullet_content}", style_body))
            elif re.match(r'^\d+\.\s+', line):
                num_content = format_markdown_for_reportlab(re.sub(r'^\d+\.\s+', '', line))
                slide_flowables.append(Paragraph(f"• {num_content}", style_body))
            else:
                clean_line = format_markdown_for_reportlab(line)
                slide_flowables.append(Paragraph(clean_line, style_body))

        # Append slide module to story with a pagebreak
        story.extend(slide_flowables)
        story.append(PageBreak())

    # Build document using NumberedCanvas
    def canvas_maker(*args, **kwargs):
        c = NumberedCanvas(*args, **kwargs)
        c.doc_title = video_meta.get("title", "Lecture Notes")
        return c

    doc.build(story, canvasmaker=canvas_maker)

    if progress_callback:
        progress_callback(1.0, "PDF document compilation completed successfully!")

    return output_path


# ---------------------------------------------------------------------------
# Streamlit Application User Interface
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="YouTube Lecture Synthesizer",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Clean Styling
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .status-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .badge {
            background-color: #EEF2FF;
            color: #4338CA;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🎓 YouTube Lecture Synthesizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Extract presentation slides with Computer Vision, transcribe spoken audio, '
        'read slide text via OCR, and synthesize comprehensive lecture notes into a publication-ready PDF.</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------------------------
    # Sidebar: Configurations
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🤖 LLM Note Synthesis")
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["gemini", "openai", "offline"],
            index=0,
            help="Select provider for multimodal lecture synthesis. 'offline' generates structured notes without API calls."
        )

        api_key = ""
        model_name = "gemini-2.5-flash"

        if llm_provider == "gemini":
            env_key = os.environ.get("GEMINI_API_KEY", "")
            api_key = st.text_input(
                "Gemini API Key",
                value=env_key,
                type="password",
                help="Google Gemini API Key. Can also be set in GEMINI_API_KEY env variable."
            )
            model_name = st.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
        elif llm_provider == "openai":
            env_key = os.environ.get("OPENAI_API_KEY", "")
            api_key = st.text_input(
                "OpenAI API Key",
                value=env_key,
                type="password",
                help="OpenAI API Key. Can also be set in OPENAI_API_KEY env variable."
            )
            model_name = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o"])
        else:
            st.info("Offline mode: Synthesizes structured notes from OCR & transcripts using built-in templates (no API key required).")

        st.divider()

        st.subheader("👁️ Slide Detection (CV)")
        sample_interval = st.slider(
            "Frame Check Interval (seconds)",
            min_value=1.0,
            max_value=6.0,
            value=2.5,
            step=0.5,
            help="How often video frames are sampled for slide transitions."
        )

        ssim_threshold = st.slider(
            "Transition Sensitivity (SSIM Threshold)",
            min_value=0.50,
            max_value=0.95,
            value=0.82,
            step=0.02,
            help="Lower values trigger fewer slide changes. 0.80 - 0.85 is ideal for presentation slides."
        )

        min_slide_duration = st.slider(
            "Minimum Slide Duration (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Debounce interval to eliminate rapid animations and speaker movements."
        )

        st.divider()

        st.subheader("🔤 Slide OCR & PDF")
        ocr_engine = st.selectbox("OCR Engine", ["pytesseract", "easyocr"], index=0)
        custom_tesseract_path = st.text_input(
            "Custom Tesseract Path (optional)",
            value=detect_tesseract_path() or "",
            help="Auto-detected if installed in standard paths."
        )

        pdf_page_format = st.selectbox("PDF Page Format", ["A4", "Letter"], index=0)

    # -----------------------------------------------------------------------
    # Main Panel: Input & Trigger
    # -----------------------------------------------------------------------
    input_tab1, input_tab2 = st.tabs(["🔗 YouTube Lecture URL", "📁 Upload Lecture Video File"])

    source_to_process = None
    is_uploaded = False
    should_run = False

    with input_tab1:
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            video_url = st.text_input(
                "Enter Public YouTube Lecture URL:",
                value="",
                placeholder="https://www.youtube.com/watch?v=..."
            )
        with col_btn:
            st.write("")
            st.write("")
            process_yt_clicked = st.button("🚀 Process YouTube", type="primary", use_container_width=True, key="btn_yt")

        # Quick Samples
        st.caption("Try sample lecture links:")
        c1, c2, c3 = st.columns(3)
        if c1.button("📌 CS50 Lecture Sample"):
            st.session_state["sample_url"] = "https://www.youtube.com/watch?v=e9Eds2HPxLE"
        if c2.button("📌 Neural Networks 3B1B"):
            st.session_state["sample_url"] = "https://www.youtube.com/watch?v=aircAruvnKk"
        if c3.button("📌 MIT AI Lecture"):
            st.session_state["sample_url"] = "https://www.youtube.com/watch?v=T3Fc7GS3ne4"

        if "sample_url" in st.session_state and not video_url:
            video_url = st.session_state["sample_url"]

    with input_tab2:
        st.markdown(
            "**Bypass Cloud YouTube Rate Limits:** If YouTube restricts downloads from cloud hosting providers "
            "(HTTP 403 Forbidden), you can directly upload your `.mp4`, `.mov`, `.webm`, or `.mkv` lecture video here."
        )
        col_up, col_up_btn = st.columns([4, 1])
        with col_up:
            uploaded_file = st.file_uploader("Choose video file", type=["mp4", "mov", "mkv", "webm"], key="video_uploader")
        with col_up_btn:
            st.write("")
            st.write("")
            process_upload_clicked = st.button("🚀 Process Upload", type="primary", use_container_width=True, key="btn_upload")

    # Session State Initialization for Results
    if "results" not in st.session_state:
        st.session_state["results"] = None

    # Determine triggering action
    if process_yt_clicked:
        if not video_url or not video_url.strip():
            st.error("Please enter a valid YouTube video URL.")
        elif not extract_youtube_id(video_url.strip()):
            st.error("Invalid YouTube URL. Please provide a standard YouTube video link.")
        else:
            should_run = True
            source_to_process = video_url.strip()

    elif process_upload_clicked:
        if not uploaded_file:
            st.error("Please select a video file to upload.")
        else:
            should_run = True
            is_uploaded = True

    # -----------------------------------------------------------------------
    # Execution Pipeline
    # -----------------------------------------------------------------------
    if should_run:
        # Prepare workspace directory
        temp_dir = tempfile.mkdtemp(prefix="lecture_synth_")
        pdf_output_path = os.path.join(temp_dir, "lecture_notes.pdf")

        if is_uploaded:
            local_upload_path = os.path.join(temp_dir, uploaded_file.name)
            with open(local_upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source_to_process = local_upload_path

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, message: str):
            progress_bar.progress(min(1.0, max(0.0, pct)))
            status_text.markdown(f"**Step Progress:** {message}")

        try:
            # 1. Slide Extraction
            update_progress(0.05, "Extracting unique slides with Computer Vision...")
            slides_data, video_meta = extract_slides(
                video_path_or_url=source_to_process,
                output_dir=temp_dir,
                sample_interval=sample_interval,
                ssim_threshold=ssim_threshold,
                min_slide_duration=min_slide_duration,
                progress_callback=update_progress
            )

            if not slides_data:
                st.warning("No slides could be detected. Please check the video or adjust sensitivity.")
                return

            # 2. Audio & Transcript Extraction
            update_progress(0.40, "Retrieving spoken transcript...")
            transcript_segments, transcript_method = get_transcript(
                video_url=source_to_process,
                output_dir=temp_dir,
                progress_callback=update_progress
            )

            # 3. Slide OCR
            update_progress(0.55, "Running OCR on slide keyframes...")
            slides_data = run_ocr(
                slides_data=slides_data,
                tesseract_cmd=custom_tesseract_path,
                ocr_engine=ocr_engine,
                progress_callback=update_progress
            )

            # 4. LLM Synthesis
            update_progress(0.70, "Synthesizing multimodal lecture notes...")
            slides_data = synthesize_notes(
                slides_data=slides_data,
                transcript_data=transcript_segments,
                api_key=api_key,
                provider=llm_provider,
                model_name=model_name,
                progress_callback=update_progress
            )

            # 5. PDF Generation
            update_progress(0.90, "Compiling ReportLab PDF document...")
            final_pdf_path = generate_pdf(
                slides_data=slides_data,
                video_meta=video_meta,
                output_path=pdf_output_path,
                page_format=pdf_page_format,
                progress_callback=update_progress
            )

            # Read PDF bytes for persistent download
            with open(final_pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.session_state["results"] = {
                "slides_data": slides_data,
                "video_meta": video_meta,
                "transcript_segments": transcript_segments,
                "transcript_method": transcript_method,
                "pdf_bytes": pdf_bytes,
                "pdf_filename": f"{re.sub(r'[^a-zA-Z0-9_-]', '_', video_meta.get('title', 'lecture'))[:40]}_notes.pdf"
            }

            update_progress(1.0, "All processing completed successfully!")
            st.success("🎉 Lecture notes generated successfully!")

        except Exception as e:
            st.error(f"Error during lecture synthesis: {str(e)}")
            st.exception(e)
            return

    # -----------------------------------------------------------------------
    # Results & Side-by-Side Preview Section
    # -----------------------------------------------------------------------
    if st.session_state["results"]:
        results = st.session_state["results"]
        slides_data = results["slides_data"]
        video_meta = results["video_meta"]
        pdf_bytes = results["pdf_bytes"]
        pdf_filename = results["pdf_filename"]
        transcript_method = results["transcript_method"]

        st.divider()

        # Overview & Prominent Download Button
        c_meta, c_dl = st.columns([3, 2])
        with c_meta:
            st.subheader(f"📖 {video_meta.get('title', 'Lecture Notes')}")
            st.write(
                f"**Presenter:** {video_meta.get('uploader')} | "
                f"**Duration:** {format_timestamp(video_meta.get('duration', 0))} | "
                f"**Slides Extracted:** {len(slides_data)} | "
                f"**Transcript Source:** {transcript_method}"
            )
        with c_dl:
            st.download_button(
                label="📥 Download Lecture Notes PDF",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

        st.subheader("📑 Side-by-Side Slide & Synthesis Inspector")

        slide_options = [
            f"Slide {s['slide_index']:02d} [{s['timestamp_str']}] : {s['slide_title'][:40]}"
            for s in slides_data
        ]
        selected_slide_idx = st.selectbox(
            "Jump to Slide:",
            options=range(len(slides_data)),
            format_func=lambda i: slide_options[i]
        )

        cur_slide = slides_data[selected_slide_idx]

        # Side-by-side presentation preview
        col_visual, col_notes = st.columns([1, 1], gap="medium")

        with col_visual:
            st.markdown(f"#### 🖼️ Slide Screenshot `[{cur_slide['timestamp_str']}]`")
            if os.path.exists(cur_slide["image_path"]):
                st.image(cur_slide["image_path"], use_container_width=True)

            with st.expander("🔍 Slide OCR Extracted Text", expanded=True):
                st.code(cur_slide.get("ocr_text", "(No text detected)"), language="text")

        with col_notes:
            st.markdown("#### 📝 Teacher's Explanation & Synthesized Notes")
            st.markdown(cur_slide.get("synthesized_notes", ""))

            with st.expander("🎙️ Spoken Transcript (This Slide's Time Interval)", expanded=False):
                st.info(cur_slide.get("verbal_explanation", "(No spoken words recorded)"))

        # Full Transcript Accordion
        with st.expander("📜 Full Video Transcript with Timestamps"):
            all_transcripts = results.get("transcript_segments", [])
            if all_transcripts:
                for seg in all_transcripts[:150]:
                    st.text(f"[{format_timestamp(seg['start'])}] {seg['text']}")
                if len(all_transcripts) > 150:
                    st.caption(f"... and {len(all_transcripts) - 150} more segments.")
            else:
                st.write("No full transcript segments available.")


if __name__ == "__main__":
    main()