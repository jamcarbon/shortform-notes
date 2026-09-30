"""Speech-to-text. Two backends:

* ``openai``: ``gpt-transcribe`` (about $0.0045 per minute of audio). Needs OPENAI_API_KEY.
* ``local``: faster-whisper, on the GPU when ctranslate2 can see one (float16), else the CPU
              (int8). No key and no network after the one-time model download (about 75 MB for
              ``base``). Install with ``pip install "shortform-notes[local]"``.

``SHORTFORM_NOTES_WHISPER_DEVICE`` picks the device: ``auto`` (default), ``cuda`` or ``cpu``.
ctranslate2 is already a faster-whisper dependency, so GPU detection costs no extra install.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
from pathlib import Path

from shortform_notes.config import Settings
from shortform_notes.media import MAX_AUDIO_BYTES

logger = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()  # the web server imports on several threads; load each model once
_DLL_DIRS_ADDED = False


def _add_nvidia_dll_dirs() -> None:
    """Make the pip-installed CUDA runtime (the ``[cuda]`` extra) findable on Windows.

    ctranslate2 ships without cuBLAS / cuDNN. NVIDIA publishes them as wheels
    (``nvidia-cublas-cu12``, ``nvidia-cudnn-cu12``) that unpack their DLLs under
    ``site-packages/nvidia/*/bin``, which is not on the DLL search path. Linux wheels
    are found through the package's RPATH, so this only matters on Windows.
    """
    global _DLL_DIRS_ADDED
    if _DLL_DIRS_ADDED or os.name != "nt":
        return
    _DLL_DIRS_ADDED = True
    try:
        import nvidia  # namespace package from the nvidia-* wheels
    except ImportError:
        return
    for root in getattr(nvidia, "__path__", []):
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            # ctranslate2 loads cuDNN lazily by name, which consults PATH rather than add_dll_directory.
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def cuda_available() -> bool:
    """True when ctranslate2 reports a CUDA device. Never raises: a missing wheel or driver means no."""
    try:
        _add_nvidia_dll_dirs()
        import ctranslate2  # lazy: arrives with faster-whisper

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 (any failure to probe is "no GPU")
        return False


def resolve_device(requested: str) -> tuple[str, str]:
    """(device, compute_type). float16 on a GPU, int8 on the CPU, which is what each is fastest at."""
    device = requested if requested in ("cuda", "cpu") else ("cuda" if cuda_available() else "cpu")
    return device, ("float16" if device == "cuda" else "int8")


@functools.lru_cache(maxsize=4)
def _load_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel  # lazy: optional, heavy

    if device == "cuda":
        _add_nvidia_dll_dirs()
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _run(audio_path: str, model_name: str, device: str, compute_type: str) -> str | None:
    with _MODEL_LOCK:
        model = _load_model(model_name, device, compute_type)
    # No VAD filter: it strips sung and music-backed speech, which is most of short-form video.
    segments, _info = model.transcribe(audio_path)
    # ``segments`` is lazy: decoding (and any CUDA library error) happens while iterating.
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    # Instrumental audio decodes to a run of ". . ." rather than nothing; that is no transcript.
    return text if any(ch.isalnum() for ch in text) else None


def _whisper_sync(audio_path: str, model_name: str, requested_device: str = "auto") -> str | None:
    device, compute_type = resolve_device(requested_device)
    logger.info("local transcription: %s on %s (%s)", model_name, device, compute_type)
    try:
        return _run(audio_path, model_name, device, compute_type)
    except Exception as exc:
        if device != "cuda":
            raise
        # The usual cause is a CUDA runtime library (cuBLAS / cuDNN) missing from the DLL path: the
        # GPU is visible but unusable. A CPU transcript beats no transcript.
        logger.warning("CUDA transcription failed (%s); retrying on the CPU", str(exc).splitlines()[0][:200])
        logger.info("local transcription: %s on cpu (int8)", model_name)
        return _run(audio_path, model_name, "cpu", "int8")


async def _transcribe_local(audio_path: str, settings: Settings) -> str | None:
    return await asyncio.to_thread(_whisper_sync, audio_path, settings.whisper_model, settings.whisper_device)


async def _transcribe_openai(audio_path: str, settings: Settings) -> str | None:
    from openai import AsyncOpenAI  # lazy: optional dependency

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    audio_bytes = await asyncio.to_thread(Path(audio_path).read_bytes)
    response = await asyncio.wait_for(
        client.audio.transcriptions.create(
            model=settings.openai_transcribe_model,
            file=(Path(audio_path).name, audio_bytes),
        ),
        timeout=120,
    )
    return (getattr(response, "text", "") or "").strip() or None


async def transcribe(audio_path: str, settings: Settings) -> str | None:
    """Return the verbatim transcript, or None if the file is too large or the backend returns nothing."""
    size = (await asyncio.to_thread(os.stat, audio_path)).st_size
    if settings.transcribe_provider == "openai" and size > MAX_AUDIO_BYTES:
        logger.warning("audio %s is %d bytes, over the OpenAI upload cap", audio_path, size)
        return None
    if settings.transcribe_provider == "local":
        return await _transcribe_local(audio_path, settings)
    if settings.transcribe_provider == "openai":
        return await _transcribe_openai(audio_path, settings)
    return None
