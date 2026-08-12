"""Turn ComfyUI IMAGE / VIDEO inputs into the data URLs an OpenAI-compatible API expects.

Images use the widely supported `image_url` content block. Video uses `video_url`, which is not
part of the OpenAI spec -- the providers that accept video at all use this shape, and the rest
reject the request with a readable error.
"""

from __future__ import annotations

import base64
import io

from .client import OpenAICompatibleError

# av/ffmpeg container names -> mime types. Anything unknown falls back to video/<name>.
_VIDEO_MIME = {
    "mp4": "video/mp4",
    "m4v": "video/mp4",
    "mov": "video/quicktime",
    "quicktime": "video/quicktime",
    "webm": "video/webm",
    "matroska": "video/x-matroska",
    "mkv": "video/x-matroska",
    "avi": "video/x-msvideo",
    "mpegts": "video/mp2t",
}


def _data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def _to_pil(frame):
    """One IMAGE frame ([H, W, C] float 0..1) as a PIL image."""
    import numpy as np
    from PIL import Image

    array = frame.detach().cpu().float().clamp(0.0, 1.0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, "L").convert("RGB")

    channels = array.shape[-1]
    if channels == 1:
        return Image.fromarray(array[..., 0], "L").convert("RGB")
    if channels == 3:
        return Image.fromarray(array, "RGB")
    if channels == 4:
        return Image.fromarray(array, "RGBA")
    raise OpenAICompatibleError(f"Cannot send an image with {channels} channels.")


def image_data_urls(images, image_format: str = "jpeg", max_side: int = 0) -> list[str]:
    """Encode every frame of an IMAGE batch as a data URL, one per frame."""
    if images is None:
        return []
    if getattr(images, "ndim", 0) == 3:  # a single frame without the batch dimension
        images = images.unsqueeze(0)
    if getattr(images, "ndim", 0) != 4:
        raise OpenAICompatibleError(f"Unexpected IMAGE shape: {tuple(getattr(images, 'shape', ()))}")

    fmt = (image_format or "jpeg").lower()
    if fmt not in ("jpeg", "png", "webp"):
        raise OpenAICompatibleError(f"Unsupported image format '{image_format}'.")

    urls = []
    for frame in images:
        picture = _to_pil(frame)
        if max_side and max(picture.size) > max_side:
            from PIL import Image

            picture.thumbnail((max_side, max_side), Image.LANCZOS)
        if fmt == "jpeg" and picture.mode != "RGB":
            picture = picture.convert("RGB")

        buffer = io.BytesIO()
        if fmt == "jpeg":
            picture.save(buffer, format="JPEG", quality=92, optimize=True)
        elif fmt == "png":
            picture.save(buffer, format="PNG", optimize=True)
        else:
            picture.save(buffer, format="WEBP", quality=92)
        urls.append(_data_url(f"image/{fmt}", buffer.getvalue()))
    return urls


def _video_bytes(video) -> bytes:
    source = video.get_stream_source()
    if isinstance(source, str):
        with open(source, "rb") as handle:
            return handle.read()
    source.seek(0)
    return source.read()


def _video_mime(video) -> str:
    try:
        container = (video.get_container_format() or "").split(",")[0].strip().lower()
    except Exception:  # noqa: BLE001 - container detection is best-effort
        container = ""
    if not container:
        return "video/mp4"
    return _VIDEO_MIME.get(container, f"video/{container}")


def video_data_url(video, max_mb: int = 20) -> str | None:
    """Encode a VIDEO as a data URL, refusing sizes the provider will reject anyway."""
    if video is None:
        return None

    payload = _video_bytes(video)
    if not payload:
        raise OpenAICompatibleError("The connected video is empty.")

    limit = max(1, int(max_mb)) * 1024 * 1024
    if len(payload) > limit:
        raise OpenAICompatibleError(
            f"The video is {len(payload) / 1024 / 1024:.1f} MB, over the {max_mb} MB limit. "
            "Trim it, lower the resolution or frame rate, or raise video_max_mb if your provider "
            "allows bigger uploads."
        )
    return _data_url(_video_mime(video), payload)


def content_blocks(text: str, image_urls: list[str], video_url: str | None, image_detail: str = "auto"):
    """Build the `content` value for a user message.

    Returns a plain string when there is no media, so requests to providers that dislike block
    arrays keep working exactly as before.
    """
    if not image_urls and not video_url:
        return text

    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for url in image_urls:
        image_url: dict = {"url": url}
        if image_detail and image_detail != "auto":
            image_url["detail"] = image_detail
        blocks.append({"type": "image_url", "image_url": image_url})
    if video_url:
        blocks.append({"type": "video_url", "video_url": {"url": video_url}})
    return blocks


def describe(image_urls: list[str], video_url: str | None) -> str:
    """Short human-readable summary for the log line."""
    parts = []
    if image_urls:
        size = sum(len(url) for url in image_urls) * 3 // 4
        parts.append(f"{len(image_urls)} image(s), ~{size / 1024:.0f} KB")
    if video_url:
        parts.append(f"1 video, ~{len(video_url) * 3 // 4 / 1024 / 1024:.1f} MB")
    return ", ".join(parts) or "no media"


__all__ = [
    "content_blocks",
    "describe",
    "image_data_urls",
    "video_data_url",
]
