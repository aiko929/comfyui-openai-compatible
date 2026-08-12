"""Ask your provider which of its models actually accept images and video.

The /models endpoint does not report modalities, so this sends a tiny probe request to each model
and records whether it was accepted. Images are a 2x2 PNG, video a fraction-of-a-second black
clip -- a handful of tokens per model, but it is still one real request per model per modality.

Usage (from the package directory):

    python tools/probe_modalities.py --api-key sk-...
    python tools/probe_modalities.py                      # uses OPENAI_COMPATIBLE_API_KEY
    python tools/probe_modalities.py --only gpt,gemini    # just the models whose id matches
    python tools/probe_modalities.py --modalities image   # skip the video probe
    python tools/probe_modalities.py --json report.json   # also write machine-readable results

Nothing is sent anywhere except the endpoint you point it at.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io as io_module
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import OpenAICompatibleError, chat_completion, list_models  # noqa: E402

DEFAULT_BASE_URL = "https://api.openai.com/v1"

QUESTION = "Reply with the single word: ok"
IMAGE_QUESTION = "What word is written in this image? Answer with just that word."

# Errors that say nothing about modality support -- rate limits, capacity, upstream hiccups.
TRANSIENT = ("429", "500", "502", "503", "504", "no deployments available", "timed out", "overloaded")
# Wording providers use when the model itself cannot accept the content.
UNSUPPORTED = (
    "support image input", "support video", "does not support image", "image input",
    "vision", "modality", "multimodal", "invalid content type", "unsupported content",
)


def make_image() -> bytes:
    """Generate a real 256x256 PNG with legible text.

    Deliberately not a hardcoded base64 blob: a 2x2 pixel or subtly malformed image gets
    rejected by providers as a bad *image*, which looks exactly like a missing *capability*.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (256, 256), (30, 90, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle([64, 64, 192, 192], fill=(240, 200, 40))
    draw.text((10, 10), "HELLO", fill=(255, 255, 255))
    buffer = io_module.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_video() -> bytes | None:
    """A minimal mp4, encoded on the fly so no binary blob has to live in the repo."""
    try:
        import av
    except ImportError:
        return None

    try:
        import numpy as np
    except ImportError:
        return None

    size, frames = 64, 12
    buffer = io_module.BytesIO()
    try:
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=6)
        stream.width, stream.height = size, size
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            # A white square sliding left to right, so a model that really sees the video has
            # something to describe -- a static black clip proves much less.
            array = np.zeros((size, size, 3), dtype=np.uint8)
            array[:, :, 2] = 180  # blue background
            left = int(index * (size - 16) / max(1, frames - 1))
            array[24:40, left : left + 16] = 255
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode(None))
        container.close()
    except Exception as error:  # noqa: BLE001 - probing is best-effort
        print(f"  (could not build a probe video: {error})")
        return None
    return buffer.getvalue()


def data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def blocks_for(modality: str, image: bytes | None, video: bytes | None) -> list[dict] | None:
    if modality == "text":
        return [{"type": "text", "text": QUESTION}]
    if modality == "image" and image:
        return [
            {"type": "text", "text": IMAGE_QUESTION},
            {"type": "image_url", "image_url": {"url": data_url("image/png", image)}},
        ]
    if modality == "video" and video:
        return [
            {"type": "text", "text": "Describe this video in one short sentence."},
            {"type": "video_url", "video_url": {"url": data_url("video/mp4", video)}},
        ]
    return None


def classify(message: str) -> str:
    """'no' only when the provider says the content type is the problem."""
    lowered = message.lower()
    if any(hint in lowered for hint in TRANSIENT):
        return "transient"
    if any(hint in lowered for hint in UNSUPPORTED):
        return "no"
    return "error"


async def probe(base_url, api_key, model, modality, image, video, timeout, retries=2):
    """Returns (verdict, note, answer). Retries transient failures before giving a verdict."""
    content = blocks_for(modality, image, video)
    if content is None:
        return "skipped", "no probe available", ""

    note = ""
    for attempt in range(retries + 1):
        try:
            answer = await chat_completion(
                base_url=base_url, api_key=api_key, model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=64, timeout=timeout,
            )
            return "yes", "", " ".join(answer.split())[:100]
        except OpenAICompatibleError as error:
            note = " ".join(str(error).split())
            verdict = classify(note)
            if verdict != "transient":
                return verdict, note[:400], ""
            if attempt < retries:
                await asyncio.sleep(6)
        except Exception as error:  # noqa: BLE001
            return "error", f"{type(error).__name__}: {error}", ""
    return "unknown", f"still failing after {retries + 1} tries: {note[:400]}", ""


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_COMPATIBLE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default="", help="Defaults to OPENAI_COMPATIBLE_API_KEY / OPENAI_API_KEY.")
    parser.add_argument("--only", default="", help="Comma-separated substrings; only matching model ids are probed.")
    parser.add_argument("--modalities", default="image,video", help="Which probes to run (text,image,video).")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--json", default="", help="Write the full result table to this file.")
    args = parser.parse_args()

    wanted = [m.strip() for m in args.modalities.split(",") if m.strip()]
    try:
        models = await list_models(args.base_url, args.api_key, timeout=30.0)
    except OpenAICompatibleError as error:
        print(f"Could not list models: {error}")
        return 1

    if args.only:
        needles = [n.strip().lower() for n in args.only.split(",") if n.strip()]
        models = [m for m in models if any(n in m.lower() for n in needles)]
    if not models:
        print("No models matched.")
        return 1

    video = make_video() if "video" in wanted else None
    if "video" in wanted and video is None:
        print("PyAV unavailable, skipping the video probe.\n")
        wanted = [m for m in wanted if m != "video"]
    image = make_image() if "image" in wanted else None

    print(f"Probing {len(models)} model(s) at {args.base_url} for: {', '.join(wanted)}\n")
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results: dict[str, dict] = {}

    async def one(model: str, modality: str) -> dict:
        async with semaphore:
            verdict, note, answer = await probe(
                args.base_url, args.api_key, model, modality, image, video, args.timeout
            )
        return {"supported": verdict, "note": note, "answer": answer}

    async def run(model: str):
        # Baseline first: if plain text fails, any media verdict for this model is meaningless.
        row = {"text": await one(model, "text")}
        if row["text"]["supported"] != "yes":
            for modality in wanted:
                row[modality] = {"supported": "unknown", "note": "the model failed a text-only request", "answer": ""}
        else:
            for modality in wanted:
                row[modality] = await one(model, modality)
        results[model] = row

        summary = "  ".join(f"{m}={row[m]['supported']}" for m in ["text"] + wanted)
        print(f"{model:<45} {summary}")
        for modality in ["text"] + wanted:
            entry = row[modality]
            if entry["supported"] == "yes" and entry["answer"]:
                print(f"{'':<45}   {modality} replied: {entry['answer']!r}")
            elif entry["note"]:
                print(f"{'':<45}   {modality}: {entry['note']}")

    await asyncio.gather(*(run(model) for model in models))

    print("\n--- summary ---")
    for modality in wanted:
        good = sorted(m for m, row in results.items() if row[modality]["supported"] == "yes")
        print(f"\n{modality}: {len(good)}/{len(results)} accepted")
        for model in good:
            print(f"  {model}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"base_url": args.base_url, "results": results}, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
