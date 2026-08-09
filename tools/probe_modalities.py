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
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import OpenAICompatibleError, chat_completion, list_models  # noqa: E402

DEFAULT_BASE_URL = "https://api.mammouth.ai/v1"

# 2x2 red PNG.
IMAGE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGP8z8DAwMDAxAADCAYADiAA/9k5f8gAAAAASUVORK5CYII="
)

QUESTION = "Reply with the single word: ok"


def make_video() -> bytes | None:
    """A minimal mp4, encoded on the fly so no binary blob has to live in the repo."""
    try:
        import io

        import av
    except ImportError:
        return None

    buffer = io.BytesIO()
    try:
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=4)
        stream.width, stream.height = 32, 32
        stream.pix_fmt = "yuv420p"
        for _ in range(4):
            frame = av.VideoFrame(32, 32, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes(len(plane)))
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
        container.close()
    except Exception as error:  # noqa: BLE001 - probing is best-effort
        print(f"  (could not build a probe video: {error})")
        return None
    return buffer.getvalue()


def data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def blocks_for(modality: str, video: bytes | None) -> list[dict] | None:
    if modality == "text":
        return [{"type": "text", "text": QUESTION}]
    if modality == "image":
        return [
            {"type": "text", "text": QUESTION},
            {"type": "image_url", "image_url": {"url": data_url("image/png", IMAGE_PNG)}},
        ]
    if modality == "video" and video:
        return [
            {"type": "text", "text": QUESTION},
            {"type": "video_url", "video_url": {"url": data_url("video/mp4", video)}},
        ]
    return None


async def probe(base_url: str, api_key: str, model: str, modality: str, video: bytes | None, timeout: float):
    content = blocks_for(modality, video)
    if content is None:
        return "skipped", "no probe available"
    try:
        await chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=16,
            timeout=timeout,
        )
        return "yes", ""
    except OpenAICompatibleError as error:
        message = " ".join(str(error).split())
        return "no", message[:200]
    except Exception as error:  # noqa: BLE001
        return "error", f"{type(error).__name__}: {error}"


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

    print(f"Probing {len(models)} model(s) at {args.base_url} for: {', '.join(wanted)}\n")
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results: dict[str, dict] = {}

    async def run(model: str):
        row = {}
        for modality in wanted:
            async with semaphore:
                verdict, note = await probe(args.base_url, args.api_key, model, modality, video, args.timeout)
            row[modality] = {"supported": verdict, "note": note}
        results[model] = row
        summary = "  ".join(f"{m}={row[m]['supported']}" for m in wanted)
        print(f"{model:<45} {summary}")
        for modality in wanted:
            note = row[modality]["note"]
            if note and row[modality]["supported"] != "yes":
                print(f"{'':<45}   {modality}: {note}")

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
