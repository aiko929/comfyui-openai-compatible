"""Chat-completion node for user-supplied OpenAI-compatible endpoints."""

from __future__ import annotations

import logging

from comfy_api.latest import io

from . import media, store
from .client import (
    DEFAULT_BASE_URL,
    MODEL_PLACEHOLDER,
    chat_completion,
)

MAX_TEXT_INPUTS = 16


def _workflow_id(extra_pnginfo) -> str | None:
    """The frontend stamps a stable uuid on each workflow; use it to scope stored answers."""
    if isinstance(extra_pnginfo, dict):
        workflow = extra_pnginfo.get("workflow")
        if isinstance(workflow, dict):
            value = workflow.get("id")
            if isinstance(value, str) and value:
                return value
    return None


def _unescape(value: str) -> str:
    """Let single-line widgets express newlines/tabs as \\n and \\t."""
    return (value or "").replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _ordered_texts(prompts: dict | None) -> list[str]:
    """Autogrow hands us {"text_1": ..., "text_3": ...}; return connected values in slot order."""
    prompts = prompts or {}
    texts = []
    for name in sorted(prompts, key=lambda n: int(n.rsplit("_", 1)[-1])):
        value = prompts[name]
        if value is None:
            continue
        value = str(value).strip()
        if value:
            texts.append(value)
    return texts


class OpenAICompatibleChat(io.ComfyNode):
    """Send one or more text inputs to any OpenAI-compatible /chat/completions endpoint."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpenAICompatibleChat",
            display_name="OpenAI Compatible LLM",
            category="api/text",
            description=(
                "Talk to any OpenAI-compatible endpoint (Mammouth, OpenRouter, Groq, LM Studio, "
                "Ollama, vLLM, ...). Enter the base URL and API key, press 'Refresh models' to "
                "load the model list, then connect as many text inputs as you need."
            ),
            search_aliases=["openai", "llm", "chatgpt", "mammouth", "openrouter", "ollama", "prompt"],
            inputs=[
                io.String.Input(
                    "base_url",
                    default=DEFAULT_BASE_URL,
                    placeholder="https://api.mammouth.ai/v1",
                    tooltip="Endpoint root, i.e. the part before /models and /chat/completions.",
                ),
                io.String.Input(
                    "api_key",
                    default="",
                    placeholder="sk-...",
                    tooltip=(
                        "API key. Careful: it is saved inside the workflow. Leave empty to use the "
                        "OPENAI_COMPATIBLE_API_KEY / OPENAI_API_KEY environment variable, or type "
                        "'env:MY_VARIABLE' to read a specific one."
                    ),
                ),
                io.Combo.Input(
                    "model",
                    options=[MODEL_PLACEHOLDER],
                    default=MODEL_PLACEHOLDER,
                    tooltip="Filled by the 'Refresh models' button from GET {base_url}/models.",
                ),
                io.Autogrow.Input(
                    "prompts",
                    template=io.Autogrow.TemplateNames(
                        io.String.Input("text", multiline=True),
                        names=[f"text_{i}" for i in range(1, MAX_TEXT_INPUTS + 1)],
                        min=1,
                    ),
                    tooltip=(
                        "Text inputs that make up the prompt. A new slot appears every time you "
                        f"connect one, up to {MAX_TEXT_INPUTS}. Empty inputs are skipped."
                    ),
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Optional instructions sent as the system message.",
                ),
                io.Combo.Input(
                    "input_mode",
                    options=["join", "separate_messages"],
                    default="join",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "join: glue all text inputs into a single user message. "
                        "separate_messages: send each text input as its own user message."
                    ),
                ),
                io.String.Input(
                    "separator",
                    default="\\n\\n",
                    optional=True,
                    advanced=True,
                    tooltip="Used by 'join' mode. \\n and \\t are turned into real newlines/tabs.",
                ),
                io.Float.Input(
                    "temperature",
                    default=1.0,
                    min=-1.0,
                    max=2.0,
                    step=0.01,
                    optional=True,
                    advanced=True,
                    tooltip="Sampling temperature. Set to -1 to leave it out of the request.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=0,
                    min=0,
                    max=1000000,
                    optional=True,
                    advanced=True,
                    tooltip="Maximum tokens in the answer. 0 leaves it up to the provider.",
                ),
                io.Int.Input(
                    "timeout",
                    default=180,
                    min=5,
                    max=3600,
                    optional=True,
                    advanced=True,
                    tooltip="Seconds to wait for the response.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Not sent to the API: it only controls whether this node re-runs instead of "
                        "returning its cached answer."
                    ),
                ),
                # NOTE: new widgets must be appended here, at the end. ComfyUI matches saved
                # widget values by position, so inserting one higher up would shift every value
                # below it in workflows that were saved before the change.
                io.Boolean.Input(
                    "reuse_last_result",
                    default=False,
                    label_on="reuse last answer",
                    label_off="generate normally",
                    optional=True,
                    tooltip=(
                        "On: output the answer this node produced last time and do not call the "
                        "API at all, no matter what changed upstream. Off: generate normally. "
                        "The stored answer survives restarts; if there is nothing stored yet, one "
                        "answer is generated and kept."
                    ),
                ),
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip=(
                        "Optional image(s) to look at. Every frame of the batch is sent as its own "
                        "image, attached to the last user message. Needs a vision-capable model."
                    ),
                ),
                io.Video.Input(
                    "video",
                    optional=True,
                    tooltip=(
                        "Optional video, inlined as a video_url block. Support is provider-specific "
                        "(Mammouth documents 20 MB for Gemini); most models reject it."
                    ),
                ),
                io.Combo.Input(
                    "image_detail",
                    options=["auto", "low", "high"],
                    default="auto",
                    optional=True,
                    advanced=True,
                    tooltip="OpenAI 'detail' hint. 'low' is much cheaper, 'high' reads fine print.",
                ),
                io.Combo.Input(
                    "image_format",
                    options=["jpeg", "png", "webp"],
                    default="jpeg",
                    optional=True,
                    advanced=True,
                    tooltip="How images are encoded. png is lossless (better for text/UI screenshots).",
                ),
                io.Int.Input(
                    "image_max_side",
                    default=0,
                    min=0,
                    max=8192,
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Downscale images so the longest side is at most this many pixels, to save "
                        "tokens and upload time. 0 sends them at full size."
                    ),
                ),
                io.Int.Input(
                    "video_max_mb",
                    default=20,
                    min=1,
                    max=500,
                    optional=True,
                    advanced=True,
                    tooltip="Refuse to upload a video larger than this, instead of failing at the provider.",
                ),
            ],
            outputs=[io.String.Output(display_name="text")],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def validate_inputs(cls, model, **kwargs):
        # Defining this disables ComfyUI's built-in combo check, which would reject any model
        # id that isn't in the placeholder list baked into the schema. **kwargs is required:
        # dynamic (Autogrow) inputs are always passed to this function.
        return True

    @classmethod
    async def execute(
        cls,
        base_url: str,
        api_key: str,
        model: str,
        prompts: io.Autogrow.Type = None,
        system_prompt: str = "",
        input_mode: str = "join",
        separator: str = "\\n\\n",
        temperature: float = 1.0,
        max_tokens: int = 0,
        timeout: int = 180,
        seed: int = 0,
        reuse_last_result: bool = False,
        images=None,
        video=None,
        image_detail: str = "auto",
        image_format: str = "jpeg",
        image_max_side: int = 0,
        video_max_mb: int = 20,
    ) -> io.NodeOutput:
        key = store.make_key(_workflow_id(cls.hidden.extra_pnginfo), cls.hidden.unique_id)
        if reuse_last_result:
            stored = store.get(key)
            if stored is not None:
                logging.info("[openai-compatible] reusing the stored answer for node %s", cls.hidden.unique_id)
                return io.NodeOutput(stored)
            logging.info(
                "[openai-compatible] node %s has no stored answer yet; generating one to reuse",
                cls.hidden.unique_id,
            )

        model = (model or "").strip()
        if not model or model == MODEL_PLACEHOLDER:
            raise ValueError(
                "No model selected. Press 'Refresh models' on the node to load the list from "
                f"{base_url.strip() or 'the endpoint'}/models."
            )

        texts = _ordered_texts(prompts)
        image_urls = media.image_data_urls(images, image_format=image_format, max_side=image_max_side)
        video_url = media.video_data_url(video, max_mb=video_max_mb)
        if not texts and not image_urls and not video_url:
            raise ValueError(
                "Nothing to send: connect a non-empty text input, an image, or a video."
            )
        if image_urls or video_url:
            logging.info("[openai-compatible] sending %s to %s", media.describe(image_urls, video_url), model)

        messages: list[dict] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        if input_mode == "separate_messages":
            user_texts = texts or [""]
        else:
            user_texts = [_unescape(separator).join(texts)]
        # Media rides along with the last user message, so it stays next to the text it belongs to.
        for index, text in enumerate(user_texts):
            last = index == len(user_texts) - 1
            content = media.content_blocks(
                text,
                image_urls if last else [],
                video_url if last else None,
                image_detail=image_detail,
            )
            messages.append({"role": "user", "content": content})

        answer = await chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=None if temperature < 0 else temperature,
            max_tokens=max_tokens or None,
            timeout=float(timeout),
        )
        store.put(key, answer)
        return io.NodeOutput(answer)
