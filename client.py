"""Small async client for any OpenAI-compatible HTTP endpoint.

Shared by the node itself and by the /openai_compatible/models route that the
frontend uses to populate the model dropdown.
"""

from __future__ import annotations

import json
import logging
import os

import aiohttp

DEFAULT_BASE_URL = "https://api.mammouth.ai/v1"
MODEL_PLACEHOLDER = "(press Refresh models)"
ENV_KEY_NAMES = ("OPENAI_COMPATIBLE_API_KEY", "OPENAI_API_KEY")

_MAX_ERROR_CHARS = 800

logger = logging.getLogger(__name__)


class OpenAICompatibleError(RuntimeError):
    """Raised for anything the user can fix: bad url, bad key, API error."""


def normalize_base_url(base_url: str) -> str:
    """Return the endpoint root without a trailing slash, e.g. https://host/v1."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise OpenAICompatibleError("No endpoint URL given. Example: https://api.mammouth.ai/v1")
    if not base.startswith(("http://", "https://")):
        base = "https://" + base
    return base


def resolve_api_key(api_key: str) -> str:
    """Resolve the key from the widget value, an `env:NAME` reference, or the environment."""
    key = (api_key or "").strip()
    if key.lower().startswith("env:"):
        name = key[4:].strip()
        value = os.environ.get(name, "").strip()
        if not value:
            raise OpenAICompatibleError(f"Environment variable '{name}' is not set (or empty).")
        return value
    if key:
        return key
    for name in ENV_KEY_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise OpenAICompatibleError(
        "No API key. Type it into the api_key widget, use 'env:MY_VAR', "
        f"or set one of: {', '.join(ENV_KEY_NAMES)}."
    )


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _shorten(text: str) -> str:
    text = (text or "").strip()
    return text[:_MAX_ERROR_CHARS] + "..." if len(text) > _MAX_ERROR_CHARS else text


def _api_error_message(payload: dict) -> str | None:
    """Some endpoints return an error object with HTTP 200."""
    error = payload.get("error")
    if not error:
        return None
    if isinstance(error, dict):
        return error.get("message") or json.dumps(error)
    return str(error)


async def _request(method: str, url: str, api_key: str, body: dict | None, timeout: float) -> dict:
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    try:
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.request(method, url, headers=_headers(api_key), json=body) as response:
                text = await response.text()
                if response.status >= 400:
                    raise OpenAICompatibleError(
                        f"{method} {url} returned HTTP {response.status}: {_shorten(text)}"
                    )
    except aiohttp.ClientError as exc:
        raise OpenAICompatibleError(f"Could not reach {url}: {exc}") from exc
    except TimeoutError as exc:
        raise OpenAICompatibleError(f"{url} timed out after {timeout:g}s.") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenAICompatibleError(f"{url} did not return JSON: {_shorten(text)}") from exc
    if not isinstance(payload, (dict, list)):
        raise OpenAICompatibleError(f"Unexpected response from {url}: {_shorten(text)}")
    if isinstance(payload, dict):
        message = _api_error_message(payload)
        if message:
            raise OpenAICompatibleError(f"API error from {url}: {message}")
    return payload


def _extract_model_ids(payload) -> list[str]:
    """Accept {"data": [...]}, {"models": [...]} or a bare list, with dict or str items."""
    if isinstance(payload, dict):
        entries = payload.get("data")
        if entries is None:
            entries = payload.get("models")
        if entries is None:
            entries = []
    else:
        entries = payload

    models: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            model_id = entry
        elif isinstance(entry, dict):
            model_id = entry.get("id") or entry.get("name") or entry.get("model")
        else:
            model_id = None
        if model_id:
            models.append(str(model_id))

    return sorted(set(models), key=str.lower)


async def list_models(base_url: str, api_key: str, timeout: float = 30.0) -> list[str]:
    """GET {base_url}/models and return the available model ids."""
    base = normalize_base_url(base_url)
    key = resolve_api_key(api_key)
    payload = await _request("GET", f"{base}/models", key, None, timeout)
    models = _extract_model_ids(payload)
    if not models:
        raise OpenAICompatibleError(f"{base}/models returned no models.")
    return models


def _message_text(message: dict) -> str:
    """Pull text out of a chat message, tolerating list-style content blocks."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                chunks.append(block["text"])
        joined = "".join(chunks)
        if joined.strip():
            return joined
    # Some reasoning models put everything in reasoning_content when they emit no answer.
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


async def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 180.0,
) -> str:
    """POST {base_url}/chat/completions and return the assistant's text."""
    base = normalize_base_url(base_url)
    key = resolve_api_key(api_key)

    body: dict = {"model": model, "messages": messages, "stream": False}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens:
        body["max_tokens"] = max_tokens

    payload = await _request("POST", f"{base}/chat/completions", key, body, timeout)

    choices = payload.get("choices") or []
    if not choices:
        raise OpenAICompatibleError(f"No choices in response: {_shorten(json.dumps(payload))}")
    choice = choices[0]
    if choice.get("finish_reason") == "content_filter":
        raise OpenAICompatibleError("The request was blocked by the provider's content filter.")

    text = _message_text(choice.get("message") or {})
    if not text:
        raise OpenAICompatibleError(
            f"Model '{model}' returned an empty message: {_shorten(json.dumps(payload))}"
        )
    return text
