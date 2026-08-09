"""HTTP route used by the node's frontend extension to populate the model dropdown."""

from __future__ import annotations

import logging

from aiohttp import web

from .client import OpenAICompatibleError, list_models

logger = logging.getLogger(__name__)

MODELS_ROUTE = "/openai_compatible/models"


async def _models_handler(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Expected a JSON body."}, status=400)

    if not isinstance(body, dict):
        return web.json_response({"error": "Expected a JSON object."}, status=400)

    try:
        models = await list_models(
            base_url=body.get("base_url", ""),
            api_key=body.get("api_key", ""),
            timeout=float(body.get("timeout", 30)),
        )
    except OpenAICompatibleError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:  # unexpected: log it server-side, still answer the UI
        logger.exception("Failed to list models for an OpenAI-compatible endpoint")
        return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    return web.json_response({"models": models})


def register_routes() -> None:
    """Attach the models route to ComfyUI's server, if one is running."""
    try:
        from server import PromptServer
    except ImportError:  # imported outside of a ComfyUI process (tests, tooling)
        return

    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        logger.warning("comfyui-openai-compatible: no PromptServer instance, %s not registered", MODELS_ROUTE)
        return

    instance.routes.post(MODELS_ROUTE)(_models_handler)
