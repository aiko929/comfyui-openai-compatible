"""ComfyUI nodes for user-supplied OpenAI-compatible endpoints."""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import OpenAICompatibleChat
from .routes import register_routes

WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]


class OpenAICompatibleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [OpenAICompatibleChat]

    @override
    async def on_load(self) -> None:
        register_routes()


async def comfy_entrypoint() -> OpenAICompatibleExtension:
    return OpenAICompatibleExtension()
