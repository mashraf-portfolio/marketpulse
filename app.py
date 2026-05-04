"""HF Spaces entry point. Reads env vars and launches the Gradio app."""

from __future__ import annotations

import os

from src.ui.gradio_app import build_demo

USE_REMOTE_API = os.getenv("USE_REMOTE_API", "false").lower() == "true"
API_URL = os.getenv("API_URL", "http://localhost:8000")


if __name__ == "__main__":
    demo = build_demo(use_remote_api=USE_REMOTE_API, api_url=API_URL)
    demo.launch(server_name="0.0.0.0", server_port=7860)
