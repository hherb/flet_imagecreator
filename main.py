#!/usr/bin/env python3
"""Entry point for the Flet Image Creator application.

This module provides the main entry point for running the Qwen Image Generator
GUI application. It imports and runs the Flet application from the source module.

Usage:
    python main.py

Or with uv:
    uv run main.py

The application will open a desktop window with the image generation interface.
"""

import flet as ft

from src.qwenimage_fletui import main


def run() -> None:
    """Launch the Flet Image Creator application.

    This function initializes and runs the Flet application with the main
    UI function. It serves as the primary entry point for the application.
    """
    ft.app(target=main)


if __name__ == "__main__":
    run()
