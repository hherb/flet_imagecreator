# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Flet-based GUI application for generating images using the Qwen diffusion model. The application provides a user-friendly interface for AI image generation with customizable parameters.

## Architecture

- **main.py**: Simple entry point (currently just prints hello message)
- **src/qwenimage_fletui.py**: Main application containing:
  - `QwenImageGenerator` class: Handles model loading and image generation
  - Flet UI components for user interaction
  - Threading for non-blocking model operations

The app uses a single-file architecture with the main UI and generator logic in `qwenimage_fletui.py`.

## Development Commands

Run the application:
```bash
python src/qwenimage_fletui.py
```

Install dependencies:
```bash
uv sync
```

## Key Technical Details

- **Model**: Uses Qwen/Qwen-Image diffusion pipeline via HuggingFace transformers
- **Device Support**: Auto-detects MPS (Apple Silicon), CUDA, or CPU
- **UI Framework**: Flet (Python web/desktop framework)
- **Threading**: Model loading and image generation run in background threads to prevent UI blocking
- **Image Handling**: PIL for image processing, base64 encoding for Flet display

## Dependencies

- `flet[all]>=0.28.3`: UI framework
- `torch>=2.7.1`: PyTorch for model inference  
- `transformers>=4.54.1`: HuggingFace transformers for diffusion pipeline

## Model Behavior

- Model loads on first use (not at startup)
- Supports various aspect ratios: square, widescreen, portrait
- Default generation parameters: 50 steps, CFG scale 4.0
- Images saved with timestamp filenames in current directory