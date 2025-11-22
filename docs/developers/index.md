# Developer Documentation

Welcome to the Flet Image Creator developer documentation. This section is for developers who want to understand, modify, or extend the application.

## Contents

### [Quickstart Guide](quickstart.md)

Get up and running with development quickly:
- Installation and setup
- Project structure overview
- Running the application
- Basic code examples
- Extension points

### [Architecture Documentation](architecture.md)

Deep dive into the application design:
- Layer diagram and component relationships
- Data classes and pure functions
- Side effects and I/O operations
- Threading model
- Data flow diagrams
- Extension points

## Quick Links

### Running the App

```bash
# Install dependencies
uv sync

# Run the application
uv run python src/qwenimage_fletui.py
```

### Key Files

| File | Purpose |
|------|---------|
| `src/qwenimage_fletui.py` | Main application module |
| `main.py` | Entry point script |
| `src/__init__.py` | Package exports |
| `pyproject.toml` | Project configuration |

### Main Classes

| Class | Purpose |
|-------|---------|
| `QwenImageGenerator` | Model loading and image generation |
| `UIUpdater` | Centralized UI updates |
| `ImageGenerationOrchestrator` | Workflow coordination |
| `ModelLoader` | HuggingFace model loading |

### Key Functions

| Function | Purpose |
|----------|---------|
| `get_optimal_device()` | Detect best compute device |
| `validate_generation_params()` | Validate UI inputs |
| `enhance_prompt()` | Add quality modifiers |
| `convert_image_to_base64()` | Encode images for display |

## Development Setup

1. Clone the repository
2. Run `uv sync` to install dependencies
3. Run `uv sync --extra dev` for development tools
4. Use `uv run mypy src/` for type checking
5. Use `uv run ruff check src/` for linting

## Contributing

See the [Contributing Guide](../../CONTRIBUTING.md) for guidelines on submitting changes.
