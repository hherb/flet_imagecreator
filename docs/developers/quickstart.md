# Developer Quickstart Guide

This guide will help you get started with developing and extending the Flet Image Creator application.

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- GPU with CUDA support (recommended) or Apple Silicon Mac with MPS, or CPU (slower)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hherb/flet_imagecreator.git
cd flet_imagecreator
```

### 2. Install Dependencies

Using uv (recommended):

```bash
uv sync
```

Using pip:

```bash
pip install -e ".[dev]"
```

### 3. Run the Application

```bash
# Using uv
uv run python src/qwenimage_fletui.py

# Or using the entry point
uv run python main.py

# Or directly with Python
python src/qwenimage_fletui.py
```

## Project Structure

```
flet_imagecreator/
├── main.py                    # Application entry point
├── pyproject.toml             # Project configuration
├── src/
│   ├── __init__.py            # Package exports
│   └── qwenimage_fletui.py    # Main application module
├── docs/
│   ├── developers/            # Developer documentation
│   └── users/                 # User documentation
└── CLAUDE.md                  # AI assistant instructions
```

## Architecture Overview

The application follows a layered architecture with clear separation of concerns:

### 1. Pure Functions Layer (Business Logic)

Located at the top of `qwenimage_fletui.py`, these functions have no side effects and are easy to test:

- `get_optimal_device()` - Detects the best available compute device
- `validate_generation_params()` - Validates and converts UI parameters
- `enhance_prompt()` - Appends quality modifiers to prompts
- `calculate_progress()` - Computes progress percentages
- `convert_image_to_base64()` - Encodes images for display

### 2. Data Classes

Immutable containers for typed data:

- `GenerationParams` - Image generation parameters (frozen dataclass)
- `ModelConfig` - Model loading configuration

### 3. Side Effects Layer (I/O Operations)

Classes that perform I/O operations:

- `ModelLoader` - Loads the diffusion pipeline from HuggingFace
- `ImageSaver` - Saves generated images to disk
- `UIUpdater` - Updates Flet UI components

### 4. Orchestration Layer

- `ImageGenerationOrchestrator` - Coordinates generation workflow and image storage
- `QwenImageGenerator` - Main interface to the Qwen diffusion model

### 5. UI Layer

The `main()` function sets up the Flet UI with:
- Prompt input fields
- Parameter controls (aspect ratio, steps, CFG scale, seed)
- Progress tracking components
- Image display container

## Key Components

### QwenImageGenerator

The main class for interacting with the Qwen model:

```python
from src.qwenimage_fletui import QwenImageGenerator, GenerationParams

# Create generator instance
generator = QwenImageGenerator()

# Load the model (downloads on first run)
generator.load_model(progress_callback=print)

# Create parameters
params = GenerationParams(
    prompt="A beautiful sunset over the ocean",
    negative_prompt="blurry, low quality",
    enhancement="Ultra HD, 4K, cinematic composition",
    width=1664,
    height=928,
    num_inference_steps=50,
    cfg_scale=4.0,
    seed=42
)

# Generate image
image = generator.generate_image(params, progress_callback=lambda p: print(p))
image.save("output.png")
```

### Device Detection

The application automatically selects the best available device:

```python
from src.qwenimage_fletui import get_optimal_device

device, dtype = get_optimal_device()
print(f"Using {device} with {dtype}")
# Output: "Using cuda with torch.bfloat16"
```

Priority order:
1. MPS (Apple Silicon)
2. CUDA (NVIDIA GPU)
3. CPU (fallback)

### Parameter Validation

Use `validate_generation_params()` to convert raw UI inputs:

```python
from src.qwenimage_fletui import validate_generation_params

ui_params = {
    "prompt": "A cat",
    "negative_prompt": "",
    "enhancement": "4K, detailed",
    "aspect_ratio": "1:1 (Square)",
    "steps": 50,
    "cfg_scale": 4.0,
    "seed": "42"
}

params = validate_generation_params(ui_params)
print(params.width, params.height)  # 1328, 1328
```

## Extending the Application

### Adding New Aspect Ratios

Edit the `ASPECT_RATIOS` constant in `qwenimage_fletui.py`:

```python
ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 (Square)": (1328, 1328),
    "16:9 (Widescreen)": (1664, 928),
    # Add new ratios here
    "21:9 (Ultrawide)": (1792, 768),
}
```

### Adding Progress Callbacks

The generation system supports progress callbacks for real-time updates:

```python
def my_progress_callback(progress_data: Dict[str, Any]) -> None:
    print(f"Step {progress_data['progress_text']}")
    print(f"Progress: {progress_data['progress_value'] * 100:.1f}%")

image = generator.generate_image(params, progress_callback=my_progress_callback)
```

### Creating a Batch Processing Script

```python
from src.qwenimage_fletui import (
    QwenImageGenerator,
    validate_generation_params,
    generate_filename,
)

generator = QwenImageGenerator()
generator.load_model()

prompts = [
    "A mountain landscape",
    "A city skyline at night",
    "An underwater coral reef",
]

for prompt in prompts:
    params = validate_generation_params({
        "prompt": prompt,
        "aspect_ratio": "16:9 (Widescreen)",
        "steps": 50,
        "cfg_scale": 4.0,
        "seed": 42
    })

    image = generator.generate_image(params)
    filename = generate_filename()
    image.save(filename)
    print(f"Saved: {filename}")
```

## Development Tools

### Running Type Checks

```bash
uv run mypy src/
```

### Running Linting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Running Tests

```bash
uv run pytest
```

## Common Issues

### Model Download Fails

The Qwen model is large (~20GB). Ensure you have:
- Stable internet connection
- Sufficient disk space
- HuggingFace access (may need to accept model license)

### Out of Memory Errors

If you encounter OOM errors:
1. Reduce image dimensions (use smaller aspect ratio)
2. Reduce inference steps
3. Use CPU instead of GPU (slower but uses less VRAM)

### Import Errors

If `QwenImagePipeline` import fails, the application falls back to the generic `DiffusionPipeline`. This is normal if you're using an older version of the diffusers library.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## API Reference

See the inline docstrings in `src/qwenimage_fletui.py` for detailed API documentation. All public functions and classes include comprehensive docstrings with:
- Description of purpose
- Arguments and their types
- Return values
- Usage examples
