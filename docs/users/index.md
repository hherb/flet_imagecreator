# User Documentation

Welcome to the Flet Image Creator user documentation. This section will help you get started and make the most of the application.

## Contents

### [Getting Started](getting-started.md)

New to Flet Image Creator? Start here:
- System requirements
- Installation instructions
- First-time setup
- Quick start guide

### [User Guide](user-guide.md)

Complete guide to all features:
- Interface overview
- Input fields explained
- Generation parameters
- Using buttons and controls
- Progress tracking
- Tips for best results

### [Prompt Writing Tips](prompt-tips.md)

Learn to write better prompts:
- Prompt structure
- Subject descriptions
- Style references
- Lighting and mood
- Quality terms
- Example prompts by category

### [Troubleshooting](troubleshooting.md)

Having issues? Find solutions:
- Installation problems
- Model loading issues
- Generation errors
- UI issues
- System-specific problems

## Quick Start

1. **Install:** `uv sync`
2. **Run:** `uv run python src/qwenimage_fletui.py`
3. **Load Model:** Click "Load Model" (first time takes a few minutes)
4. **Generate:** Enter a prompt and click "Generate Image"
5. **Save:** Click "Save Image" to keep your creation

## Tips for Great Results

- **Be descriptive:** "A golden sunset over calm ocean waves" beats "sunset"
- **Add style:** Include artistic style like "digital art" or "oil painting"
- **Use quality terms:** Add "highly detailed, 4K" for better results
- **Experiment with seeds:** Same seed + same prompt = same result

## Frequently Asked Questions

**Q: How long does the first model download take?**
A: 30-60 minutes depending on your internet speed. The model is ~20 GB.

**Q: Can I use this offline?**
A: Yes, after the initial model download.

**Q: Why is generation slow on my computer?**
A: Without a GPU, generation uses CPU which is much slower. Consider using CUDA (NVIDIA) or MPS (Apple Silicon).

**Q: Where are my saved images?**
A: In the folder where you ran the application, with timestamped filenames.

**Q: Can I use the generated images commercially?**
A: Check the Qwen model license on HuggingFace for usage terms.
