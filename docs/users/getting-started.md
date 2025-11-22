# Getting Started with Flet Image Creator

Welcome to Flet Image Creator! This guide will help you install and start generating AI images.

## What is Flet Image Creator?

Flet Image Creator is a desktop application for generating images using artificial intelligence. It uses the Qwen diffusion model to transform your text descriptions into beautiful images.

**Key Features:**
- Generate images from text descriptions
- Multiple aspect ratios (square, widescreen, portrait)
- Adjustable quality settings
- Real-time generation progress
- Save images to your computer

## System Requirements

### Minimum Requirements
- **Operating System:** Windows 10/11, macOS 11+, or Linux
- **Python:** 3.12 or higher
- **RAM:** 16 GB minimum
- **Storage:** 30 GB free space (for model download)
- **Internet:** Required for first-time model download

### Recommended for Best Performance
- **GPU:** NVIDIA graphics card with 8+ GB VRAM, or Apple Silicon Mac
- **RAM:** 32 GB or more

## Installation

### Step 1: Install Python

Download and install Python 3.12 or higher from [python.org](https://www.python.org/downloads/).

### Step 2: Install uv (Recommended)

uv is a fast Python package manager. Install it following the instructions at [docs.astral.sh/uv](https://docs.astral.sh/uv/).

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 3: Download Flet Image Creator

Download the application:
```bash
git clone https://github.com/hherb/flet_imagecreator.git
cd flet_imagecreator
```

Or download and extract the ZIP file from the repository.

### Step 4: Install Dependencies

```bash
uv sync
```

This will download all required packages automatically.

## Running the Application

Start the application with:

```bash
uv run python src/qwenimage_fletui.py
```

Or:

```bash
uv run python main.py
```

A window will open with the Image Generator interface.

## First-Time Setup

### Loading the Model

When you first run the application:

1. Click the **"Load Model"** button
2. Wait for the model to download (this may take several minutes on first run)
3. The status will show "Model loaded successfully!" when ready
4. The **"Generate Image"** button will become active

**Note:** The model is approximately 20 GB and will be cached on your computer for future use.

## Quick Start: Generate Your First Image

1. **Enter a prompt** in the "Prompt" field
   - Example: "A beautiful sunset over a calm ocean with vibrant orange and purple colors"

2. **Select an aspect ratio** from the dropdown
   - 16:9 (Widescreen) is great for landscapes
   - 1:1 (Square) is perfect for portraits
   - 9:16 (Portrait) works well for vertical subjects

3. **Click "Generate Image"**

4. **Wait for generation** (progress bar shows status)

5. **View your image** in the display area below

6. **Save the image** by clicking "Save Image"

## Next Steps

- Read the [User Guide](user-guide.md) for detailed instructions on all features
- Learn about [Prompt Writing Tips](prompt-tips.md) to get better results
- Check the [Troubleshooting Guide](troubleshooting.md) if you encounter issues
