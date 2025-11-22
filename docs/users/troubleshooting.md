# Troubleshooting Guide

This guide helps you resolve common issues with Flet Image Creator.

## Installation Issues

### Python Not Found

**Symptom:** `python: command not found` or similar errors

**Solutions:**
1. Ensure Python 3.12+ is installed: `python --version` or `python3 --version`
2. Add Python to your system PATH
3. On Windows, re-run the installer and check "Add Python to PATH"

### uv Installation Fails

**Symptom:** Cannot install uv package manager

**Solutions:**
1. Try installing with pip instead: `pip install uv`
2. Check your internet connection
3. On corporate networks, you may need proxy settings

### Dependency Installation Fails

**Symptom:** `uv sync` or `pip install` errors

**Solutions:**
1. Ensure you have sufficient disk space (at least 10 GB)
2. Try upgrading pip: `pip install --upgrade pip`
3. On Windows, install Visual C++ Build Tools
4. Check for network/firewall issues

## Model Loading Issues

### Model Download Fails

**Symptom:** "Error loading model" or download hangs

**Solutions:**
1. **Check internet connection** - Model download requires stable internet
2. **Check disk space** - You need ~30 GB free for the model
3. **Clear cache and retry:**
   ```bash
   # Linux/macOS
   rm -rf ~/.cache/huggingface/

   # Windows
   rmdir /s %USERPROFILE%\.cache\huggingface
   ```
4. **Use a VPN** if HuggingFace is blocked in your region

### Model Loading Takes Forever

**Symptom:** "Loading model..." status for more than 30 minutes

**Solutions:**
1. First download can take 30-60 minutes depending on internet speed
2. Subsequent loads should be faster (cached model)
3. Check Task Manager/Activity Monitor for download progress
4. Consider downloading during off-peak hours

### Out of Memory During Load

**Symptom:** Application crashes or "CUDA out of memory" error

**Solutions:**
1. Close other applications using GPU memory
2. Reduce system memory usage
3. Try using CPU instead (slower but uses less memory):
   - The app auto-detects devices, but you can modify the code to force CPU

## Generation Issues

### "Model not loaded" Error

**Symptom:** Error when clicking "Generate Image"

**Solution:** Click "Load Model" first and wait for "Model loaded successfully!" status.

### Generation is Very Slow

**Symptom:** Each generation takes more than 5 minutes

**Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Using CPU instead of GPU | Install CUDA (NVIDIA) or use Apple Silicon |
| Too many inference steps | Reduce steps to 30-50 |
| High resolution | Use smaller aspect ratio |
| Other apps using GPU | Close GPU-intensive applications |

**Expected Times:**
- GPU (CUDA/MPS): 30-120 seconds for 50 steps
- CPU: 5-30 minutes for 50 steps

### CUDA Out of Memory

**Symptom:** "CUDA out of memory" or "torch.cuda.OutOfMemoryError"

**Solutions:**
1. **Reduce image size** - Use smaller aspect ratio
2. **Reduce steps** - Lower to 30-40 steps
3. **Close other GPU apps** - Games, video editors, other AI tools
4. **Restart your computer** to clear GPU memory
5. **Enable memory-efficient attention** (for developers):
   ```python
   pipe.enable_attention_slicing()
   ```

### Images Look Wrong

**Problem: Blurry or Low Quality**
- Increase inference steps to 50-70
- Add "highly detailed, sharp focus" to prompt
- Add "blurry, low quality" to negative prompt

**Problem: Wrong Colors**
- Be specific about colors in prompt: "vibrant blue sky" vs "sky"
- Use reference lighting: "golden hour", "studio lighting"

**Problem: Distorted Faces/Bodies**
- Add "well-proportioned, anatomically correct" to prompt
- Add "distorted, deformed" to negative prompt
- Try different seeds

**Problem: Doesn't Match Prompt**
- Increase CFG scale to 5-7
- Simplify prompt to focus on main elements
- Break complex prompts into key phrases

### Progress Bar Not Moving

**Symptom:** Progress bar stays at 0% or freezes

**Solutions:**
1. Wait a few minutes - first step can take longer
2. Check if application is "Not Responding" in Task Manager
3. If frozen for more than 10 minutes, restart the application

## UI Issues

### Window is Too Small

**Symptom:** Cannot see all controls

**Solution:** Resize the window by dragging edges. Default size is 1200x800.

### Random Seed Button Doesn't Work

**Symptom:** Clicking "Random" doesn't change the seed

**Solution:** This was fixed in recent versions. Update to the latest version.

### Configuration Panel Won't Expand

**Symptom:** Cannot see prompt fields after generation

**Solution:** Click the arrow/expand button (▲) next to the title.

### Image Display is Blank

**Symptom:** "Generated image will appear here" never changes

**Possible Causes:**
1. Generation hasn't completed - wait for progress bar
2. Generation failed - check status text for errors
3. Memory issue - restart application

## System-Specific Issues

### macOS

**Problem: "Cannot be opened because the developer cannot be verified"**

Solution:
1. Go to System Preferences > Security & Privacy
2. Click "Allow Anyway" for the blocked app
3. Or run: `xattr -cr /path/to/flet_imagecreator`

**Problem: MPS (Metal) not detected on Apple Silicon**

Solution:
1. Ensure you're using native ARM Python, not Rosetta
2. Update to macOS 12.3 or later
3. Update PyTorch to latest version

### Windows

**Problem: Anti-virus blocks the application**

Solution:
1. Add an exception for the flet_imagecreator folder
2. Temporarily disable real-time protection during first run

**Problem: "DLL load failed"**

Solution:
1. Install Microsoft Visual C++ Redistributable
2. Update graphics drivers
3. Install CUDA toolkit if using NVIDIA GPU

### Linux

**Problem: Display issues or window doesn't appear**

Solution:
1. Ensure a display server is running (X11 or Wayland)
2. Set `DISPLAY` environment variable if using SSH
3. Install required system libraries:
   ```bash
   sudo apt install libgtk-3-0 libglib2.0-0
   ```

## Getting Help

If you're still experiencing issues:

1. **Check the logs** - Look at terminal output for error messages
2. **Update dependencies** - Run `uv sync` to update packages
3. **Search existing issues** - Check the GitHub issues page
4. **Open a new issue** - Provide:
   - Operating system and version
   - Python version (`python --version`)
   - Error message (full text)
   - Steps to reproduce

## Diagnostic Information

To help troubleshoot, run this to gather system info:

```bash
python -c "
import sys
import torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"
```

Include this output when reporting issues.
