# VLM Privacy Protection - Test Suite

This directory contains test scripts for validating different components of the adversarial training system.

## Test Files

### [`test_preprocessing.py`](test_preprocessing.py)
**Purpose**: Validates image preprocessing and bounding box transformation logic.

**Features:**
- Tests preprocessing for different VLM models (LLaVA, BLIP-2, InstructBLIP)
- Verifies correct handling of resize vs. center crop strategies
- Compares manual preprocessing with processor output
- Visualizes preprocessed images with scaled bounding boxes

**Usage:**
```bash
python test/test_preprocessing.py
```

**What it validates:**
- Image resizing (shortest edge for LLaVA, direct resize for BLIP-2/InstructBLIP)
- Center cropping for CLIP-based models
- Bounding box coordinate scaling
- Patch index calculation

---

### [`test_rollout.py`](test_rollout.py)
**Purpose**: Tests attention rollout computation and visualization.

**Features:**
- Generates attention rollout masks for test images
- Supports all VLM models
- Visualizes CLS token attention to image patches
- Saves visualizations for manual inspection

**Usage:**
```bash
python test/test_rollout.py
```

**What it validates:**
- Attention weight extraction from vision models
- Rollout computation (fusion + discard logic)
- Mask generation and visualization
- Model compatibility

---

### [`test_capture_hooks.py`](test_capture_hooks.py)
**Purpose**: Validates `AttentionValueCapture` hooks across different architectures.

**Features:**
- Tests attention and value matrix capture
- Supports model selection via command-line
- Debug mode to trace hook execution
- Tests attention rollout integration

**Usage:**
```bash
# Test BLIP-2
python test/test_capture_hooks.py --model_id blip2

# Test LLaVA with debug output
python test/test_capture_hooks.py --model_id llava --debug

# Test InstructBLIP
python test/test_capture_hooks.py --model_id instructblip

# Use full model ID
python test/test_capture_hooks.py --model_id Salesforce/blip2-flan-t5-xl
```

**Arguments:**
- `--model_id`: Model to test (`blip2`, `llava`, `instructblip`, or full HF model ID)
- `--debug`: Enable debug output from hooks

**What it validates:**
- Hook registration for different architectures
- Attention weight capture (shape, gradients)
- Value matrix extraction (separate v_proj vs. combined qkv)
- Gradient flow for optimization
- Attention rollout computation

---

## Running All Tests

```bash
# Run all tests sequentially
python test/test_preprocessing.py
python test/test_rollout.py
python test/test_capture_hooks.py --model_id blip2
```

## Test Requirements

All tests use the sample data in [`data/images/`](../data/images/) and [`data/bboxes/`](../data/bboxes/). Make sure these directories contain test images and XML annotations.

## Expected Output

- **test_preprocessing.py**: Visualization images saved to `test_results/`
- **test_rollout.py**: Attention maps saved to `test_results/`
- **test_capture_hooks.py**: Console output showing captured layers and shapes
