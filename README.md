# VLM Privacy Protection

This repository implements adversarial perturbation training to minimize Vision-Language Model (VLM) attention to specific Regions of Interest (ROI), based on the VIP methodology.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Vikhyath-vec/VLM-Privacy-Protection.git
cd VLM-Privacy-Protection
pip install -r requirements.txt
```

## 📊 Adversarial Image Training

### Basic Usage

Generate adversarial images that minimize model attention to ROI while maintaining visual quality:

```bash
python src/train_adversarial_image.py \
  --image_path data/images/example.JPEG \
  --xml_path data/bboxes/example.xml \
  --model_id Salesforce/blip2-flan-t5-xl \
  --num_iterations 100
```

### Recommended Configuration

**For imperceptible perturbations:**
```bash
python src/train_adversarial_image.py \
  --image_path data/images/ILSVRC2012_val_00000073.JPEG \
  --xml_path data/bboxes/ILSVRC2012_val_00000073.xml \
  --model_id Salesforce/blip2-flan-t5-xl \
  --num_iterations 200 \
  --attention_aggregation rollout \
  --value_aggregation l2 \
  --lambda_v 0.1 \
  --perceptual_loss lpips \
  --lambda_perceptual 0.5 \
  --epsilon 0.03 --norm linf \
  --learning_rate 0.01
```

### Command-Line Arguments

#### **Core Parameters**
| Argument | Default | Description |
|----------|---------|-------------|
| `--image_path` | Required | Path to input image |
| `--xml_path` | Required | Path to XML annotation (bounding boxes) |
| `--model_id` | `blip2-flan-t5-xl` | VLM model: `blip2-flan-t5-xl`, `llava-1.5-7b-hf`, `instructblip-vicuna-7b` |
| `--output_dir` | `adversarial_output` | Directory for output images and deltas |

#### **Training Parameters**
| Argument | Default | Description |
|----------|---------|-------------|
| `--num_iterations` | 100 | Number of optimization iterations |
| `--learning_rate` | 0.01 | Learning rate for delta optimization |
| `--num_layers` | -1 | Number of layers to use (-1 = all) |

#### **Loss Components**
| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--attention_aggregation` | `rollout` | `sum`, `rollout` | Method to aggregate attention |
| `--value_aggregation` | `l2` | `l2`, `frobenius` | Method to aggregate values |
| `--lambda_v` | 1.0 | float | Weight for value regularization |
| `--perceptual_loss` | `none` | `none`, `mse`, `lpips`, `vgg`, `ssim` | Perceptual similarity method |
| `--lambda_perceptual` | 0.0 | float | Weight for perceptual loss |

#### **Perturbation Constraints**
| Argument | Default | Description |
|----------|---------|-------------|
| `--epsilon` | 0.0 | Max perturbation (0 = no constraint). Recommended: 0.03 for linf |
| `--norm` | `linf` | Constraint norm: `linf` (per-pixel) or `l2` (total energy) |

### Example Commands

**Conservative (High Visual Quality):**
```bash
python src/train_adversarial_image.py \
  --image_path data/images/example.JPEG \
  --xml_path data/bboxes/example.xml \
  --perceptual_loss lpips \
  --lambda_perceptual 1.0 \
  --epsilon 0.03 --norm linf
```

**Balanced:**
```bash
python src/train_adversarial_image.py \
  --image_path data/images/example.JPEG \
  --xml_path data/bboxes/example.xml \
  --lambda_v 0.1 \
  --perceptual_loss mse \
  --lambda_perceptual 5.0 \
  --epsilon 0.05
```

**Attack-Focused:**
```bash
python src/train_adversarial_image.py \
  --image_path data/images/example.JPEG \
  --xml_path data/bboxes/example.xml \
  --lambda_v 0.1 \
  --epsilon 0.12 --norm linf
```

### Output

Training produces:
- `adversarial_output/adversarial_<image_name>.JPEG` - Adversarial image
- `adversarial_output/delta_<image_name>.npy` - Perturbation (δ)

## 🧪 Testing

Comprehensive test suite in [`test/`](test/) directory. See [test/README.md](test/README.md) for details.

```bash
# Test preprocessing
python test/test_preprocessing.py

# Test attention rollout
python test/test_rollout.py

# Test capture hooks for BLIP-2
python test/test_capture_hooks.py --model_id blip2

# Test with debug output
python test/test_capture_hooks.py --model_id llava --debug
```

## 📂 Project Structure

```
VLM-Privacy-Protection/
├── src/
│   ├── train_adversarial_image.py  # Main training script
│   ├── attention_rollout.py        # Attention rollout implementation
│   └── utils.py                    # Preprocessing utilities
├── test/
│   ├── test_preprocessing.py       # Tests preprocessing logic
│   ├── test_rollout.py            # Tests attention rollout
│   └── test_capture_hooks.py      # Tests attention/value capture
├── prepare-data/
│   ├── create_mapping.py          # ImageNet class mapping
│   └── create_dataset.py          # Dataset subset creation
└── data/
    ├── images/                     # Input images
    └── bboxes/                     # XML annotations
```

## 📖 Data Preparation

Scripts in `prepare-data/` create ImageNet validation subsets. See original README section below for details.

### Download Data

- **Images:** [ILSVRC2013_DET_val.tar](https://image-net.org/data/ILSVRC/2013/ILSVRC2013_DET_val.tar)
- **Bounding Boxes:** [ILSVRC2013_DET_bbox_val.tgz](https://image-net.org/data/ILSVRC/2013/ILSVRC2013_DET_bbox_val.tgz)

### Create Subset

```bash
# 1. Generate mapping
python prepare-data/create_mapping.py --output_path imagenet_mapping.json

# 2. Create subset (filters for ≤2 objects, verified by VLMs)
python prepare-data/create_dataset.py \
    --image_dir "/path/to/ILSVRC2013_DET_val" \
    --bbox_dir "/path/to/ILSVRC2013_DET_bbox_val" \
    --n 1000 \
    --dest_dir "data/subset_1000" \
    --mapping_path "imagenet_mapping.json" \
    --seed 42
```

## 🔬 How It Works

The adversarial training optimizes a perturbation δ to minimize VLM attention to protected regions:

1. **Input**: Clean image + ROI bounding boxes
2. **Optimization**: Minimize attention to ROI patches via gradient descent
3. **Regularization**: Control value matrices and visual similarity
4. **Constraints**: Epsilon bounds keep perturbations imperceptible

**Loss Components:**
- **Attention Loss**: Sum of attention weights to ROI patches
- **Value Regularization**: L2/Frobenius norm of value vectors
- **Perceptual Loss**: Visual similarity (MSE/LPIPS/VGG/SSIM)
- **Epsilon Constraint**: L∞ or L2 bound on δ
