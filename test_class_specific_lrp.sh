#!/bin/bash

# Test script for class-specific LRP

echo "========================================="
echo "Testing Class-Specific LRP"
echo "========================================="

# Test image with porcupine
IMAGE="data/images/ILSVRC2012_val_00000021.JPEG"
XML="data/bboxes/ILSVRC2012_val_00000021.xml"
MAPPING="data/imagenet_mapping.json"

echo ""
echo "Running class-specific LRP attack..."
python src/train_adversarial_image.py \
  --image_path "$IMAGE" \
  --xml_path "$XML" \
  --model_id Salesforce/blip2-flan-t5-xl \
  --attention_aggregation lrp_class_specific \
  --use_class_specific \
  --mapping_path "$MAPPING" \
  --num_iterations 10 \
  --output_dir test_output/class_specific_lrp \
  --visualize_attention

echo ""
echo "========================================="
echo "Test complete!"
echo "Check test_output/class_specific_lrp/ for results"
echo "========================================="
