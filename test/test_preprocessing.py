import argparse
import os
import sys
import torch
from PIL import Image
from transformers import AutoProcessor

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import (
    preprocess_image_and_boxes,
    get_patch_size,
    visualize_preprocessing
)

def main():
    parser = argparse.ArgumentParser(description="Test Image Preprocessing")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--xml_path", type=str, required=True, help="Path to XML annotation")
    parser.add_argument("--model_id", type=str, default="Salesforce/blip2-flan-t5-xl", help="Model ID", choices=[
        "Salesforce/blip2-flan-t5-xl",
        "llava-hf/llava-1.5-7b-hf",
        "Salesforce/instructblip-vicuna-7b",
    ])
    parser.add_argument("--output_dir", type=str, default="test_results", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading processor for {args.model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(args.model_id)
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    print("Preprocessing...")
    resized_img, scaled_objects, target_size = preprocess_image_and_boxes(args.image_path, args.xml_path, processor)
    
    # Get patch size from model config
    patch_size = get_patch_size(args.model_id)
    
    # Save our manually preprocessed version
    vis_path = os.path.join(args.output_dir, "preprocessing_check.png")
    visualize_preprocessing(resized_img, scaled_objects, patch_size, vis_path)
    
    # Also process through the actual processor for comparison
    print("\nProcessing through actual processor for verification...")
    raw_image = Image.open(args.image_path).convert('RGB')
    
    # Process with the actual processor (without text)
    # For vision-language models, we need to handle differently
    if "llava" in args.model_id.lower():
        # LLaVA processor expects text input
        inputs = processor(text="dummy", images=raw_image, return_tensors="pt")
    else:
        # BLIP-2 and InstructBLIP
        inputs = processor(images=raw_image, return_tensors="pt")
    
    # Extract the processed image tensor and convert back to PIL for visualization
    # The processor returns normalized tensors, we need to denormalize
    processed_tensor = inputs['pixel_values'][0]  # Shape: [C, H, W]
    
    # Get the processor's normalization parameters
    if hasattr(processor, 'image_processor'):
        config = processor.image_processor
    else:
        config = processor
    
    mean = torch.tensor(config.image_mean).view(3, 1, 1)
    std = torch.tensor(config.image_std).view(3, 1, 1)
    
    # Denormalize: tensor * std + mean
    denormalized = processed_tensor * std + mean
    
    # Clip to [0, 1] and convert to uint8
    denormalized = torch.clamp(denormalized, 0, 1)
    import numpy as np
    processor_image_array = (denormalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    processor_image = Image.fromarray(processor_image_array)
    
    # Save processor version
    processor_img_path = os.path.join(args.output_dir, "processor_output.png")
    processor_image.save(processor_img_path)
    print(f"Saved processor output to {processor_img_path}")
    
    # Save our manual version for easy comparison
    manual_img_path = os.path.join(args.output_dir, "manual_output.png")
    resized_img.save(manual_img_path)
    print(f"Saved manual output to {manual_img_path}")
    
    print("\nDone. Compare the two outputs to verify preprocessing correctness.")

if __name__ == "__main__":
    main()
