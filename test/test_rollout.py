import argparse
import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForVision2Seq, InstructBlipProcessor, InstructBlipForConditionalGeneration
import gc

# Add src to path to import AttentionRollout
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from attention_rollout import AttentionRollout

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def test_model(model_name, model_id, image_paths, output_dir):
    print(f"\n--- Testing {model_name} ({model_id}) ---")
    try:
        # Clear cache before loading new model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if model_name == "llava":
            processor = AutoProcessor.from_pretrained(model_id)
            model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="cpu")
        elif model_name == "blip2":
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="cpu")
        elif model_name == "instructblip":
            processor = InstructBlipProcessor.from_pretrained(model_id)
            model = InstructBlipForConditionalGeneration.from_pretrained(model_id, device_map="cpu")
        else:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"Model {model_name} loaded.")
        
        # Initialize Rollout
        rollout = AttentionRollout(model, head_fusion="max", discard_ratio=0.5)

        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
                
            print(f"Processing {os.path.basename(img_path)}...")
            raw_image = Image.open(img_path).convert('RGB')
            
            # Prepare inputs
            text = "Describe the image."
            if model_name == "llava":
                inputs = processor(text=text, images=raw_image, return_tensors="pt").to(model.device)
            else:
                inputs = processor(images=raw_image, text=text, return_tensors="pt").to(model.device)
            
            pixel_values = inputs.pixel_values
            
            # Run Rollout
            mask, _ = rollout(pixel_values, layer_idx=-1)
            mask = mask[0]
            
            # Resize mask to image size
            np_img = np.array(raw_image)
            mask_resized = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
            vis = show_mask_on_image(np_img, mask_resized)
            
            # Save
            output_filename = os.path.join(output_dir, f"{model_name}_{os.path.basename(img_path)}_mask.png")
            cv2.imwrite(output_filename, vis)
            print(f"Saved visualization to {output_filename}")
            
        # Cleanup
        del model
        del processor
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Failed to test {model_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Test Attention Rollout on VLMs")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Images to test
    # Using relative paths assuming running from repo root
    image_dir = os.path.join(os.path.dirname(__file__), '../data/images')
    # Pick a few existing images
    test_images = [
        os.path.join(image_dir, "ILSVRC2012_val_00000147.JPEG"),
        os.path.join(image_dir, "ILSVRC2012_val_00000073.JPEG")
    ]
    
    models_to_test = [
        ("blip2", "Salesforce/blip2-flan-t5-xl"),
        ("instructblip", "Salesforce/instructblip-vicuna-7b"),
        ("llava", "llava-hf/llava-1.5-7b-hf")
    ]
    
    for name, model_id in models_to_test:
        test_model(name, model_id, test_images, args.output_dir)

if __name__ == "__main__":
    main()
