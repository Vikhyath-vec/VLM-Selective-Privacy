import os
import random
import shutil
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration

class VLMVerifier:
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.loaded = False

    def load_models(self):
        """
        Loads LLaVA, BLIP-2, and InstructBLIP into memory.
        Uses device_map="auto" to offload to CPU if GPU memory is insufficient.
        """
        if self.loaded:
            return

        print("Loading VLMs (this may take a while)...")
        
        try:
            # LLaVA
            print("Loading LLaVA-1.5-7b...")
            self.processors['llava'] = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.models['llava'] = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                device_map="auto"
            )

            # BLIP-2
            print("Loading BLIP-2 Flan-T5-XL...")
            self.processors['blip2'] = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
            self.models['blip2'] = AutoModelForVision2Seq.from_pretrained(
                "Salesforce/blip2-flan-t5-xl",
                device_map="auto"
            )

            # InstructBLIP
            print("Loading InstructBLIP Vicuna-7b...")
            self.processors['instructblip'] = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.models['instructblip'] = InstructBlipForConditionalGeneration.from_pretrained(
                "Salesforce/instructblip-vicuna-7b",
                device_map="auto"
            )
            
            self.loaded = True
            print("All models loaded successfully.")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Ensure you have 'transformers', 'accelerate', and 'bitsandbytes' installed.")
            exit(1)

    def verify_single_label(self, image, label):
        """Verifies if a single label is detected by all models."""
        prompt = f"Is there any {label} in the image?"
        
        # 1. LLaVA
        # LLaVA Chat Template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        inputs = self.processors['llava'].apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.models['llava'].device)
        
        outputs = self.models['llava'].generate(**inputs, max_new_tokens=30)
        llava_out = self.processors['llava'].decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip().lower()
        
        if "yes" not in llava_out:
            return False

        # 2. BLIP-2
        inputs = self.processors['blip2'](image, prompt, return_tensors="pt").to(self.models['blip2'].device)
        outputs = self.models['blip2'].generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_new_tokens=30,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        blip2_out = self.processors['blip2'].batch_decode(outputs, skip_special_tokens=True)[0].strip().lower()
        
        if "yes" not in blip2_out:
            return False

        # 3. InstructBLIP
        inputs = self.processors['instructblip'](image, prompt, return_tensors="pt").to(self.models['instructblip'].device)
        outputs = self.models['instructblip'].generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_new_tokens=30,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        instruct_out = self.processors['instructblip'].batch_decode(outputs, skip_special_tokens=True)[0].strip().lower()
        
        if "yes" not in instruct_out:
            return False
            
        return True

    def verify_all_labels(self, image_path, labels):
        """Verifies that ALL labels in the list are detected by ALL models."""
        try:
            image = Image.open(image_path).convert("RGB")
            for label in labels:
                if not self.verify_single_label(image, label):
                    print(f"  [Verification Failed] Label '{label}' not detected by all models.")
                    return False
            return True
        except Exception as e:
            print(f"Error during verification of {image_path}: {e}")
            return False

def load_mapping(mapping_path):
    """Loads the ImageNet wnid to label mapping."""
    try:
        with open(mapping_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return {}

def count_objects(xml_path):
    """Counts the number of objects in the XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return len(root.findall('object'))
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return 0

def get_labels(xml_path, mapping):
    """Gets the human-readable labels of all objects in the XML."""
    labels = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            wnid = obj.find('name').text
            labels.append(mapping.get(wnid, wnid))
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
    return labels

def select_random_subset(image_dir, bbox_dir, n, dest_dir, mapping_path, seed, verifier):
    """
    Selects n random images with <= 2 objects and verifies detection.
    """
    image_path = Path(image_dir)
    bbox_path = Path(bbox_dir)
    dest_path = Path(dest_dir)
    
    mapping = load_mapping(mapping_path)

    if not image_path.exists():
        print(f"Error: Image directory '{image_dir}' does not exist.")
        return
    if not bbox_path.exists():
        print(f"Error: Bounding box directory '{bbox_dir}' does not exist.")
        return

    # Create destination directories
    dest_images_path = dest_path / "images"
    dest_bboxes_path = dest_path / "bboxes"
    dest_images_path.mkdir(parents=True, exist_ok=True)
    dest_bboxes_path.mkdir(parents=True, exist_ok=True)

    # Get list of all potential image files
    image_extensions = {".JPEG", ".jpg", ".jpeg"}
    all_images = [f for f in os.listdir(image_path) if Path(f).suffix in image_extensions]
    
    # Shuffle to ensure randomness in selection
    random.seed(seed)
    random.shuffle(all_images)
    
    selected_images = []
    print(f"Searching for {n} images with <= 2 objects and valid detection...")
    
    # Load models once before the loop
    verifier.load_models()
    
    for img_name in all_images:
        if len(selected_images) >= n:
            break
            
        # Check corresponding XML
        xml_name = Path(img_name).stem + ".xml"
        src_xml = bbox_path / xml_name
        
        if not src_xml.exists():
            continue
            
        # Filter 1: Object Count <= 2 and >= 1
        if count_objects(src_xml) > 2 or count_objects(src_xml) < 1:
            continue
            
        # Filter 2: Detection Verification
        labels = get_labels(src_xml, mapping)
        src_img = image_path / img_name
        
        print(f"Checking {img_name} with labels {labels}...")
        if verifier.verify_all_labels(src_img, labels):
            selected_images.append(img_name)
            
            # Copy files
            dst_img = dest_images_path / img_name
            shutil.copy2(src_img, dst_img)
            
            dst_xml = dest_bboxes_path / xml_name
            shutil.copy2(src_xml, dst_xml)
            
            print(f"  Selected: {img_name}")

    print(f"Successfully copied {len(selected_images)} image/xml pairs to {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select random subset of ImageNet validation data.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("--bbox_dir", type=str, required=True, help="Path to directory containing bounding box XMLs")
    parser.add_argument("--n", type=int, required=True, help="Number of images to select")
    parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory")
    parser.add_argument("--mapping_path", type=str, default="imagenet_mapping.json", help="Path to ImageNet mapping JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    verifier = VLMVerifier()
    select_random_subset(args.image_dir, args.bbox_dir, args.n, args.dest_dir, args.mapping_path, args.seed, verifier)
