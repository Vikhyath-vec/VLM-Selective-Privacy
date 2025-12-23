import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
from transformers import AutoConfig

def parse_xml(xml_path):
    """
    Parses Pascal VOC XML to extract bounding boxes.
    Returns a list of dicts: {'name': str, 'bbox': [xmin, ymin, xmax, ymax]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})
    return objects

def get_target_size(processor):
    """
    Extracts target image size from the processor config.
    """
    # Different processors store size in different places
    if hasattr(processor, 'image_processor'):
        config = processor.image_processor
    else:
        config = processor
        
    if hasattr(config, 'crop_size'):
        return config.crop_size['height'], config.crop_size['width']
    elif hasattr(config, 'size'):
        if isinstance(config.size, dict):
            return config.size['height'], config.size['width']
        return config.size, config.size
    
    # Default fallback
    print("Warning: Could not detect size from processor, defaulting to 224x224")
    return 224, 224

def get_patch_size(model_id):
    """
    Extracts patch size from the vision model config.
    Different model architectures store the vision model in different places.
    """
    print(f"Loading model to extract patch size from {model_id}...")
    
    try:
        config = AutoConfig.from_pretrained(model_id)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "patch_size"):
            patch_size = config.vision_config.patch_size
        elif hasattr(config, "patch_size"):
            patch_size = config.patch_size
        else:
            print(f"Warning: Unknown model architecture for {model_id}, defaulting to patch size 14")
            patch_size = 14
            
        print(f"Extracted patch size: {patch_size}")
        return patch_size
    except Exception as e:
        print(f"Error loading model config: {e}")
        print("Defaulting to patch size 14")
        return 14

def preprocess_image_and_boxes(image_path, xml_path, processor):
    """
    Loads image, resizes it to model input size, and scales bounding boxes.
    Properly handles different preprocessing strategies:
    - CLIPImageProcessor (LLaVA): Resize shortest edge, then center crop
    - BlipImageProcessor (BLIP-2, InstructBLIP): Direct resize
    """
    # Load Image
    raw_image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = raw_image.size
    
    # Get processor config
    if hasattr(processor, 'image_processor'):
        config = processor.image_processor
    else:
        config = processor
    
    # Check if processor does center cropping (like CLIPImageProcessor for LLaVA)
    do_center_crop = getattr(config, 'do_center_crop', False)
    
    # Load and prepare to scale boxes
    objects = parse_xml(xml_path)
    scaled_objects = []
    
    if do_center_crop:
        # CLIPImageProcessor: Resize shortest edge first, then center crop
        # Get the resize size (shortest edge)
        if hasattr(config, 'size') and isinstance(config.size, dict):
            if 'shortest_edge' in config.size:
                resize_shortest = config.size['shortest_edge']
            else:
                resize_shortest = min(config.size.get('height', 336), config.size.get('width', 336))
        else:
            resize_shortest = 336  # Default for CLIP
        
        # Get crop size
        if hasattr(config, 'crop_size'):
            crop_h = config.crop_size['height']
            crop_w = config.crop_size['width']
        else:
            crop_h = crop_w = resize_shortest
        
        print(f"Original: {orig_w}x{orig_h}, Resize shortest edge to: {resize_shortest}, Crop to: {crop_w}x{crop_h}")
        
        # Step 1: Resize maintaining aspect ratio (shortest edge = resize_shortest)
        if orig_w < orig_h:
            new_w = resize_shortest
            new_h = int(orig_h * resize_shortest / orig_w)
        else:
            new_h = resize_shortest
            new_w = int(orig_w * resize_shortest / orig_h)
        
        resized_image = raw_image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        
        # Step 2: Center crop
        left = (new_w - crop_w) // 2
        top = (new_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        final_image = resized_image.crop((left, top, right, bottom))
        
        # Scale bounding boxes through both transformations
        for obj in objects:
            bbox = obj['bbox']  # xmin, ymin, xmax, ymax in original image
            
            # Apply resize transformation
            resize_scale_x = new_w / orig_w
            resize_scale_y = new_h / orig_h
            resized_bbox = [
                bbox[0] * resize_scale_x,
                bbox[1] * resize_scale_y,
                bbox[2] * resize_scale_x,
                bbox[3] * resize_scale_y
            ]
            
            # Apply crop transformation (shift by crop offset)
            cropped_bbox = [
                resized_bbox[0] - left,
                resized_bbox[1] - top,
                resized_bbox[2] - left,
                resized_bbox[3] - top
            ]
            
            # Convert to int and clip to crop boundaries
            final_bbox = [
                max(0, min(crop_w, int(cropped_bbox[0]))),
                max(0, min(crop_h, int(cropped_bbox[1]))),
                max(0, min(crop_w, int(cropped_bbox[2]))),
                max(0, min(crop_h, int(cropped_bbox[3])))
            ]
            
            # Only add if bbox is still visible after cropping
            if final_bbox[2] > final_bbox[0] and final_bbox[3] > final_bbox[1]:
                scaled_objects.append({'name': obj['name'], 'bbox': final_bbox})
        
        return final_image, scaled_objects, (crop_w, crop_h)
    
    else:
        # BlipImageProcessor: Direct resize
        target_h, target_w = get_target_size(processor)
        print(f"Original: {orig_w}x{orig_h}, Direct resize to: {target_w}x{target_h}")
        
        resized_image = raw_image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        
        # Scale boxes
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        for obj in objects:
            bbox = obj['bbox']  # xmin, ymin, xmax, ymax
            scaled_bbox = [
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y)
            ]
            # Clip to image boundaries
            scaled_bbox[0] = max(0, scaled_bbox[0])
            scaled_bbox[1] = max(0, scaled_bbox[1])
            scaled_bbox[2] = min(target_w, scaled_bbox[2])
            scaled_bbox[3] = min(target_h, scaled_bbox[3])
            
            scaled_objects.append({'name': obj['name'], 'bbox': scaled_bbox})
        
        return resized_image, scaled_objects, (target_w, target_h)

def get_patch_indices(bbox, image_size, patch_size=14):
    """
    Identifies which patches fall inside the bounding box.
    Assumes ViT patch structure.
    """
    w, h = image_size
    n_patches_w = w // patch_size
    n_patches_h = h // patch_size
    
    xmin, ymin, xmax, ymax = bbox
    
    # Convert pixel coordinates to patch coordinates
    patch_xmin = xmin // patch_size
    patch_ymin = ymin // patch_size
    patch_xmax = (xmax - 1) // patch_size # -1 to handle boundary case
    patch_ymax = (ymax - 1) // patch_size
    
    indices = []
    for y in range(patch_ymin, patch_ymax + 1):
        for x in range(patch_xmin, patch_xmax + 1):
            # Calculate flat index (assuming row-major)
            # Note: ViT usually has a CLS token at index 0. 
            # If we are mapping to the sequence *after* CLS, the index is y * n_w + x.
            # If we include CLS, it's y * n_w + x + 1.
            # AttentionRollout usually handles the CLS token separately or expects 2D spatial mask.
            # Here we return 2D indices (row, col)
            indices.append((y, x))
            
    return indices

def visualize_preprocessing(image, objects, patch_size=14, output_path="debug_preprocessing.png"):
    """
    Visualizes the resized image, bounding boxes, and grid patches.
    """
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    h, w = img_cv.shape[:2]
    
    # Draw grid
    for x in range(0, w, patch_size):
        cv2.line(img_cv, (x, 0), (x, h), (50, 50, 50), 1)
    for y in range(0, h, patch_size):
        cv2.line(img_cv, (0, y), (w, y), (50, 50, 50), 1)
        
    # Draw boxes and highlight patches
    overlay = img_cv.copy()
    
    for obj in objects:
        bbox = obj['bbox']
        cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Highlight patches
        indices = get_patch_indices(bbox, (w, h), patch_size)
        for r, c in indices:
            px = c * patch_size
            py = r * patch_size
            cv2.rectangle(overlay, (px, py), (px + patch_size, py + patch_size), (0, 0, 255), -1)
            
    # Blend overlay
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0, img_cv)
    
    cv2.imwrite(output_path, img_cv)
    print(f"Saved visualization to {output_path}")
