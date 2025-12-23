import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2ForConditionalGeneration, InstructBlipForConditionalGeneration

def print_vision_modules(model_name, model_class):
    print(f"--- Inspecting {model_name} ---")
    try:
        model = model_class.from_pretrained(model_name, device_map="auto")
        
        # Identify vision model part
        if hasattr(model, 'vision_tower'): # LLaVA
            vision_model = model.vision_tower
        elif hasattr(model, 'vision_model'): # BLIP-2 / InstructBLIP
            vision_model = model.vision_model
        else:
            print("Could not find standard vision module. Printing top level modules:")
            print(model)
            return

        print("Vision Model Type:", type(vision_model))
        for name, module in vision_model.named_modules():
            print(f"Layer: {name} | Type: {type(module)}")
                
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

print("Starting inspection...")
# LLaVA
print_vision_modules("llava-hf/llava-1.5-7b-hf", LlavaForConditionalGeneration)

# BLIP-2
print_vision_modules("Salesforce/blip2-flan-t5-xl", Blip2ForConditionalGeneration)

# InstructBLIP
print_vision_modules("Salesforce/instructblip-vicuna-7b", InstructBlipForConditionalGeneration)
