import os
import glob
import subprocess
import json
import csv
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="batch_results")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Setup directories
    img_dir = os.path.join(args.data_dir, "images")
    bbox_dir = os.path.join(args.data_dir, "bboxes")
    mapping_path = os.path.join(args.data_dir, "imagenet_mapping.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get images
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.JPEG")))
    if args.limit:
        image_files = image_files[:args.limit]
    
    print(f"Found {len(image_files)} images to process.")
    summary_data = []

    # Processing Loop
    for img_path in tqdm(image_files, desc="Processing"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(bbox_dir, f"{basename}.xml")
        
        if not os.path.exists(xml_path):
            continue
            
        single_output_dir = os.path.join(args.output_dir, "details")
        
        # Command (Matches your successful single run)
        cmd = [
            "python", "src/train_adversarial_image.py",
            "--image_path", img_path,
            "--xml_path", xml_path,
            "--model_id", "llava-hf/llava-1.5-7b-hf",
            "--num_iterations", "500",
            "--epsilon", "0.1",
            "--norm", "linf",
            "--learning_rate", "0.01",
            "--output_dir", single_output_dir,
            "--evaluate",
            "--lambda_v", "0.01",
            "--attention_aggregation", "flow",
            "--num_layers", "16",
            "--mapping_path", mapping_path
        ]
        
        try:
            # Capture output to debug if needed
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Read JSON result
            eval_json_path = os.path.join(single_output_dir, f"eval_{basename}.json")
            if os.path.exists(eval_json_path):
                with open(eval_json_path, 'r') as f:
                    data = json.load(f)
                
                det = data.get('object_detection', [{}])[0]
                desc = data.get('description', {})
                
                summary_data.append({
                    "Image_ID": basename,
                    "Object_Hidden_Success": det.get('attack_success', False),
                    "Description_Safe_Success": desc.get('attack_success', False),
                    "Target_Object": det.get('object_name', 'N/A'),
                    "Clean_Desc": desc.get('clean_response', ''),
                    "Adv_Desc": desc.get('adv_response', '')
                })
        except subprocess.CalledProcessError as e:
            print(f"\\n[ERROR] Failed on {basename}")
            print(e.stderr) # Print the actual python error
            continue

    # Stats & Graph
    if summary_data:
        # Save CSV
        csv_path = os.path.join(args.output_dir, "final_summary.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
            
        # Calc Accuracies
        total = len(summary_data)
        det_acc = sum(1 for x in summary_data if x['Object_Hidden_Success']) / total * 100
        desc_acc = sum(1 for x in summary_data if x['Description_Safe_Success']) / total * 100
        
        print(f"\\nResults (N={total}):")
        print(f"Object Detection Hiding Rate: {det_acc:.1f}%")
        print(f"Description Privacy Rate:     {desc_acc:.1f}%")
        
        # Plot
        plt.figure(figsize=(6, 5))
        metrics = ['Object Detection\\nHiding', 'Description\\nPrivacy']
        values = [det_acc, desc_acc]
        bars = plt.bar(metrics, values, color=['#2ca02c', '#1f77b4'], width=0.5)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height+1, f'{height:.1f}%', 
                     ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 110)
        plt.ylabel('Success Rate (%)')
        plt.title(f'Privacy Protection Results (N={total})')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, "success_rates_graph.png"), dpi=300)
        print("Graph saved.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
