import json
import urllib.request
import re
import argparse

def create_imagenet_mapping(output_file):
    """
    Downloads the ImageNet class mapping from the official URL and saves it to a JSON file.
    """
    url = "https://image-net.org/challenges/LSVRC/2014/browse-det-synsets"
    
    print(f"Downloading mapping from {url}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read().decode('utf-8')
            
        # The format is "wnid: label<br>"
        # We can use regex to find all occurrences
        # Pattern: n\d+ matches the wnid (e.g., n02084071)
        # followed by ": " and then the label until "<br>" or newline
        
        mapping = {}
        
        # Split by <br> or newlines to handle potential variations
        lines = re.split(r'<br>|\n', html)
        
        count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for pattern "nXXXXXXX: label"
            match = re.match(r'(n\d+):\s*(.+)', line)
            if match:
                wnid = match.group(1)
                label = match.group(2).strip()
                mapping[wnid] = label
                count += 1
            
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=4)
            
        print(f"Successfully created mapping file: {output_file}")
        print(f"Parsed {count} synsets.")
        
    except Exception as e:
        print(f"Error creating mapping file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ImageNet mapping file.")
    parser.add_argument("--output_path", type=str, default="imagenet_mapping.json", help="Path to save the mapping JSON")
    args = parser.parse_args()

    create_imagenet_mapping(args.output_path)
