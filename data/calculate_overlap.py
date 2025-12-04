import json
import os

BASE_DIR = r"c:\Users\Luke\Desktop\UF\senior-project\ml\data"

PAIRS = [
    (os.path.join(BASE_DIR, "train"), os.path.join(BASE_DIR, "train_2")),
    (os.path.join(BASE_DIR, "val"), os.path.join(BASE_DIR, "val_2")),
]

def load_json(path):
    with open(path, 'r') as f:
        return set(json.load(f))

def main():
    for dir1, dir2 in PAIRS:
        if not os.path.exists(dir1) or not os.path.exists(dir2):
            print(f"Skipping pair {dir1} -> {dir2} (directory not found)")
            continue

        print(f"Comparing {os.path.basename(dir1)} and {os.path.basename(dir2)}:")
        
        files1 = [f for f in os.listdir(dir1) if f.endswith('.json')]
        
        found_match = False
        for filename in files1:
            path1 = os.path.join(dir1, filename)
            path2 = os.path.join(dir2, filename)
            
            if os.path.exists(path2):
                found_match = True
                try:
                    set1 = load_json(path1)
                    set2 = load_json(path2)
                    
                    intersection = set1.intersection(set2)
                    len_intersection = len(intersection)
                    len1 = len(set1)
                    len2 = len(set2)
                    
                    pct1 = (len_intersection / len1 * 100) if len1 > 0 else 0
                    pct2 = (len_intersection / len2 * 100) if len2 > 0 else 0
                    
                    print(f"  File: {filename}")
                    print(f"    Intersection count: {len_intersection}")
                    print(f"    % of {os.path.basename(dir1)} ({len1}): {pct1:.2f}%")
                    print(f"    % of {os.path.basename(dir2)} ({len2}): {pct2:.2f}%")
                    print("-" * 40)
                except Exception as e:
                    print(f"    Error processing {filename}: {e}")
        
        if not found_match:
            print("  No matching JSON files found.")
        print("\n")

if __name__ == "__main__":
    main()
