#!/usr/bin/env python3
"""
Utility to list available checkpoints and their structure.
Usage: python list_checkpoints.py [--model_type MODEL]
"""

import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="List available checkpoints")
    parser.add_argument(
        "--model_type",
        type=str,
        help="Filter to specific model type (llama, cola, fira, galore, relora, sltrain)"
    )
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("CHECKPOINTS/models-60m"),
        help="Checkpoint root directory"
    )
    
    args = parser.parse_args()
    
    checkpoint_root = Path(args.checkpoint_root).resolve()
    
    if not checkpoint_root.exists():
        print(f"❌ Checkpoint root not found: {checkpoint_root}")
        return 1
    
    # Map folder names to model types
    model_mapping = {
        "llama-60m": "llama",
        "cola-60m": "cola",
        "fira-60m": "fira",
        "galore-60m": "galore",
        "relora-60m": "relora",
        "sltrain-60m": "sltrain",
    }
    
    # Group checkpoints by model type
    checkpoints_by_type = defaultdict(list)
    
    for model_folder in sorted(checkpoint_root.iterdir()):
        if not model_folder.is_dir():
            continue
        
        folder_name = model_folder.name
        if folder_name not in model_mapping:
            continue
        
        model_type = model_mapping[folder_name]
        
        # Skip if filtering by model type
        if args.model_type and model_type != args.model_type:
            continue
        
        # Collect checkpoint names
        for checkpoint_dir in sorted(model_folder.iterdir()):
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("model_"):
                checkpoints_by_type[model_type].append(checkpoint_dir.name)
    
    # Print results
    print("📋 Available Checkpoints")
    print("=" * 60)
    print()
    
    total_checkpoints = 0
    
    for model_type in sorted(checkpoints_by_type.keys()):
        checkpoints = checkpoints_by_type[model_type]
        checkpoint_nums = sorted([int(c.split("_")[1]) for c in checkpoints])
        
        print(f"🔹 {model_type:15} ({len(checkpoints):2} checkpoints)")
        print(f"   Folder: {model_mapping.get(model_type + '-60m', '')}  ")
        
        # Print checkpoint names in ranges for readability
        ranges = []
        if checkpoint_nums:
            start = checkpoint_nums[0]
            end = checkpoint_nums[0]
            
            for num in checkpoint_nums[1:]:
                if num == end + 1:
                    end = num
                else:
                    if start == end:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{end}")
                    start = num
                    end = num
            
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
        
        print(f"   Steps : {', '.join(ranges)}")
        print()
        total_checkpoints += len(checkpoints)
    
    print("=" * 60)
    print(f"✅ Total: {total_checkpoints} checkpoints across {len(checkpoints_by_type)} model types")
    print()
    print("💡 Usage Examples:")
    print()
    print("  Run all checkpoints for specific model types:")
    print("    python loss-landscape/scripts/run_landscape_batch.py --only_models llama cola")
    print()
    print("  Run specific checkpoints across all models:")
    print("    python loss-landscape/scripts/run_landscape_batch.py \\")
    print("      --only_checkpoints model_1000 model_5000 model_10000")
    print()
    print("  Run subset by model and checkpoint:")
    print("    python loss-landscape/scripts/run_landscape_batch.py \\")
    print("      --only_models fira galore relora \\")
    print("      --only_checkpoints model_1000 model_2000 model_3000")
    print()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
