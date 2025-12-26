#!/usr/bin/env python3
"""Cleanup old checkpoint files and organize model directory."""

import argparse
import sys
from pathlib import Path

def cleanup_old_checkpoints(models_dir: Path, keep_final_only: bool = True) -> None:
    """Remove old checkpoint files.
    
    Args:
        models_dir: Directory containing model files
        keep_final_only: If True, keep only final models
    """
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        print(f"Directory not found: {models_dir}")
        return
    
    checkpoint_files = list(models_dir.glob("*_episode_*.pkl"))
    final_files = list(models_dir.glob("*_final.pkl"))
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    print(f"Found {len(final_files)} final model files")
    
    if keep_final_only and checkpoint_files:
        print(f"\nRemoving {len(checkpoint_files)} checkpoint files...")
        for checkpoint in checkpoint_files:
            try:
                checkpoint.unlink()
                print(f"  ✓ Removed {checkpoint.name}")
            except Exception as e:
                print(f"  ✗ Failed to remove {checkpoint.name}: {e}")
        
        print(f"\n✅ Cleanup complete!")
        print(f"Kept {len(final_files)} final models:")
        for final in final_files:
            size_mb = final.stat().st_size / (1024 * 1024)
            print(f"  - {final.name} ({size_mb:.2f} MB)")
    else:
        print("No action taken (use --cleanup to remove checkpoints)")


def analyze_model_directory(models_dir: Path) -> None:
    """Analyze model directory and print statistics.
    
    Args:
        models_dir: Directory containing model files
    """
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        print(f"Directory not found: {models_dir}")
        return
    
    all_files = list(models_dir.glob("*.pkl"))
    checkpoint_files = [f for f in all_files if "_episode_" in f.name]
    final_files = [f for f in all_files if f.name.endswith("_final.pkl")]
    
    total_size = sum(f.stat().st_size for f in all_files)
    checkpoint_size = sum(f.stat().st_size for f in checkpoint_files)
    final_size = sum(f.stat().st_size for f in final_files)
    
    print(f"\n{'='*60}")
    print(f"Model Directory Analysis: {models_dir.name}")
    print(f"{'='*60}")
    print(f"\nTotal files: {len(all_files)}")
    print(f"  - Final models: {len(final_files)}")
    print(f"  - Checkpoints: {len(checkpoint_files)}")
    
    print(f"\nDisk usage:")
    print(f"  - Total: {total_size / (1024**2):.2f} MB")
    print(f"  - Final models: {final_size / (1024**2):.2f} MB")
    print(f"  - Checkpoints: {checkpoint_size / (1024**2):.2f} MB")
    
    if checkpoint_files:
        savings = checkpoint_size / (1024**2)
        print(f"\n💡 Potential space savings: {savings:.2f} MB")
        print(f"   Run with --cleanup to remove checkpoint files")
    
    print(f"\n{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cleanup and analyze model directories")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Models directory to clean (default: data/models)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze models directory",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove checkpoint files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze and cleanup all subdirectories",
    )
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    
    # Get all subdirectories
    if args.all:
        subdirs = [d for d in models_dir.iterdir() if d.is_dir()]
    else:
        subdirs = [models_dir]
    
    for subdir in subdirs:
        if args.analyze or args.all:
            analyze_model_directory(subdir)
        
        if args.cleanup or args.all:
            cleanup_old_checkpoints(subdir, keep_final_only=True)


if __name__ == "__main__":
    main()
