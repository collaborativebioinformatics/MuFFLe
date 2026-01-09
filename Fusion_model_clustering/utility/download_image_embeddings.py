#!/usr/bin/env python3
"""
WSI Feature Download Utility for CHIMERA Task 3

Downloads WSI patch features and coordinates from the public S3 bucket.
Works on cloud instances (Brev, AWS, GCP) and local machines.

Usage:
    # Download all data (from project root)
    python utility/download_image_embeddings.py
    
    # Download with progress tracking
    python utility/download_image_embeddings.py --progress
    
    # Check status without downloading
    python utility/download_image_embeddings.py --check-only
    
    # Download in parallel (faster)
    python utility/download_image_embeddings.py --workers 8
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- Configuration ---
BUCKET_NAME = "chimera-challenge"
S3_PREFIX = "v2/task3/features/"
ALLOWED_EXTENSIONS = ('.pt', '.npy')

# Determine project root (parent of utility folder)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_ROOT = PROJECT_ROOT / "data" / "wsi_data"


def get_s3_client():
    """Initialize S3 client with anonymous access."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        
        return boto3.client('s3', config=Config(
            signature_version=UNSIGNED,
            connect_timeout=30,
            read_timeout=60,
            retries={'max_attempts': 3}
        ))
    except ImportError:
        print("❌ boto3 not installed. Run: pip install boto3")
        sys.exit(1)


def list_s3_objects(s3_client):
    """List all WSI objects in S3 bucket."""
    print(f"📋 Listing objects in s3://{BUCKET_NAME}/{S3_PREFIX}...")
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)
    
    objects = []
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            s3_key = obj['Key']
            if s3_key.endswith(ALLOWED_EXTENSIONS):
                objects.append({
                    'key': s3_key,
                    'size': obj['Size']
                })
    
    return objects


def get_local_status():
    """Check what files exist locally."""
    features_dir = LOCAL_ROOT / "features"
    coords_dir = LOCAL_ROOT / "coordinates"
    
    local_features = set()
    local_coords = set()
    
    if features_dir.exists():
        local_features = {f.name for f in features_dir.glob("*.pt")}
    if coords_dir.exists():
        local_coords = {f.name for f in coords_dir.glob("*.npy")}
    
    return local_features, local_coords


def download_file(s3_client, s3_key, local_path, progress=False):
    """Download a single file from S3."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        if local_path.exists():
            return {'status': 'skipped', 'file': local_path.name}
        
        s3_client.download_file(BUCKET_NAME, s3_key, str(local_path))
        return {'status': 'downloaded', 'file': local_path.name}
        
    except Exception as e:
        return {'status': 'failed', 'file': local_path.name, 'error': str(e)}


def download_all(s3_client, objects, workers=4, progress=True):
    """Download all files using thread pool."""
    total = len(objects)
    downloaded = 0
    skipped = 0
    failed = 0
    
    start_time = time.time()
    
    def process_object(obj):
        s3_key = obj['key']
        relative_path = s3_key.replace(S3_PREFIX, "")
        local_path = LOCAL_ROOT / relative_path
        return download_file(s3_client, s3_key, local_path)
    
    print(f"\n🚀 Starting download of {total} files with {workers} workers...")
    print(f"   Destination: {LOCAL_ROOT}")
    print("-" * 60)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_object, obj): obj for obj in objects}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            
            if result['status'] == 'downloaded':
                downloaded += 1
                if progress:
                    print(f"  [{i}/{total}] ✓ Downloaded: {result['file']}")
            elif result['status'] == 'skipped':
                skipped += 1
                if progress:
                    print(f"  [{i}/{total}] ○ Skipped (exists): {result['file']}")
            else:
                failed += 1
                print(f"  [{i}/{total}] ✗ Failed: {result['file']} - {result.get('error', 'Unknown')}")
    
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"\n📊 Download Summary:")
    print(f"   Downloaded: {downloaded}")
    print(f"   Skipped:    {skipped}")
    print(f"   Failed:     {failed}")
    print(f"   Time:       {elapsed:.1f} seconds")
    
    return downloaded, skipped, failed


def check_status(s3_client):
    """Print detailed status of local vs S3 data."""
    print("\n" + "=" * 60)
    print("DATA STATUS CHECK")
    print("=" * 60)
    
    # Get S3 objects
    objects = list_s3_objects(s3_client)
    
    s3_features = {Path(o['key']).name for o in objects if o['key'].endswith('.pt')}
    s3_coords = {Path(o['key']).name for o in objects if o['key'].endswith('.npy')}
    
    # Get local files
    local_features, local_coords = get_local_status()
    
    # Calculate differences
    missing_features = s3_features - local_features
    missing_coords = s3_coords - local_coords
    
    print(f"\n📦 S3 Bucket Status:")
    print(f"   Features (.pt):    {len(s3_features)} files")
    print(f"   Coordinates (.npy): {len(s3_coords)} files")
    
    total_size = sum(o['size'] for o in objects)
    print(f"   Total size:        {total_size / (1024**3):.1f} GB")
    
    print(f"\n💻 Local Status:")
    print(f"   Features (.pt):    {len(local_features)} files")
    print(f"   Coordinates (.npy): {len(local_coords)} files")
    print(f"   Location:          {LOCAL_ROOT}")
    
    print(f"\n📥 Missing Files:")
    print(f"   Features:    {len(missing_features)} files")
    print(f"   Coordinates: {len(missing_coords)} files")
    
    if missing_features:
        print(f"\n   Missing features (first 5): {sorted(missing_features)[:5]}")
    if missing_coords:
        print(f"   Missing coordinates (first 5): {sorted(missing_coords)[:5]}")
    
    # Estimate download size for missing files
    missing_size = sum(
        o['size'] for o in objects 
        if Path(o['key']).name in missing_features or Path(o['key']).name in missing_coords
    )
    print(f"\n   Estimated download: {missing_size / (1024**3):.1f} GB")
    
    print("=" * 60)
    
    return len(missing_features) + len(missing_coords)


def main():
    parser = argparse.ArgumentParser(
        description='Download CHIMERA WSI features from S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check status, do not download'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel download workers (default: 4)'
    )
    
    parser.add_argument(
        '--progress',
        action='store_true',
        default=True,
        help='Show progress for each file'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show summary, not individual files'
    )
    
    args = parser.parse_args()
    
    # Initialize S3 client
    s3_client = get_s3_client()
    
    # Check status
    missing_count = check_status(s3_client)
    
    if args.check_only:
        print("\n✅ Check complete. Use without --check-only to download.")
        return
    
    if missing_count == 0:
        print("\n✅ All files already downloaded!")
        return
    
    # Confirm download
    print(f"\n⚠️  This will download ~160 GB of data. Continue? [y/N]")
    try:
        response = input().strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return
    except EOFError:
        # Non-interactive mode (e.g., script), proceed
        pass
    
    # Get objects and download
    objects = list_s3_objects(s3_client)
    downloaded, skipped, failed = download_all(
        s3_client, 
        objects, 
        workers=args.workers,
        progress=not args.quiet
    )
    
    if failed == 0:
        print("\n✅ Download Complete! Data is organized in:")
        print(f"   {LOCAL_ROOT}")
    else:
        print(f"\n⚠️  Download completed with {failed} failures.")
        print("   Re-run the script to retry failed downloads.")


if __name__ == "__main__":
    main()
