#!/usr/bin/env python3
"""
WSI Feature Download Utility for CHIMERA Task 3

Downloads WSI patch features and coordinates from the public S3 bucket
to enable local pipeline execution.

Usage:
    # Download all available WSI data
    python download_wsi_data.py --all
    
    # Download specific patients
    python download_wsi_data.py --patients 3A_001 3A_002 3A_003
    
    # Download only features (skip coordinates)
    python download_wsi_data.py --all --features-only
    
    # Check what's available without downloading
    python download_wsi_data.py --check-only
"""

import argparse
import sys
import os
import io
from pathlib import Path
from typing import List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# S3 Configuration
S3_BUCKET = "chimera-challenge"
S3_PREFIX = "v2/task3/"

# Local paths
PROJECT_ROOT = Path(__file__).parent
FEATURES_DIR = PROJECT_ROOT / "data" / "wsi_data" / "features"
COORDINATES_DIR = PROJECT_ROOT / "data" / "wsi_data" / "coordinates"


def get_s3_filesystem():
    """Initialize S3 filesystem with anonymous access."""
    try:
        import s3fs
        return s3fs.S3FileSystem(anon=True)
    except ImportError:
        logger.error("s3fs not installed. Run: pip install s3fs")
        sys.exit(1)


def list_available_patients(fs) -> List[str]:
    """List all patient IDs available in S3."""
    try:
        # List feature files
        feature_path = f"{S3_BUCKET}/{S3_PREFIX}features/features/"
        files = fs.ls(feature_path)
        
        patient_ids = []
        for f in files:
            filename = os.path.basename(f)
            if filename.endswith('_HE.pt'):
                pid = filename.replace('_HE.pt', '')
                patient_ids.append(pid)
        
        return sorted(patient_ids)
    except Exception as e:
        logger.error(f"Failed to list S3 contents: {e}")
        return []


def download_file(fs, s3_path: str, local_path: Path, description: str = "") -> bool:
    """Download a single file from S3."""
    try:
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already exists
        if local_path.exists():
            logger.debug(f"Skipping {description} (exists): {local_path.name}")
            return True
        
        logger.info(f"Downloading {description}: {local_path.name}")
        
        with fs.open(s3_path, 'rb') as remote:
            with open(local_path, 'wb') as local:
                local.write(remote.read())
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {local_path.name}: {e}")
        return False


def download_patient_data(fs, patient_id: str, 
                          download_features: bool = True,
                          download_coordinates: bool = True) -> dict:
    """Download all data for a single patient."""
    results = {'patient_id': patient_id, 'features': None, 'coordinates': None}
    
    if download_features:
        s3_path = f"{S3_BUCKET}/{S3_PREFIX}features/features/{patient_id}_HE.pt"
        local_path = FEATURES_DIR / f"{patient_id}_HE.pt"
        results['features'] = download_file(fs, s3_path, local_path, "features")
    
    if download_coordinates:
        s3_path = f"{S3_BUCKET}/{S3_PREFIX}features/coordinates/{patient_id}_HE.npy"
        local_path = COORDINATES_DIR / f"{patient_id}_HE.npy"
        results['coordinates'] = download_file(fs, s3_path, local_path, "coordinates")
    
    return results


def get_local_patient_ids() -> tuple:
    """Get patient IDs with local data."""
    features_local = set()
    coords_local = set()
    
    if FEATURES_DIR.exists():
        for f in FEATURES_DIR.glob("*_HE.pt"):
            features_local.add(f.stem.replace('_HE', ''))
    
    if COORDINATES_DIR.exists():
        for f in COORDINATES_DIR.glob("*_HE.npy"):
            coords_local.add(f.stem.replace('_HE', ''))
    
    return features_local, coords_local


def print_status(fs):
    """Print current data status."""
    logger.info("\n📊 Data Status Check")
    logger.info("=" * 50)
    
    # S3 available
    s3_patients = list_available_patients(fs)
    logger.info(f"S3 bucket available patients: {len(s3_patients)}")
    
    # Local available
    features_local, coords_local = get_local_patient_ids()
    logger.info(f"Local features downloaded: {len(features_local)}")
    logger.info(f"Local coordinates downloaded: {len(coords_local)}")
    
    # Missing
    features_missing = set(s3_patients) - features_local
    coords_missing = set(s3_patients) - coords_local
    
    logger.info(f"\nFeatures missing: {len(features_missing)}")
    logger.info(f"Coordinates missing: {len(coords_missing)}")
    
    if features_missing:
        logger.info(f"\nMissing feature IDs (first 10): {sorted(features_missing)[:10]}")
    
    return s3_patients, features_missing, coords_missing


def main():
    parser = argparse.ArgumentParser(
        description='Download WSI features from CHIMERA S3 bucket',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available patients'
    )
    
    parser.add_argument(
        '--patients',
        nargs='+',
        help='Specific patient IDs to download'
    )
    
    parser.add_argument(
        '--features-only',
        action='store_true',
        help='Download only features (skip coordinates)'
    )
    
    parser.add_argument(
        '--coords-only',
        action='store_true',
        help='Download only coordinates (skip features)'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Check status without downloading'
    )
    
    parser.add_argument(
        '--missing-only',
        action='store_true',
        help='Only download missing files'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel download workers (default: 4)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize S3
    fs = get_s3_filesystem()
    
    # Check only mode
    if args.check_only:
        print_status(fs)
        return
    
    # Determine what to download
    download_features = not args.coords_only
    download_coords = not args.features_only
    
    # Get patient list
    if args.patients:
        patient_ids = args.patients
    elif args.all or args.missing_only:
        s3_patients, features_missing, coords_missing = print_status(fs)
        
        if args.missing_only:
            # Only download what's missing
            patient_ids = sorted(features_missing | coords_missing)
        else:
            patient_ids = s3_patients
    else:
        parser.error("Specify --all, --missing-only, or --patients")
        return
    
    if not patient_ids:
        logger.info("No patients to download.")
        return
    
    logger.info(f"\n🚀 Starting download of {len(patient_ids)} patients...")
    logger.info(f"   Features: {'Yes' if download_features else 'No'}")
    logger.info(f"   Coordinates: {'Yes' if download_coords else 'No'}")
    logger.info(f"   Workers: {args.workers}")
    
    # Ensure directories exist
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    COORDINATES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download with parallel workers
    success_count = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_patient_data, 
                fs, pid, 
                download_features, 
                download_coords
            ): pid 
            for pid in patient_ids
        }
        
        for future in as_completed(futures):
            pid = futures[future]
            try:
                result = future.result()
                if all(v is None or v for v in [result['features'], result['coordinates']]):
                    success_count += 1
                else:
                    failed.append(pid)
            except Exception as e:
                logger.error(f"Error processing {pid}: {e}")
                failed.append(pid)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📦 Download Complete")
    logger.info(f"   Successful: {success_count}")
    logger.info(f"   Failed: {len(failed)}")
    
    if failed:
        logger.warning(f"   Failed IDs: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    
    # Final status
    print_status(fs)


if __name__ == "__main__":
    main()

