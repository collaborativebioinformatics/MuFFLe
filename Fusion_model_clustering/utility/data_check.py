import torch
import numpy as np
import s3fs
import io
import boto3
from botocore.client import Config
from botocore import UNSIGNED

# -------------------------------
# Configuration
# -------------------------------
BUCKET = "chimera-challenge"
PATIENT_ID = "3A_001"
PREFIX = "v2/task3/"

s3 = boto3.client(
    "s3",
    config=Config(signature_version=UNSIGNED)
)

fs = s3fs.S3FileSystem(anon=True)

def verify_spatial_alignment(patient_id):
    print(f"--- Verifying Spatial Alignment for: {patient_id} ---")
    
    # Paths
    pt_path = f"{BUCKET}/{PREFIX}features/features/{patient_id}_HE.pt"
    npy_path = f"{BUCKET}/{PREFIX}features/coordinates/{patient_id}_HE.npy"
    
    # 1. Load Features (.pt)
    with fs.open(pt_path, "rb") as f:
        features = torch.load(io.BytesIO(f.read()), map_location="cpu")
    
    # 2. Load Coordinates (.npy)
    with fs.open(npy_path, "rb") as f:
        # np.load works directly with s3fs file objects
        coords = np.load(f)
        
    # 3. Alignment Check
    num_features = features.shape[0]
    num_coords = coords.shape[0]
    
    print(f"Total Patch Embeddings: {num_features}")
    print(f"Total Patch Coordinates: {num_coords}")
    
    if num_features == num_coords:
        print("✅ SUCCESS: Features and Coordinates are perfectly aligned.")
        print(f"Sample Coord (X, Y): ({coords[0]['x']}, {coords[0]['y']})")
    else:
        print("❌ ERROR: Mismatch between features and coordinates!")

verify_spatial_alignment(PATIENT_ID)