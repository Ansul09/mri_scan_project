import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def normalize(volume):
    """Normalize the volume to have values between 0 and 1"""
    volume = volume.astype(np.float32)
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val != min_val:
        volume = (volume - min_val) / (max_val - min_val)
    return volume

# --------------------------
# Define paths
# --------------------------
DATA_DIR = r"D:\projects\major-project\Brain_tumor_project\patient_list"
OUTPUT_DIR = r"D:\projects\major-project\Brain_tumor_project\preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✅ Preprocessed data will be saved in: {OUTPUT_DIR}")

# --------------------------
# Process Patients
# --------------------------
patients = [os.path.join(DATA_DIR, p) for p in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, p))]

for patient_path in tqdm(patients, desc="Processing patients"):
    patient = os.path.basename(patient_path)
    
    t1gd_path = None
    seg_path = None
    
    for fname in os.listdir(patient_path):
        lower_name = fname.lower()
        if "t1gd" in lower_name and fname.endswith(".nii.gz"):
            t1gd_path = os.path.join(patient_path, fname)
        elif "glistrboost" in lower_name and fname.endswith(".nii.gz"):
            seg_path = os.path.join(patient_path, fname)

    try:
        if t1gd_path is None or seg_path is None:
            raise FileNotFoundError("Missing T1GD or GLISTRboost file")
        
        # Load and normalize T1GD image
        t1gd_img = nib.load(t1gd_path).get_fdata()
        t1gd_img = normalize(t1gd_img)
        t1gd_img = np.expand_dims(t1gd_img, axis=-1)

        # Load segmentation mask
        seg = nib.load(seg_path).get_fdata().astype(np.uint8)

        # Save
        np.save(os.path.join(OUTPUT_DIR, f"{patient}_images.npy"), t1gd_img)
        np.save(os.path.join(OUTPUT_DIR, f"{patient}_mask.npy"), seg)

    except Exception as e:
        print(f"❌ Error processing {patient}: {e}")

print("🎉 Preprocessing complete. Files saved in:", OUTPUT_DIR)
