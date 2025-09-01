import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from tkinter import Tk, filedialog, simpledialog

def visualize_patient(patient_folder):
    # Find MRI file (t1ce or t1Gd)
    t1ce_path = None
    seg_path = None
    
    for file in os.listdir(patient_folder):
        file_lower = file.lower()
        if file_lower.endswith(".nii.gz"):
            if "t1ce" in file_lower or "t1gd" in file_lower:
                t1ce_path = os.path.join(patient_folder, file)
            elif "seg" in file_lower or "glistrboost" in file_lower:
                seg_path = os.path.join(patient_folder, file)

    if not t1ce_path or not seg_path:
        print(f"Skipping {patient_folder} (missing MRI or segmentation)")
        return

    # Load images
    t1ce_img = nib.load(t1ce_path).get_fdata()
    seg_img = nib.load(seg_path).get_fdata()

    # Choose middle axial slice
    slice_index = t1ce_img.shape[2] // 2
    t1ce_slice = t1ce_img[:, :, slice_index]
    seg_slice = seg_img[:, :, slice_index]

    # Plot with overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(t1ce_slice.T, cmap='gray', origin='lower')
    plt.imshow(np.ma.masked_where(seg_slice.T == 0, seg_slice.T), cmap='autumn', alpha=0.6, origin='lower')
    plt.title(f"{os.path.basename(patient_folder)} - MRI with Tumor Overlay (Slice {slice_index})")
    plt.axis('off')
    plt.show()

# --------------------------
# Folder Picker (Interactive)
# --------------------------
Tk().withdraw()  # Hide tkinter root window
patient_list_folder = filedialog.askdirectory(title="Select patient_list folder")

if not patient_list_folder:
    print("❌ No folder selected. Exiting...")
else:
    # Collect all patient folders
    patient_folders = [os.path.join(patient_list_folder, d) 
                       for d in os.listdir(patient_list_folder) 
                       if os.path.isdir(os.path.join(patient_list_folder, d))]

    print(f"Found {len(patient_folders)} patients.")

    # Ask how many patients to visualize
    num_patients = simpledialog.askinteger("Input", f"Enter number of patients to visualize (max {len(patient_folders)}):",
                                           minvalue=1, maxvalue=len(patient_folders))

    if num_patients:
        for folder in patient_folders[:num_patients]:
            visualize_patient(folder)
    else:
        print("❌ No number entered. Exiting...")
