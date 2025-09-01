import os
import nibabel as nib
import matplotlib.pyplot as plt
import math
from tkinter import Tk, filedialog, simpledialog

def visualize_patient_dynamic(patient_folder):
    files = [f for f in os.listdir(patient_folder) if f.endswith('.nii.gz')]
    if not files:
        print(f"No .nii.gz files found in {patient_folder}")
        return

    volumes = []
    titles = []
    for f in files:
        file_path = os.path.join(patient_folder, f)
        img = nib.load(file_path)
        data = img.get_fdata()
        volumes.append(data)
        title = os.path.splitext(os.path.splitext(f)[0])[0]
        titles.append(title)
        print(f"Loaded {f} with shape {data.shape}")

    n = len(volumes)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(5 * cols, 5 * rows))
    for i, volume in enumerate(volumes):
        slice_z = volume.shape[2] // 2
        plt.subplot(rows, cols, i + 1)
        if 'seg' in titles[i].lower() or 'mask' in titles[i].lower():
            plt.imshow(volume[:, :, slice_z].T, cmap='jet', origin='lower')
        else:
            plt.imshow(volume[:, :, slice_z].T, cmap='gray', origin='lower')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle(f'Visualization for {os.path.basename(patient_folder)}',
                 fontsize=18, y=1.02)
    plt.show()

# --------------------------
# Folder Picker (Interactive)
# --------------------------
Tk().withdraw()  # Hide tkinter root window
patient_list_folder = filedialog.askdirectory(title="Select patient_list folder")

if not patient_list_folder:
    print("❌ No folder selected. Exiting...")
else:
    # Get all patient folders
    patient_folders = [os.path.join(patient_list_folder, d)
                       for d in os.listdir(patient_list_folder)
                       if os.path.isdir(os.path.join(patient_list_folder, d))]

    print(f"Found {len(patient_folders)} patient folders.")

    # Ask how many patients to visualize
    num_patients = simpledialog.askinteger("Input",
                                           f"Enter number of patients to visualize (max {len(patient_folders)}):",
                                           minvalue=1, maxvalue=len(patient_folders))

    if num_patients:
        for patient_folder in patient_folders[:num_patients]:
            visualize_patient_dynamic(patient_folder)
    else:
        print("❌ No number entered. Exiting...")
