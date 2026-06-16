import glob
import os
from pathlib import Path
import nibabel as nib

from pix2repv2.utils import utils

data_folder_path = os.getenv("MnMs_DATASET_FOLDER") + "/testing"

csv_path = os.path.join(
    Path(data_folder_path).parent,
    "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
)

patient_dirs = sorted(glob.glob(os.path.join(data_folder_path, "*")))

for p_dir in patient_dirs:
    pid = os.path.basename(p_dir)

    img_path = os.path.join(p_dir, f"{pid}_sa_cine.nii.gz")
    gt_path = os.path.join(p_dir, f"{pid}_sa_cine_gt.nii.gz")

    # skip missing files
    if not (os.path.exists(img_path) and os.path.exists(gt_path)):
        print(f"Missing files for {pid}, skipping")
        continue

    # # Step 1: rename original files
    # img_cine_path = os.path.join(p_dir, f"{pid}_sa_cine.nii.gz")
    # gt_cine_path = os.path.join(p_dir, f"{pid}_sa_cine_gt.nii.gz")

    # shutil.move(img_path, img_cine_path)
    # shutil.move(gt_path, gt_cine_path)

    # Step 2: load cine volumes
    img_nii = nib.load(img_path)
    gt_nii = nib.load(gt_path)

    img_data = img_nii.get_fdata()
    gt_data = gt_nii.get_fdata()
    # Step 3: get ED and ES indices
    ed_idx, es_idx = utils._parse_mnms_ed_es_idx(csv_path, pid)

    # Step 4: extract 3D volumes and save
    for frame, idx in zip(["ed", "es"], [ed_idx, es_idx]):
        img_3d = img_data[:, :, :, idx]  # keep 3D shape
        gt_3d = gt_data[:, :, :, idx]

        # preserve affine and zooms
        header_3d = img_nii.header.copy()
        header_3d.set_data_shape(img_3d.shape)  # set to 3D shape (X,Y,Z)
        header_3d.set_zooms(header_3d.get_zooms()[:3])  # drop T

        img_out_path = os.path.join(p_dir, f"{pid}_sa_{frame}.nii.gz")
        gt_out_path = os.path.join(p_dir, f"{pid}_sa_{frame}_gt.nii.gz")

        nib.save(
            nib.Nifti1Image(img_3d, affine=img_nii.affine, header=header_3d),
            img_out_path,
        )
        nib.save(
            nib.Nifti1Image(gt_3d, affine=gt_nii.affine, header=header_3d),
            gt_out_path,
        )

    print(f"Processed patient {pid}")
