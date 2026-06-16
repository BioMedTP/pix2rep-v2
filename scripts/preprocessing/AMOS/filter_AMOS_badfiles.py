import os
import nibabel as nib

root = "/path/to/amos22_unlabeled_ct_5900_6199"  # TODO: <-- UPDATE THIS PATH

bad_files = []

for fname in sorted(os.listdir(root)):
    if fname.startswith("._") or fname.startswith("."):
        continue

    if not fname.endswith(".nii.gz"):
        continue

    path = os.path.join(root, fname)

    try:
        img = nib.load(path)
        shape = img.shape  # header read only (fast)

        if len(shape) != 3:
            print(f"Not 3D: {fname} — shape={shape}")
            bad_files.append(fname)

        if img.shape[0] != img.shape[1]:
            print(f"Not square: {fname} — shape={shape}")
            bad_files.append(fname)

    except Exception as e:
        print(f"Failed to read: {fname} — {e}")
        bad_files.append(fname)

print("---- Summary ----")
print(f"Bad files: {len(bad_files)}")
