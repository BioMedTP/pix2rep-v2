import os
import re

images_dir = "/path/to/AMOS/amos22/imagesTr"  # TODO: <-- UPDATE THIS PATH
output_file = "/path/to/AMOS/amos_ids_train.txt"  # TODO: <-- UPDATE THIS PATH

ids = []

for fname in os.listdir(images_dir):
    match = re.match(r"amos_(\d{4})\.nii\.gz", fname)
    if match:
        ids.append(match.group(1))

ids.sort()

with open(output_file, "w") as f:
    for i in ids:
        f.write(i + "\n")

print(f"Saved {len(ids)} IDs to {output_file}")
