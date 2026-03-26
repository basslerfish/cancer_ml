"""
Load data.
"""

from pathlib import Path

import nibabel as nib

def find_files(folder: Path) -> tuple:
    """
    Find T1 and GTV files.
    """
    t1_file = None
    gtv_file = None
    for element in folder.iterdir():
        if "t1c" in element.name:
            t1_file = element
        if "gtv" in element.name:
            gtv_file = element
    if t1_file is None:
        raise FileNotFoundError(f"{folder}: T1 file not found.")
    return t1_file, gtv_file

def load_images(folder: Path) -> tuple:
    """Load a single example."""
    t1_file, gtv_file = find_files(folder)
    t1_obj = nib.load(t1_file)
    t1_data = t1_obj.get_fdata()
    gtv_obj = nib.load(gtv_file)
    gtv_data = gtv_obj.get_fdata()
    return t1_data, gtv_data

