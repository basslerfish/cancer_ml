"""
Load data.
"""
from pathlib import Path

import nibabel as nib


def find_t1_and_gtv_files(folder: Path) -> tuple:
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


def find_sample_folders(source: Path) -> list:
    """
    Find all folders with T1 and GTV files in a directory.
    """
    sample_folders = []
    for element in source.iterdir():
        if element.is_dir():
            t1_file, gtv_file = find_t1_and_gtv_files(element)
            if (t1_file is not None) and (gtv_file is not None):
                sample_folders.append(element)
    return sample_folders



def load_images(t1_file: Path | str, gtv_file: Path | str) -> tuple:
    """Load T1 and GTV images when provided with paths."""
    assert isinstance(t1_file, (Path, str)), type(t1_file)
    t1_obj = nib.load(t1_file)
    t1_data = t1_obj.get_fdata()
    gtv_obj = nib.load(gtv_file)
    gtv_data = gtv_obj.get_fdata()
    return t1_data, gtv_data


def find_and_load_sample(sample_folder: Path) -> tuple:
    """Find files and load images together."""
    t1_file, gtv_file = find_t1_and_gtv_files(sample_folder)
    t1_data, gtv_data = load_images(t1_file, gtv_file)
    return t1_data, gtv_data


