import os
import shutil
from pathlib import Path


def copy_files_to_next_experiment_folder(target_directory):
    """
    Copies all files from the script directory to the next experiment folder in the target directory.
    :param target_directory: The target directory where the experiment folders are stored.
    :return: The path to the new experiment folder.
    """
    # Get the directory where the script file resides
    script_dir = Path(__file__).parent.resolve()

    # Ensure the target directory exists
    target_dir = Path(target_directory).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find the next experiment folder
    existing_experiment_folders = [
        d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("experiment ")
    ]
    next_experiment_number = (
        max(
            int(d.name.split()[-1]) for d in existing_experiment_folders
        ) + 1
        if existing_experiment_folders
        else 1
    )
    next_experiment_folder = target_dir / f"experiment {next_experiment_number}"
    next_experiment_folder.mkdir()

    # Copy all files from the script directory to the next experiment folder
    for file in script_dir.iterdir():
        if file.is_file():
            shutil.copy(file, next_experiment_folder / file.name)

    return next_experiment_folder
