import shutil
import re
from pathlib import Path
from parse import parse
from utils.parseargs import parseargs

def copy_files_by_pattern(input_file_pattern: str, output_file_pattern: str, dry=False):
    """
    Generalized function to copy files by extracting named parameters
    from a source path and injecting them into a destination path.
    """
    # 1. Identify all placeholders in the input pattern (e.g., {ckpt_name})
    placeholders = re.findall(r"\{(\w+)\}", input_file_pattern)

    # 2. Create a glob-compatible string by replacing {var} with *
    # This allows us to search the filesystem efficiently
    glob_pattern = input_file_pattern
    for ph in placeholders:
        glob_pattern = glob_pattern.replace(f"{{{ph}}}", "*")

    # 3. Find all files matching the general structure
    # We use the current directory as root for the glob
    search_root = Path(".")
    candidate_paths = sorted(search_root.glob(glob_pattern))

    if candidate_paths:
        print(f"Found {len(candidate_paths)} files matching the pattern {glob_pattern} in {search_root}")
    else:
        print(f"No files found matching the pattern {glob_pattern} in {search_root}")
        return

    copy_count = 0
    for path in candidate_paths:
        path_str = str(path)

        # 4. Extract values using the 'parse' library
        result = parse(input_file_pattern, path_str)

        if result:
            # 5. Generate destination path using extracted dictionary
            try:
                dest_str = output_file_pattern.format(**result.named)
                dest_path = Path(dest_str)
                if dest_path.exists():
                    continue

                # 6. Copy the file (shutil.copy2 preserves timestamps/metadata)
                if dry:
                    print(f"Debug: Copy {path_str} -> {dest_str}")
                    return

                dest_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(path_str, dest_str)
                copy_count += 1

            except KeyError as e:
                print(f"Skipping {path_str}: Missing required variable {e} for output pattern.")

    print(f"Successfully copied {copy_count} files.")

def iter_path_or_dir(path_or_dir: str, pattern: str="*"):
    path_obj = Path(path_or_dir)
    if path_obj.is_file():
        return [path_obj]
    if path_obj.is_dir():
        return sorted(path_obj.glob(pattern))
    raise ValueError(f"Path or directory not found: {path_or_dir}")

def main(input_pattern, output_pattern, dry: bool=False):
    """ Moving files are not supported, manually delete the original files after copying. """
    copy_files_by_pattern(input_pattern, output_pattern, dry=dry)

if __name__ == "__main__":
    parseargs(main)
