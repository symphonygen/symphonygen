import itertools
import os
from pathlib import Path
import random
import subprocess
from tqdm import tqdm
from arch.config import ASSET_DIR, DEBUG_REWARD
from utils.conventions import NO_HARMO_SUFFIXES
from utils.my_multiprocess import my_multiprocess
from utils.parseargs import parseargs
from utils.remove_harmo_outline import midi_remove_harmo_outline

# MuseScore Executable Configuration
# The extracted AppImage (see utils/headless_musescore.sh) is preferred on headless servers
if os.path.exists(f"{ASSET_DIR}/squashfs-root/AppRun"):
    print("Using headless MuseScore")
    MSCORE_EXEC = f"xvfb-run --auto-servernum {ASSET_DIR}/squashfs-root/AppRun"
else:
    MSCORE_EXEC = f"{ASSET_DIR}/MuseScore-3.6.2.548021370-x86_64.AppImage"

def process_convert_midi_to_audio(midi_path: Path, output_path: Path, clean_intermediate: bool=False):
    if midi_path.suffixes == NO_HARMO_SUFFIXES:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    no_harmo_midi_path = output_path.with_suffix(''.join(NO_HARMO_SUFFIXES))
    maybe_no_harmo_midi_path = midi_remove_harmo_outline(midi_path, no_harmo_midi_path)

    command = [*MSCORE_EXEC.split(), "-o", str(output_path), str(maybe_no_harmo_midi_path)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if DEBUG_REWARD:
            print(f"Error when converting {maybe_no_harmo_midi_path}:")
            print(e.stdout)
    if clean_intermediate:
        no_harmo_midi_path.unlink(missing_ok=True)

def main(
    midi_path: str, extension='mp3', recursive: bool=False,
    sample_per_cond: bool=False,
):
    midi_root = Path(midi_path)
    if midi_root.is_dir():
        midi_paths = sorted((midi_root.rglob if recursive else midi_root.glob)("*.mid"))
    else:
        midi_paths = [midi_root]
        midi_root = midi_root.parent

    if sample_per_cond:
        midi_paths = itertools.groupby(midi_paths, lambda x: x.parent)
        midi_paths = [random.choice(list(group)) for _, group in midi_paths]

    tasks: list[tuple[Path, Path]] = []
    for midi_path in tqdm(midi_paths):
        rel_path = midi_path.relative_to(midi_root)
        output_path = midi_root / extension / str(rel_path.with_suffix('.' + extension)).replace('/', '_')
        if output_path.exists():
            continue
        tasks.append((midi_path, output_path))

    my_multiprocess(process_convert_midi_to_audio, tasks)

if __name__ == "__main__":
    parseargs(main)
