import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
from tqdm import tqdm
from rl.config import REF_FEATURE_DIR
from rl.base_class import Rewarder
from rl.reward.extract_clamp3 import CLAMP3Extractor, process_convert_midi_to_audio
from rl.reward.utils_ import get_cosine_sim, load_ref_features
from arch.config import TMP_DIR
from utils.conventions import CLAMP3_CACHE_JSON, NO_HARMO_SUFFIXES
from utils.distributed import dist_gather_dict, is_main_process, rank
from utils.my_multiprocess import my_batch_convert

REF_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

class Clamp3Reward(Rewarder):
    config = {"name": "clamp3"}

    def __init__(self, ref_folder=REF_FEATURE_DIR, persist_audio=False):
        """
        persist_audio: keep the rendered audio next to the MIDI files
            (final evaluation, also required by `--zip_mp3`) instead of
            a temporary directory (training rollouts).
        """
        self.persist_audio = persist_audio
        refs = self.load_ref_features_or_encode(ref_folder)
        self.ref_centroid = refs.mean(axis=0)

    @staticmethod
    def load_ref_features_or_encode(ref_folder):
        """ Encode the reference audio set to .npy features on first use. """
        ref_root = Path(ref_folder)
        if not ref_root.is_dir():
            raise FileNotFoundError(
                f"CLaMP 3 reference set not found at {ref_root}. "
                f"Put your reference audio files there (see README step 2); "
                f"they are encoded on first use."
            )

        audio_todo = [
            path for path in sorted(ref_root.rglob("*"))
            if path.suffix.lower() in REF_AUDIO_EXTS and not path.with_suffix(".npy").exists()
        ]
        if audio_todo:
            print(f"Encoding {len(audio_todo)} reference audio files in {ref_root}...")
            for audio_path in audio_todo:
                feature_path = audio_path.with_suffix(".npy")
                # Write via a per-rank temp file + atomic rename, so concurrent
                # ranks (or an interrupted run) never leave a corrupt .npy
                tmp_path = feature_path.with_name(feature_path.name + f".tmp{rank}")
                with open(tmp_path, "wb") as f:
                    np.save(f, CLAMP3Extractor.extract(audio_path))
                os.replace(tmp_path, feature_path)

        if not any(ref_root.rglob("*.npy")):
            raise FileNotFoundError(
                f"No reference features (.npy) or audio files "
                f"({'/'.join(sorted(REF_AUDIO_EXTS))}) found in {ref_root}"
            )
        return load_ref_features(ref_root)

    def compute_reward_dict(self, folder: str | Path, rel_names: list[str | Path], missing_ok=False):
        root = Path(folder).resolve()
        rel_names = [Path(n) for n in rel_names]

        cache_path = root / CLAMP3_CACHE_JSON
        cached: dict[str, float] = {}
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)

        todo = [n for n in rel_names if str(n) not in cached]
        if cached and not todo:
            print(f"Using cached CLaMP 3 rewards for {root}")
        new_rewards = self.compute_rewards(root, todo) if todo else {}
        # NOTE: collective op, must be reached by every rank
        new_rewards = dist_gather_dict(new_rewards)
        if new_rewards:
            cached |= new_rewards
            if is_main_process():
                with open(cache_path, "w") as f:
                    json.dump(cached, f, indent=4)

        ret = {}
        missing = []
        for rel_name in rel_names:
            if str(rel_name) in cached:
                ret[str(rel_name)] = cached[str(rel_name)]
            else:
                missing.append(str(rel_name))
        if missing:
            msg = (f"No CLaMP 3 reward for {len(missing)}/{len(rel_names)} songs "
                   f"(audio rendering failed?), e.g. {missing[:3]}")
            if not missing_ok:
                raise KeyError(msg)
            print(f"Warning: {msg}")
        return ret

    def compute_rewards(self, root: Path, rel_names: list[Path]) -> dict[str, float]:
        midis = [root / rel_name.with_suffix('.mid') for rel_name in rel_names]
        valid_names = []
        features = []

        if self.persist_audio:
            cm = contextlib.nullcontext(root)
        else:
            cm = tempfile.TemporaryDirectory(prefix="mp3_", dir=TMP_DIR)
        with cm as audio_dir:
            audio_root = Path(audio_dir)

            my_batch_convert(process_convert_midi_to_audio, root, audio_root, midis, ".mp3", check_empty=True, retry=3)

            for rel_name in tqdm(rel_names, desc="Running Clamp3"):
                audio_path = audio_root / rel_name.with_suffix('.mp3')
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    feature = CLAMP3Extractor.extract(audio_path).flatten()
                    valid_names.append(rel_name)
                    features.append(feature)

        if features:
            features = np.stack(features)
            sims = get_cosine_sim(features, self.ref_centroid)
        else:
            sims = []

        return {
            str(rel_name): float(sim)
            for rel_name, sim in zip(valid_names, sims, strict=True)
        }

if __name__ == "__main__":
    folder = sys.argv[1]
    rel_names = [
        path.stem for path in sorted(Path(folder).glob("*.mid"))
        if path.suffixes != NO_HARMO_SUFFIXES
    ]
    rewards = Clamp3Reward(persist_audio=True).compute_reward_dict(folder, rel_names, missing_ok=True)
    infos = sorted((reward, file) for file, reward in rewards.items())
    print(f"Min reward: {infos[0][0]} for {infos[0][1]}")
    print(f"Max reward: {infos[-1][0]} for {infos[-1][1]}")
