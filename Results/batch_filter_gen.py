import json
from pathlib import Path
import random
import sys
import zipfile
from Results.config import (
    FORBID_PIANO, HARMO_MODEL_PATH, MODEL_CONFIGS, PICK_BEST_NUM, START_CHORDS,
)
from arch.config import VAL_DATA_PATH
from arch.dataset import MusicDataset
from arch.harmo.data_tensor import HarmonyTensorConverter
from arch.harmo.generator import HarmonyGenerator
from arch.harmo.model import HarmonyGPT
from arch.harmo.sampling import HarmPosConstrainer
from arch.symph import generate
from arch.symph.cond_gen import SymphonyGenerator3D
from arch.symph.data_tensor import MusicTensorConverter3D
from arch.symph.model import MusicModel3D
from arch.symph.sampling import DissonanceConstrainer
from data_prep.harmo_analysis import HarmonySkeletonAnalyzer
from data_prep.main import export_midi
from Results import evaluate
from rl.config import (
    HARMO_FILTER_LOG_PROB_HIGH, HARMO_FILTER_LOG_PROB_LOW,
    REF_FEATURE_DIR, RULES_FOR_HARMO_FILTER,
)
from rl.reward.harmo_filter import filter_harmony_by_rule
from utils.conventions import cond_dir_name, harmo_file_stem, song_file_stem, split_song_key
from utils.parseargs import parseargs

def get_dataset_harmo(exp_root: Path, num_requested: int):
    """
    Samples harmony skeletons analyzed from the validation set
    (the re-orchestration setting of the paper's second-round test).
    """
    val_dataset = MusicDataset(VAL_DATA_PATH)
    random.seed(42)
    num_candidates = min(len(val_dataset), num_requested * 15)
    harmo = []
    for i in random.sample(range(len(val_dataset)), num_candidates):
        bars = val_dataset[i]
        if any(bar.duration_q not in [16, 24, 32] for bar in bars):
            continue
        harmo.append(bars)
    harmo = filter_harmony_by_rule(harmo, num_requested)
    if len(harmo) < num_requested:
        print(f"Warning: only {len(harmo)} of {num_requested} requested "
              f"dataset skeletons passed the filter")

    for b, bars in enumerate(harmo):
        export_midi(bars, exp_root / "original" / f"{harmo_file_stem(b)}.mid")
        for bar in bars:
            bar.tracks = []

    harmo_analyzer = HarmonySkeletonAnalyzer(triad_match_strict=True)
    for bars in harmo:
        harmo_analyzer.purify_harmony_(bars)

    return harmo

def generate_harmo(num_requested: int):
    harmo_model = HarmonyGPT.from_pretrained(HARMO_MODEL_PATH).eval()
    harmo_converter = HarmonyTensorConverter()
    harmo_generator = HarmonyGenerator(harmo_converter)

    # Harmony is not at every quarter (e.g. time signature 6/8),
    # we did not account for it in later stages,
    # so we need to limit harmony to be at every quarter
    HarmPosConstrainer.limit_at_quarter = True

    if START_CHORDS:
        harmo = []
        for start_chord in START_CHORDS:
            print(f"Generating harmonies starting with {start_chord}...")
            this_chord_harmo = harmo_generator.generate_until_fulfilled(
                harmo_model, num_requested // len(START_CHORDS),
                max_batch_size=1024,
                start_chords_filter=[start_chord],
                filter=True,
                purify=True,
            )
            harmo.extend(this_chord_harmo)
    else:
        harmo = harmo_generator.generate_until_fulfilled(
            harmo_model, num_requested,
            filter=True,
            purify=True,
        )

    return harmo

def main(
    exp_dir: str,
    num_harmo: int=24,
    group_size: int=1,
    use_dataset_harmo: bool=False,
    skip_eval: bool=False,
    zip_mp3: bool=False,
):
    for model in MODEL_CONFIGS:
        if not Path(model["path"]).exists():
            raise ValueError(f"Model path {model['path']} does not exist")

    exp_root = Path(exp_dir)
    if exp_root.exists() and list(exp_root.iterdir()):
        raise ValueError(f"Experiment directory {exp_root} already exists and is not empty")
    exp_root.mkdir(parents=True, exist_ok=True)

    if use_dataset_harmo:
        harmo = get_dataset_harmo(exp_root, num_harmo)
    else:
        harmo = generate_harmo(num_harmo)

    # The conditioning skeletons: cond_harmony_<b> songs are conditioned on harmony_<b>.mid
    for b, bars in enumerate(harmo):
        export_midi(bars, exp_root / "harmony" / f"{harmo_file_stem(b)}.mid")

    converter = MusicTensorConverter3D()
    generate.FORBID_PIANO = FORBID_PIANO
    DissonanceConstrainer.enabled = True
    for model_config in MODEL_CONFIGS:
        model_root: Path = exp_root / model_config["label"]
        model_root.mkdir(parents=True, exist_ok=True)
        config = {
            "sys_argv": sys.argv,
            "model": model_config,
            "ref_feature_dir": REF_FEATURE_DIR,
            "harmony_filter_start_chords": START_CHORDS,
            "harmony_filter_rule": RULES_FOR_HARMO_FILTER,
            "harmony_filter_log_prob": [HARMO_FILTER_LOG_PROB_LOW, HARMO_FILTER_LOG_PROB_HIGH],
            "pick_best_num": PICK_BEST_NUM,
        }
        with open(model_root / "gen_config.json", "w") as f:
            json.dump(config, f, indent=4)

        model = MusicModel3D.from_pretrained(model_config["path"]).eval()
        generator = SymphonyGenerator3D(converter)
        DissonanceConstrainer.HN_weight = model_config["diss_HN"]
        DissonanceConstrainer.NN_weight = model_config["diss_NN"]
        DissonanceConstrainer.register_decay = model_config.get("register_decay", True)
        gen_bars_dict = generator.run(model, harmo, group_size=group_size)
        for b, gen_bars in gen_bars_dict.items():
            cond_id, song_id = split_song_key(b, group_size)
            save_path = model_root / cond_dir_name(cond_id) / f"{song_file_stem(song_id)}.mid"
            export_midi(gen_bars, save_path, harmo_is_cond=True)
        if not skip_eval:
            evaluate.main(model_root)

        if zip_mp3:
            # The rendered audio is left next to the MIDIs by the evaluation
            mp3_to_zip = sorted(model_root.rglob("song_*.mp3"))
            if not mp3_to_zip:
                print(f"Warning: no rendered audio found under {model_root} "
                      f"(--zip_mp3 requires the evaluation to have run)")
            with zipfile.ZipFile(exp_root / f"{model_config['label']}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr("gen_config.json", json.dumps(config, indent=4))
                for mp3 in mp3_to_zip:
                    arcname = str(mp3.relative_to(model_root)).replace("/", "_")
                    zipf.write(mp3, arcname)

if __name__ == "__main__":
    parseargs(main)
