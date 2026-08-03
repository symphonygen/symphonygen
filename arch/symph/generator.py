from pathlib import Path
import torch
from arch.base_class import Generator
from arch.config import BAR_NUM
from arch.config import *
from arch.symph.data_tensor import MusicTensorConverter3D
from arch.symph.model import MusicModel3D
from arch.symph.sampling import DissonanceConstrainer
from data_prep.data_structure import BarStruct
from data_prep.main import export_midi, preprocess_midi
from rl.config import MAX_BATCH_SIZE
from utils import midi2audio
from utils.conventions import song_file_stem, split_song_key
from utils.distributed import is_multi_gpu, rank
from utils.move_files import iter_path_or_dir
from utils.parseargs import parseargs

class SymphonyGenerator(Generator):
    converter: MusicTensorConverter3D

    def run(
        self, model,
        conditions: list[str | Path | list[BarStruct]],
        group_size: int = 2,
        analyze_harmo: bool = False,
        bar_offset: int = 0,
        save_dir: str | None = None,
    ):
        all_harmo_list = []
        for c, cond in enumerate(conditions):
            harmo_bars, warning = self.prep_harmony_bars(cond, analyze_harmo, bar_offset)
            all_harmo_list.append(harmo_bars)
            if self.print_warning and warning:
                if is_multi_gpu:
                    print(f"[Rank {rank}]", end=' ')
                print(f"Condition {c}: {warning}")

        # How many harmony conditions can we fit in one GPU pass?
        cond_batch_size = max(1, MAX_BATCH_SIZE // group_size)
        all_song_dict = {}
        for cond_offset in range(0, len(all_harmo_list), cond_batch_size):
            cur_harmo_list = all_harmo_list[cond_offset: cond_offset + cond_batch_size]
            with torch.no_grad():
                song_data = self.generate_given_harmony(model, cur_harmo_list, group_size)
            this_batch_dict = self.unpack_group(song_data)
            # `b` counts songs within this chunk, so the global song key of the
            # chunk's first song is `cond_offset * group_size` (see utils/conventions.py)
            for b, bars in this_batch_dict.items():
                all_song_dict[cond_offset * group_size + b] = bars

        for b, bars in all_song_dict.items():
            c, _ = split_song_key(b, group_size)
            self.postprocess(bars, all_harmo_list[c])

        if save_dir:
            self.save_gen(all_song_dict, save_dir)

        return all_song_dict

    def prep_harmony_bars(self, cond: str | Path | list[BarStruct], analyze_harmo: bool, bar_offset: int):
        if isinstance(cond, (str, Path)):
            harmo_bars = preprocess_midi(cond, analyze_harmo=analyze_harmo)
        else:
            harmo_bars = cond

        harmo_bars = harmo_bars[bar_offset:]

        # Length check
        bar_num = len(harmo_bars)
        if bar_num == 0:
            raise ValueError("Harmony is empty")
        elif bar_num < BAR_NUM:
            warning = f"Padding {bar_num} bars to {BAR_NUM}"
            harmo_bars += [BarStruct(duration_q=harmo_bars[-1].duration_q)] * (BAR_NUM - bar_num)
        elif bar_num > BAR_NUM:
            warning = f"Truncating {bar_num} bars to {BAR_NUM}"
            harmo_bars = harmo_bars[:BAR_NUM]
        else:
            warning = None

        return harmo_bars, warning

    def generate_given_harmony(self, model, all_harmo_list: list[list[BarStruct]], group_size: int):
        raise NotImplementedError("Depends on model architecture")

    def postprocess(self, bars: list[BarStruct], harmo_bars: list[BarStruct]):
        for bar, harmo_bar in zip(bars, harmo_bars):
            bar.duration_q = bar.duration_q or harmo_bar.duration_q
            if harmo_bar.harmony_track is None:
                bar.tracks.clear() # Enforce silence when harmony bar is empty
            else:
                # XXX: Shallow copy the harmony track from condition
                bar.harmony_track = harmo_bar.harmony_track
        if len(bars) < len(harmo_bars):
            bars.extend([BarStruct(harmo_bar.duration_q) for harmo_bar in harmo_bars[len(bars):]])
        elif len(bars) > len(harmo_bars):
            del bars[len(harmo_bars):]

    def save_gen(self, all_song_dict: dict, save_dir: str):
        save_root = Path(save_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        for b, bars in all_song_dict.items():
            save_path = save_root / f"{song_file_stem(b)}.mid"
            export_midi(bars, save_path, harmo_is_cond=True)
            if self.print_info:
                print(f"Saved generated song {b} to {save_path}")

def main(
    model_path: str,
    cond_midi_path_or_dir: str,
    analyze_harmo: bool=False,
    group_size: int=2,
    bar_offset: int=0,
    save_dir: str='.',
    export_audio: bool=False,
    forbid_piano: bool=False,
    disable_dissonance_averse: bool=False,
    hn_weight: float=DissonanceConstrainer.HN_weight,
    nn_weight: float=DissonanceConstrainer.NN_weight,
    register_decay: int=1,
    vis_dissonance: bool=False,
):
    from arch.symph import generate
    from arch.symph.cond_gen import SymphonyGenerator3D

    model = MusicModel3D.from_pretrained(model_path)
    model.eval()

    converter = MusicTensorConverter3D()
    generator = SymphonyGenerator3D(converter)
    if isinstance(cond_midi_path_or_dir, list):
        cond_paths = [Path(path) for path in cond_midi_path_or_dir]
    else:
        cond_paths = iter_path_or_dir(cond_midi_path_or_dir, "*.mid")

    generate.FORBID_PIANO = forbid_piano
    DissonanceConstrainer.enabled = not disable_dissonance_averse
    DissonanceConstrainer.HN_weight = hn_weight
    DissonanceConstrainer.NN_weight = nn_weight
    DissonanceConstrainer.register_decay = bool(register_decay)
    if vis_dissonance:
        DissonanceConstrainer.visualize = True
        DissonanceConstrainer.vis_dir = str(Path(save_dir) / "dissonance_vis")

    all_song_dict = generator.run(
        model,
        cond_paths,
        analyze_harmo=analyze_harmo,
        group_size=group_size,
        bar_offset=bar_offset,
    )
    save_root = Path(save_dir)
    for c, cond_path in enumerate(cond_paths):
        this_cond_songs = {
            split_song_key(b, group_size)[1]: bars
            for b, bars in all_song_dict.items()
            if split_song_key(b, group_size)[0] == c
        }
        cond_save_root = save_root / f"cond_{cond_path.stem}"
        generator.save_gen(this_cond_songs, cond_save_root)

        if export_audio:
            midi2audio.main(cond_save_root)

if __name__ == "__main__":
    parseargs(main)
