import os
from pathlib import Path
import random
import torch
from arch.base_class import Generator
from arch.config import DEBUG_REWARD
from arch.harmo.data_tensor import HarmonyTensorConverter
from arch.harmo.generate import generate_harmony
from arch.harmo.model import HarmonyGPT
from data_prep.data_structure import BarStruct
from data_prep.harmo_analysis import HarmonySkeletonAnalyzer
from data_prep.main import export_midi
from rl.config import HARMO_FILTER_LOG_PROB_HIGH, HARMO_FILTER_LOG_PROB_LOW, MAX_HARMO_BATCH_SIZE
from rl.reward.harmo_filter import filter_harmony_by_rule
from utils.conventions import harmo_file_stem
from utils.parseargs import parseargs

class HarmonyGenerator(Generator):
    converter: HarmonyTensorConverter

    def run(
        self, model: HarmonyGPT,
        num_batches: int = 1,
        batch_size: int = 2,
        start_chords_filter: list[str] | None = None,
        purify: bool = False,
        save_dir: str | None = None,
    ):
        all_harmo_dict = {}
        for offset in range(0, num_batches * batch_size, batch_size):
            with torch.no_grad():
                harmo_events = generate_harmony(
                    model.transformer,
                    batch_size=batch_size,
                    start_chords_filter=start_chords_filter,
                )
            this_batch_dict = self.unpack_group(harmo_events)
            for b, bars in this_batch_dict.items():
                if not bars:
                    # These are harmonies that don't have the requested start chords
                    continue
                all_harmo_dict[offset + b] = bars

        if purify:
            harmo_analyzer = HarmonySkeletonAnalyzer(triad_match_strict=True, print_info=self.print_info)
            for b, bars in all_harmo_dict.items():
                harmo_analyzer.purify_harmony_(bars)

        if save_dir:
            self.save_gen(all_harmo_dict, save_dir)

        return all_harmo_dict

    def generate_until_fulfilled(
        self, model: HarmonyGPT,
        num_requested: int,
        max_batch_size = MAX_HARMO_BATCH_SIZE,
        start_chords_filter: list[str] | None = None,
        purify: bool = False,
        filter: bool = False,
    ):
        cur_harmo = []
        while len(cur_harmo) < num_requested:
            harmo_dict = self.run(
                model,
                batch_size=max_batch_size if start_chords_filter or filter else num_requested,
                start_chords_filter=start_chords_filter,
            )
            this_batch = list(harmo_dict.values())

            if filter:
                this_batch = filter_harmony_by_rule(this_batch)
                this_batch = self.filter_by_log_prob(model, this_batch)

            if len(this_batch) > num_requested - len(cur_harmo):
                this_batch = random.sample(this_batch, num_requested - len(cur_harmo))
            cur_harmo.extend(this_batch)

        # Skeletons that pass the filters are pruned into pure template chords
        # (paper Sec. 4.2); the filters above see the skeletons as sampled
        if purify:
            harmo_analyzer = HarmonySkeletonAnalyzer(triad_match_strict=True, print_info=self.print_info)
            for bars in cur_harmo:
                harmo_analyzer.purify_harmony_(bars)

        return cur_harmo

    def filter_by_log_prob(
        self, model: HarmonyGPT,
        all_bars_list: list[list[BarStruct]],
        num_requested: int | None = None,
        low_threshold = HARMO_FILTER_LOG_PROB_LOW,
        high_threshold = HARMO_FILTER_LOG_PROB_HIGH,
    ):
        device = next(model.parameters()).device

        scored = []
        for bars in all_bars_list:
            events = self.converter.pack_data_tensor(bars)
            events = events[None, :].to(device)
            with torch.no_grad():
                log_prob = model.forward(events)["loss"].item()
            scored.append((bars, log_prob))

        good = [
            item for item, score in scored
            if low_threshold <= score <= high_threshold
        ]
        print(f"{len(good)} of {len(scored)} harmonies have log prob between {low_threshold} and {high_threshold}")
        if DEBUG_REWARD:
            print(f"Log probs: {[round(score, 2) for _, score in scored]}")

        if num_requested and len(good) > num_requested:
            good = random.sample(good, num_requested)
        return good

    def save_gen(self, all_harmo_dict: dict, save_dir: str):
        save_root = Path(save_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        for b, bars in all_harmo_dict.items():
            save_path = save_root / f"{harmo_file_stem(b)}.mid"
            export_midi(bars, save_path)
            if self.print_info:
                print(f"Saved generated harmony {b} to {save_path}")

def main(
    model_path: str,
    num_batches: int = 1,
    batch_size: int = 4,
    save_dir: str = ".",
):
    model = HarmonyGPT.from_pretrained(model_path)
    model.eval()

    converter = HarmonyTensorConverter()
    generator = HarmonyGenerator(converter)
    generator.run(
        model,
        num_batches=num_batches,
        batch_size=batch_size,
        save_dir=save_dir,
    )

if __name__ == "__main__":
    parseargs(main)
