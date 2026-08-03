import json
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from arch.config import DEBUG_GRPO
from arch.data_tensor import DataTensorConverter
from arch.harmo.generator import HarmonyGenerator
from arch.symph.generator import SymphonyGenerator
from data_prep.main import deserialize_song, export_midi, serialize_song
from rl.base_class import GRPORollout
from utils.conventions import (
    ADVANTAGES_JSON, HARMO_TOKENS_JSON, TOKENS_JSON,
    cond_dir_name, harmo_file_stem, song_file_stem, song_key, split_song_key,
)
from utils.distributed import barrier, dist_gather_dict, is_main_process, rank, world_size

class SymphonyRollout(GRPORollout):
    def __init__(self, harmo_generator: HarmonyGenerator, generator: SymphonyGenerator):
        self.harmo_generator = harmo_generator
        self.generator = generator
        if DEBUG_GRPO:
            self.generator.converter.print_warnings = True
            self.harmo_generator.print_info = True
            self.generator.print_info = True
            self.generator.print_warning = True

    def run(
        self,
        harmo_model, model,
        rollout_dir,
        prompt_size=2,
        group_size=2,
    ):
        """
        rollout_dir/
            cond_harmony_{global_b1}/
                song_{b2}.mid
            harmo_tokens.json:
                harmony_{global_b1}: tokens
            tokens.json:
                cond_harmony_{global_b1}:
                    song_{b2}: tokens
        """
        if is_main_process():
            print(f"Rollout with {prompt_size=}, {group_size=}")

        rollout_root = Path(rollout_dir)
        rollout_root.mkdir(parents=True, exist_ok=True)

        harmo_model = harmo_model.module if isinstance(harmo_model, DDP) else harmo_model
        model = model.module if isinstance(model, DDP) else model

        local_prompt_size = prompt_size // world_size
        all_harmo_list = self.harmo_generator.generate_until_fulfilled(
            harmo_model,
            local_prompt_size,
            purify=True,
            filter=True,
        )
        all_song_dict = self.generator.run(
            model,
            conditions=all_harmo_list,
            group_size=group_size,
        )

        harmo_tokens = {}
        tokens = {}
        for b1, harmo_bars in enumerate(all_harmo_list):
            global_b1 = rank * local_prompt_size + b1
            cond_root = rollout_root / cond_dir_name(global_b1)
            cond_root.mkdir(parents=True, exist_ok=True)

            group_tokens = {}
            for b2 in range(group_size):
                b = song_key(b1, b2, group_size)
                if b not in all_song_dict:
                    continue
                bars = all_song_dict[b]

                save_path = cond_root / f"{song_file_stem(b2)}.mid"
                export_midi(bars, save_path, harmo_is_cond=True)
                group_tokens[song_file_stem(b2)] = serialize_song(bars)

            harmo_tokens[harmo_file_stem(global_b1)] = serialize_song(harmo_bars)
            tokens[cond_dir_name(global_b1)] = group_tokens

        harmo_tokens = dist_gather_dict(harmo_tokens)
        tokens = dist_gather_dict(tokens)

        if is_main_process():
            with open(rollout_root / HARMO_TOKENS_JSON, "w") as f:
                json.dump(harmo_tokens, f, indent=2)
            with open(rollout_root / TOKENS_JSON, "w") as f:
                json.dump(tokens, f, indent=2)
        # Other ranks read tokens.json right after (reward computation)
        barrier()

    def get_rel_names(self, prompt_size, group_size):
        local_prompt_size = prompt_size // world_size
        rel_names = []
        for b1 in range(local_prompt_size):
            global_b1 = rank * local_prompt_size + b1
            rel_names.extend([
                f"{cond_dir_name(global_b1)}/{song_file_stem(b2)}"
                for b2 in range(group_size)
            ])
        return rel_names

class GRPODataset(Dataset):
    def __init__(self, converter: DataTensorConverter):
        self.converter = converter
        self.tokens = []
        self.advantages = []

    def load_rollouts(self, rollout_dir):
        """
        rollout_dir/
            tokens.json
                cond_harmony_{global_b1}:
                    song_{b2}: tokens
            advantages.json
                advantages:
                    cond_harmony_{global_b1}/song_{b2}: advantage
        """
        rollout_root = Path(rollout_dir)
        with open(rollout_root / TOKENS_JSON) as f:
            tokens: dict[str, dict] = json.load(f)
        with open(rollout_root / ADVANTAGES_JSON) as f:
            advantages: dict = json.load(f)["advantages"]
        self.tokens.clear()
        self.advantages.clear()
        for cond, group_tokens in tokens.items():
            for song, song_tokens in group_tokens.items():
                rel_name = f"{cond}/{song}"
                if rel_name not in advantages:
                    if is_main_process():
                        print(f"Warning: {rel_name} not in advantages")
                    continue
                self.tokens.append(song_tokens)
                self.advantages.append(advantages[rel_name])

    def __len__(self):
        return len(self.advantages)

    def __getitem__(self, idx):
        song_tokens = self.tokens[idx]
        bars = deserialize_song(song_tokens)
        data = self.converter.pack_data_tensor(bars)
        advantage = self.advantages[idx]
        return data, advantage
