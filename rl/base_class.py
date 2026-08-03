from pathlib import Path

class Rewarder:
    config: dict

    def compute_reward_dict(self, rollout_root: str | Path, rel_names: list[str | Path], missing_ok=False) -> dict[str, dict[str, float]]:
        """
        Rewarder can decide what parallelism to use.

        missing_ok: songs whose reward cannot be computed (failed
            generation/rendering) are skipped with a warning instead of raising.
        """
        raise NotImplementedError("Depends on reward function")

class GRPORollout:
    def run(self, model, rollout_dir, prompt_size, group_size):
        raise NotImplementedError

    def get_rel_names(self, prompt_size, group_size):
        """ relative name: relative path without extension """
        raise NotImplementedError

    @staticmethod
    def compute_completion_log_probs(model):
        raise NotImplementedError("Depends on model architecture")
