from pathlib import Path
from rl.base_class import Rewarder
from rl.reward.clamp3_reward import Clamp3Reward
from rl.reward.rule_based_reward import RuleBasedReward
from rl.reward.utils_ import weighted_tot_score

class CompositeReward(Rewarder):
    def __init__(
        self,
        clamp3_reward: Clamp3Reward,
        rule_based_reward: RuleBasedReward,
        rule_weight: float = 0.2,
    ):
        self.clamp3_reward = clamp3_reward
        self.rule_based_reward = rule_based_reward

        self.config = {
            "name": "composite",

            "clamp3": self.clamp3_reward.config,
            "rules": self.rule_based_reward.config,
            # The total reward is CLaMP 3 + rule_weight * track density.
            # Metrics computed with `rules_for_eval` are reported alongside
            # but never contribute to the reward ("default": 0.0).
            "weights": {
                "clamp3": 1.0,
                "track": rule_weight,
                "default": 0.0,
            },
        }

    def compute_reward_dict(self, rollout_root: str | Path, rel_names: list[str | Path], missing_ok=False):
        clamp3_rewards = self.clamp3_reward.compute_reward_dict(rollout_root, rel_names, missing_ok=missing_ok)
        rule_based_rewards = self.rule_based_reward.compute_reward_dict(rollout_root, rel_names, missing_ok=missing_ok)

        composite_dict = {}
        weights: dict = self.config["weights"]

        for rel_name in rel_names:
            rel_name = str(rel_name)

            # With missing_ok, the sub-rewarders have already warned about
            # the songs they skipped
            if rel_name not in clamp3_rewards or rel_name not in rule_based_rewards:
                continue
            clamp3 = clamp3_rewards[rel_name]
            this_rule_rewards = rule_based_rewards[rel_name]

            total_reward = weights["clamp3"] * clamp3 + weighted_tot_score(this_rule_rewards, weights)

            composite_dict[rel_name] = {
                "reward": total_reward,
                "clamp3": clamp3,
                **this_rule_rewards
            }

        return composite_dict
