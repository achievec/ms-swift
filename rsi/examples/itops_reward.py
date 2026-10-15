# Copyright (c) ModelScope Contributors. All rights reserved.
"""Scenario reward for the bundled IT-ops GRPO data.

``solution`` is a ``|`` separated keyword list. The reward is the fraction of
keywords that appear in the completion. ms-swift loads this file only when
``swift rlhf --external_plugins`` points here; the RSI mock backend never
imports it.
"""
from typing import List

from swift.rewards import ORM, orms


class ITOpsChecklist(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for completion, sol in zip(completions, solution):
            keys = [part.strip() for part in str(sol).split('|') if part.strip()]
            if not keys:
                rewards.append(0.0)
                continue
            text = completion if isinstance(completion, str) else str(completion)
            hit = sum(1 for key in keys if key in text)
            rewards.append(hit / len(keys))
        return rewards


orms['itops_checklist'] = ITOpsChecklist
