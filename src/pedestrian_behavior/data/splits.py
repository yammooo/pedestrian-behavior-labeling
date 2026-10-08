"""One reproducible assignment per recording group, independent of labels."""

import math
import random


def split_groups(groups):
    groups = sorted(set(groups))
    random.Random(0).shuffle(groups)
    n = math.ceil(.15 * len(groups))
    return {group: "validation" if i < n else "test" if i < 2*n else "training"
            for i, group in enumerate(groups)}
