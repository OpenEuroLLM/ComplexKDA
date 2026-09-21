# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import numpy as np
import torch

from group_word_problems.a5_exact_tracker import binary_icosahedral, isomorphism
from group_word_problems.train_wordproblem import make_batch, perm_group


def test_group_batch_targets_are_running_products() -> None:
    _, table, identity = perm_group("s3")
    inputs, targets = make_batch(
        table,
        identity,
        batch=8,
        length=12,
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
    )
    accumulator = np.full(len(inputs), identity)
    for position in range(inputs.shape[1]):
        accumulator = table[inputs[:, position].numpy(), accumulator]
        np.testing.assert_array_equal(targets[:, position].numpy(), accumulator)


def test_a5_quaternion_table_is_isomorphic_to_permutation_table() -> None:
    _, permutation_table, permutation_identity = perm_group("a5")
    _, quaternion_table, quaternion_identity = binary_icosahedral()
    mapping = isomorphism(permutation_table, permutation_identity, quaternion_table, quaternion_identity)
    assert sorted(mapping.tolist()) == list(range(60))
    np.testing.assert_array_equal(
        mapping[permutation_table],
        quaternion_table[mapping[:, None], mapping[None, :]],
    )
