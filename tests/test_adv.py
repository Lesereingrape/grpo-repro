from __future__ import annotations

import statistics

from grpore.adv import group_relative, mean_baseline


def test_group_relative_zero_mean_unit_std():
    advs = group_relative([0.0, 0.2, 1.0])
    assert abs(statistics.fmean(advs)) < 1e-9
    assert abs(statistics.pstdev(advs) - 1.0) < 1e-9


def test_group_relative_constant_group_is_flat():
    assert group_relative([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_group_relative_single_is_zero():
    assert group_relative([1.0]) == [0.0]


def test_group_relative_orders_by_reward():
    advs = group_relative([0.0, 1.0])
    assert advs[1] > advs[0]


def test_mean_baseline_shifts_by_running_average():
    advs, updated = mean_baseline([1.0, 1.0], running=1.0)
    assert advs == [0.0, 0.0]
    assert updated == 1.0


def test_mean_baseline_moves_toward_batch_mean():
    _, updated = mean_baseline([1.0, 1.0], running=0.0)
    assert 0.0 < updated < 1.0
