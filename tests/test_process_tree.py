from __future__ import annotations

from rheon.process_tree import activities_in_tree, estimate_variant_count, generate_process_tree, mutate_tree


def test_generate_process_tree_uses_configured_activity_bounds(small_config, rng):
    tree = generate_process_tree(small_config, rng)
    activities = activities_in_tree(tree)

    assert small_config.min_activities <= len(activities) <= small_config.max_activities
    assert set(activities).issubset(set(small_config.activity_pool))
    assert estimate_variant_count(tree) >= 1


def test_mutate_tree_records_cdlg_style_change_details(small_config, rng):
    tree = generate_process_tree(small_config, rng)
    mutation = mutate_tree(tree, small_config, rng, intensity=0.5)
    details = mutation.change_details()

    assert mutation.tree is not tree
    assert "activities_added" in details
    assert "activities_deleted" in details
    assert "activities_moved" in details
    assert "variant_count_before" not in details
    assert "variant_count_after" not in details
    assert (
        mutation.activities_added
        or mutation.activities_deleted
        or mutation.activities_moved
        or mutation.operator_swaps
    )
