"""Unit tests for the recall math in measure_recall.py, on small synthetic
data with hand-computed expected values (independent of the real dataset)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from measure_recall import recall_stats, union_candidates

# Three non-singleton entities + one singleton.
GT = {
    "S1-1": frozenset({"S2-1", "S2-2", "S3-1"}),  # 3 true matches
    "S1-2": frozenset({"S2-3"}),  # 1 true match
    "S1-3": frozenset({"S3-2"}),  # 1 true match
    "S1-4": frozenset(),  # singleton
}


def test_recall_stats_perfect_recall():
    candidates = {
        "S1-1": {"S2-1", "S2-2", "S3-1", "S3-99"},  # extra candidate is fine
        "S1-2": {"S2-3"},
        "S1-3": {"S3-2"},
    }
    stats = recall_stats(GT, candidates)
    assert stats["found_pairs"] == 5
    assert stats["total_gt_pairs"] == 5
    assert stats["micro_recall"] == 1.0
    assert stats["macro_recall"] == 1.0
    assert stats["fully_covered_entities"] == 3
    assert stats["non_singleton_entities"] == 3
    assert stats["singleton_entities"] == 1
    assert stats["triggered_singletons"] == 0


def test_recall_stats_partial_recall():
    candidates = {
        "S1-1": {"S2-1"},  # found 1/3
        "S1-2": {"S2-3"},  # found 1/1
        "S1-3": set(),  # found 0/1
    }
    stats = recall_stats(GT, candidates)
    assert stats["found_pairs"] == 2
    assert stats["total_gt_pairs"] == 5
    assert stats["micro_recall"] == 2 / 5
    # macro = mean(1/3, 1/1, 0/1) = mean(0.333, 1.0, 0.0)
    assert abs(stats["macro_recall"] - (1 / 3 + 1.0 + 0.0) / 3) < 1e-9
    assert stats["fully_covered_entities"] == 1


def test_recall_stats_flags_spurious_singleton_candidates():
    gt_singleton_only = {"S1-4": frozenset()}
    candidates = {"S1-4": {"S2-999"}}
    stats = recall_stats(gt_singleton_only, candidates)
    assert stats["triggered_singletons"] == 1
    assert stats["singleton_entities"] == 1
    # singletons never contribute to the pair-recall numerator/denominator
    assert stats["total_gt_pairs"] == 0


def test_union_candidates_merges_channels():
    channel_a = {"S1-1": {"S2-1"}, "S1-2": {"S2-3"}}
    channel_b = {"S1-1": {"S2-2", "S3-1"}, "S1-3": {"S3-2"}}
    merged = union_candidates({"a": channel_a, "b": channel_b})
    assert merged["S1-1"] == {"S2-1", "S2-2", "S3-1"}
    assert merged["S1-2"] == {"S2-3"}
    assert merged["S1-3"] == {"S3-2"}


def test_union_candidates_excludes_named_channel():
    channel_a = {"S1-1": {"S2-1"}}
    channel_b = {"S1-1": {"S2-2"}}
    merged = union_candidates({"a": channel_a, "b": channel_b}, exclude="b")
    assert merged["S1-1"] == {"S2-1"}


def test_unique_contribution_is_zero_for_fully_redundant_channel():
    # channel_b finds nothing that channel_a doesn't already find
    channel_a = {"S1-2": {"S2-3"}, "S1-3": {"S3-2"}}
    channel_b = {"S1-2": {"S2-3"}}
    full = union_candidates({"a": channel_a, "b": channel_b})
    without_b = union_candidates({"a": channel_a, "b": channel_b}, exclude="b")
    full_found = recall_stats(GT, full)["found_pairs"]
    without_b_found = recall_stats(GT, without_b)["found_pairs"]
    assert full_found - without_b_found == 0


def test_unique_contribution_is_positive_for_essential_channel():
    channel_a = {"S1-2": {"S2-3"}}
    channel_b = {"S1-3": {"S3-2"}}  # the only channel that finds S1-3's match
    full = union_candidates({"a": channel_a, "b": channel_b})
    without_b = union_candidates({"a": channel_a, "b": channel_b}, exclude="b")
    full_found = recall_stats(GT, full)["found_pairs"]
    without_b_found = recall_stats(GT, without_b)["found_pairs"]
    assert full_found - without_b_found == 1


if __name__ == "__main__":
    module = sys.modules[__name__]
    tests = [obj for name, obj in vars(module).items() if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {test.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
