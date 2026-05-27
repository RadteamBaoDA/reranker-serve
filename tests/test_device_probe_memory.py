"""The probe suggests the largest pairs-per-batch that keeps a memory margin."""

from src.engine.device_probe import suggest_max_batch_pairs


def test_picks_largest_candidate_within_margin():
    # free_fraction_after(pairs) returns the fraction of device memory that
    # would remain free after a batch of `pairs`. Margin 0.15 => need >= 0.15.
    free_after = {64: 0.60, 128: 0.40, 256: 0.20, 512: 0.05}
    chosen = suggest_max_batch_pairs(
        candidates=[64, 128, 256, 512],
        free_fraction_after=lambda p: free_after[p],
        safety_margin=0.15,
    )
    assert chosen == 256  # 512 would drop below the 15% margin


def test_falls_back_to_smallest_when_all_exceed_margin():
    chosen = suggest_max_batch_pairs(
        candidates=[64, 128],
        free_fraction_after=lambda p: 0.05,  # everything blows the margin
        safety_margin=0.15,
    )
    assert chosen == 64  # never return nothing; smallest is the safest attempt


def test_all_fit_returns_largest():
    chosen = suggest_max_batch_pairs(
        candidates=[64, 128, 256],
        free_fraction_after=lambda p: 0.5,
        safety_margin=0.15,
    )
    assert chosen == 256
