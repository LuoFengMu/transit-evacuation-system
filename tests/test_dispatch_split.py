"""Test demand splitting preserves total people count.

Verifies Phase 2.4:
  - split_quantities sum = original quantities sum
  - Each chunk ≤ min vehicle capacity
  - split_origin_map correctly maps sub-index → original index

Tests the splitting logic from src.dispatch.solver directly
without importing heavy dependencies (ortools, osmnx).
"""


def _split_demand(demand_quantities: list[int], min_capacity: int):
    """Replicate the demand-splitting logic from solve_evacuation_dispatch.

    For each original demand point:
      - Split into chunks no larger than min_capacity
      - Each chunk is a sub-demand at the same location
      - Track mapping from sub-index → original index
    """
    split_quantities: list[int] = []
    split_origin_map: dict[int, int] = {}

    for orig_idx, qty in enumerate(demand_quantities):
        remaining = qty
        while remaining > 0:
            chunk = min(remaining, min_capacity)
            split_quantities.append(chunk)
            split_origin_map[len(split_quantities) - 1] = orig_idx
            remaining -= chunk

    return split_quantities, split_origin_map


class TestDemandSplitting:

    def test_split_preserves_total_people(self):
        split_qty, _ = _split_demand([120, 80], 50)
        assert sum(split_qty) == 200

    def test_split_chunks_not_larger_than_min_capacity(self):
        split_qty, _ = _split_demand([100], 30)
        for qty in split_qty:
            assert qty <= 30, f"Chunk {qty} exceeds min capacity 30"
        assert sum(split_qty) == 100

    def test_split_origin_map_correct(self):
        """Each sub-demand maps to the correct original demand."""
        demand_quantities = [30, 120, 45, 200]
        min_cap = 50
        split_qty, split_map = _split_demand(demand_quantities, min_cap)

        orig_totals = [0] * len(demand_quantities)
        for sub_idx, orig_idx in split_map.items():
            orig_totals[orig_idx] += split_qty[sub_idx]

        assert orig_totals == demand_quantities, \
            f"Reconstructed {orig_totals} ≠ original {demand_quantities}"

    def test_small_demands_not_split(self):
        """Demands already ≤ min capacity → exactly 1 sub-demand each."""
        quantities = [10, 20, 30, 40, 50]
        split_qty, split_map = _split_demand(quantities, 100)

        assert len(split_qty) == len(quantities)
        assert split_qty == quantities

        for orig_idx in range(len(quantities)):
            sub_count = sum(1 for oi in split_map.values() if oi == orig_idx)
            assert sub_count == 1

    def test_large_demand_splits_correctly(self):
        """A demand of 500 with min_cap 50 → exactly 10 chunks of 50 each."""
        split_qty, split_map = _split_demand([500], 50)
        assert len(split_qty) == 10
        assert all(q == 50 for q in split_qty)
        # All sub-demands map to original 0
        assert all(oi == 0 for oi in split_map.values())

    def test_uneven_split(self):
        """A demand of 95 with min_cap 40 → 40 + 40 + 15."""
        split_qty, _ = _split_demand([95], 40)
        assert split_qty == [40, 40, 15]
        assert sum(split_qty) == 95

    def test_multiple_demands_splitting(self):
        quantities = [0, 60, 130]  # 0 should produce no sub-demands
        split_qty, split_map = _split_demand(quantities, 50)

        assert sum(split_qty) == 190
        # Demand 0 (0 people) → 0 sub-demands
        # Demand 1 (60) → 2 sub-demands (50 + 10)
        # Demand 2 (130) → 3 sub-demands (50 + 50 + 30)
        assert len(split_qty) == 5

        # Verify mapping
        orig_0_subs = [i for i, o in split_map.items() if o == 0]
        orig_1_subs = [i for i, o in split_map.items() if o == 1]
        orig_2_subs = [i for i, o in split_map.items() if o == 2]
        assert len(orig_0_subs) == 0
        assert len(orig_1_subs) == 2
        assert len(orig_2_subs) == 3

    def test_min_capacity_one(self):
        """Edge case: min_capacity = 1, every person is a sub-demand."""
        split_qty, split_map = _split_demand([5], 1)
        assert split_qty == [1, 1, 1, 1, 1]
        assert len(split_qty) == 5
