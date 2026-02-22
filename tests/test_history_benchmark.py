import tempfile
import unittest
from pathlib import Path

from benchmark_latest_vs_history import (
    SnapshotRef,
    choose_nearest_snapshot,
    infer_total_updates,
    neighbor_percent_points,
    parse_percent_points,
    split_matches,
)


class HistoryBenchmarkTests(unittest.TestCase):
    def test_parse_percent_points_deduplicates_and_preserves_order(self) -> None:
        pts = parse_percent_points("10, 20,20, 40, 10,90")
        self.assertEqual(pts, [10, 20, 40, 90])

    def test_parse_percent_points_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            parse_percent_points("0,10")
        with self.assertRaises(ValueError):
            parse_percent_points("10,101")

    def test_choose_nearest_snapshot_tie_breaks_earlier(self) -> None:
        snaps = [
            SnapshotRef(update=100, path=Path("snapshot_u000100.pt")),
            SnapshotRef(update=140, path=Path("snapshot_u000140.pt")),
        ]
        # Target 120 is equally distant; earlier update should be selected.
        chosen = choose_nearest_snapshot(snaps, 120)
        self.assertEqual(chosen.update, 100)

    def test_neighbor_percent_points_with_radius(self) -> None:
        self.assertEqual(neighbor_percent_points(10, 1), [9, 10, 11])
        self.assertEqual(neighbor_percent_points(1, 1), [1, 2])
        self.assertEqual(neighbor_percent_points(100, 2), [98, 99, 100])

    def test_split_matches_preserves_total(self) -> None:
        self.assertEqual(split_matches(10, 3), [4, 3, 3])
        self.assertEqual(split_matches(2, 3), [1, 1, 0])
        self.assertEqual(sum(split_matches(17, 5)), 17)

    def test_infer_total_updates_prefers_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            (out / "metrics.jsonl").write_text(
                "\n".join(
                    [
                        '{"update": 10, "avg_rank": 3.5}',
                        '{"update": 20, "avg_rank": 3.0}',
                        '{"update": 15, "avg_rank": 3.2}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            total = infer_total_updates(out, {"total_updates": 999})
            self.assertEqual(total, 999)


if __name__ == "__main__":
    unittest.main()
