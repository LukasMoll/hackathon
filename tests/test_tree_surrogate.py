import unittest

import numpy as np

from rl.tree_surrogate import (
    _apply_safety_fallback,
    _predict_action_with_scores,
    DecisionTreeParams,
    FeatureSchema,
    SimpleDecisionTreeClassifier,
    TronkscriptInferencePolicy,
    default_feature_schema,
    extract_features,
    generate_tronkscript,
)
from tronk_engine import TronkscriptParser
from tronk_ml import MLTronkEnv


class TreeSurrogateTests(unittest.TestCase):
    def test_tree_fits_simple_rule(self) -> None:
        rng = np.random.default_rng(123)
        x = rng.integers(-2, 3, size=(1200, 3), dtype=np.int16)
        y = np.zeros((x.shape[0],), dtype=np.int8)
        y[x[:, 0] <= 0] = -1
        y[(x[:, 0] > 0) & (x[:, 1] > 0)] = 1

        tree = SimpleDecisionTreeClassifier(
            DecisionTreeParams(
                max_depth=4,
                min_samples_split=8,
                min_samples_leaf=4,
                min_gain=0.0,
            )
        )
        tree.fit(x, y)
        pred = tree.predict(x)
        acc = float(np.mean(pred == y))
        self.assertGreater(acc, 0.95)

    def test_generated_tronkscript_parses(self) -> None:
        x = np.asarray(
            [
                [-2, 0, 0],
                [1, -1, 0],
                [2, 1, 0],
                [0, 0, 0],
                [2, -1, 1],
                [-1, 1, -1],
            ],
            dtype=np.int16,
        )
        y = np.asarray([-1, 0, 1, -1, 1, 0], dtype=np.int8)
        tree = SimpleDecisionTreeClassifier(
            DecisionTreeParams(max_depth=3, min_samples_split=2, min_samples_leaf=1, min_gain=0.0)
        )
        tree.fit(x, y)

        schema = FeatureSchema(tile_coords=((0, -1), (1, -1), (-1, 0)), include_turn=False, include_length=False)
        code = generate_tronkscript(tree, schema, include_main=True, wait_ticks=1000)
        parser = TronkscriptParser(code)
        program = parser.parse()
        self.assertIn("infer_action", program.functions)
        self.assertIn("main", program.functions)

    def test_tronkscript_inference_policy_matches_python_tree(self) -> None:
        schema = default_feature_schema(local_radius=1)
        # radius=1 -> 6 tile features; keep only first tile feature for deterministic split.
        schema = FeatureSchema(
            tile_coords=(schema.tile_coords[0],),
            include_turn=True,
            include_length=False,
            include_prev_action=True,
            length_cap=20,
        )

        x = np.asarray(
            [
                [-2, -1, 0],
                [-1, 0, 0],
                [0, 0, 0],
                [1, 0, 1],
                [2, 1, 1],
                [1, -1, -1],
                [0, 1, -1],
            ],
            dtype=np.int16,
        )
        y = np.asarray([-1, -1, 0, 1, 1, 1, 0], dtype=np.int8)
        tree = SimpleDecisionTreeClassifier(
            DecisionTreeParams(max_depth=3, min_samples_split=2, min_samples_leaf=1, min_gain=0.0)
        )
        tree.fit(x, y)

        lib_code = generate_tronkscript(tree, schema, include_main=False, wait_ticks=1000)
        ts = TronkscriptInferencePolicy(lib_code, infer_function_name="infer_action")
        env = MLTronkEnv(seed=5, max_steps=20, randomize_starts=True, randomize_facings=True, use_c_core=False)
        ts.reset(seed=11, max_steps=20, use_c_core=False)

        checks = 0
        while not env.done and checks < 40:
            actions = [0] * 6
            for pid in range(6):
                if not env.players[pid].alive:
                    continue
                feat = extract_features(env, pid, int(ts._runtimes[pid].globals.get("mem_prev_action", 0)), schema)
                py_action_raw, py_scores = _predict_action_with_scores(tree, feat)
                py_action = _apply_safety_fallback(env, pid, py_action_raw, py_scores)
                ts_action, _ = ts.act(env, pid)
                self.assertEqual(py_action, ts_action)
                actions[pid] = ts_action
                checks += 1
                if checks >= 40:
                    break
            env.step(actions)


if __name__ == "__main__":
    unittest.main()
