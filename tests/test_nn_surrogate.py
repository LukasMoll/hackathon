import unittest

import numpy as np

from rl.nn_surrogate import (
    QuantizedCompactPolicy,
    apply_safety_fallback,
    generate_nn_tronkscript,
)
from rl.tree_surrogate import FeatureSchema, TronkscriptInferencePolicy, extract_features
from tronk_engine import TronkscriptParser
from tronk_ml import MLTronkEnv


class NNSurrogateTests(unittest.TestCase):
    def _schema(self) -> FeatureSchema:
        # 1 tile feature + turn + prev_action => input_dim=3
        return FeatureSchema(
            tile_coords=((0, -1),),
            include_turn=True,
            include_length=False,
            include_prev_action=True,
            include_player_info=False,
            length_cap=20,
        )

    def _model(self) -> QuantizedCompactPolicy:
        return QuantizedCompactPolicy(
            input_dim=3,
            hidden1=2,
            hidden2=2,
            scale=8,
            w1=np.asarray([[4, -2, 1], [-3, 2, 0]], dtype=np.int32),
            b1=np.asarray([0, 1], dtype=np.int32),
            w2=np.asarray([[3, 1], [1, 2]], dtype=np.int32),
            b2=np.asarray([0, -1], dtype=np.int32),
            w3=np.asarray([[2, -1], [1, 1], [-1, 2]], dtype=np.int32),
            b3=np.asarray([0, 0, 0], dtype=np.int32),
        )

    def test_generated_tronkscript_parses(self) -> None:
        schema = self._schema()
        model = self._model()
        code = generate_nn_tronkscript(model, schema, include_main=True, wait_ticks=1000, safety_fallback=True)
        parser = TronkscriptParser(code)
        program = parser.parse()
        self.assertIn("infer_action", program.functions)
        self.assertIn("main", program.functions)

    def test_tronkscript_inference_matches_quantized_policy(self) -> None:
        schema = self._schema()
        model = self._model()
        lib_code = generate_nn_tronkscript(model, schema, include_main=False, wait_ticks=1000, safety_fallback=True)
        ts = TronkscriptInferencePolicy(lib_code, infer_function_name="infer_action")
        env = MLTronkEnv(seed=7, max_steps=20, randomize_starts=True, randomize_facings=True, use_c_core=False)
        ts.reset(seed=11, max_steps=20, use_c_core=False)

        prev_actions = [0] * 6
        checks = 0
        while not env.done and checks < 40:
            actions = [0] * 6
            for pid in range(6):
                if not env.players[pid].alive:
                    continue
                feat = extract_features(env, pid, prev_actions[pid], schema)
                py_raw, py_scores = model.predict_with_scores_one(feat)
                py_action = apply_safety_fallback(env, pid, py_raw, py_scores)
                ts_action, _ = ts.act(env, pid)
                self.assertEqual(int(py_action), int(ts_action))
                actions[pid] = int(ts_action)
                prev_actions[pid] = int(ts_action)
                checks += 1
                if checks >= 40:
                    break
            env.step(actions)


if __name__ == "__main__":
    unittest.main()

