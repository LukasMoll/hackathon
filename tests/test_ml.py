import tempfile
import unittest
from pathlib import Path

from tronk_ml import MLSessionStore, MLTronkEnv


class MLEnvTests(unittest.TestCase):
    def test_step_and_queries(self) -> None:
        env = MLTronkEnv(seed=1, max_steps=50)

        snap = env.step(["forward", "forward", "forward", "forward", "forward", "forward"])
        self.assertEqual(snap["step"], 1)

        exists, is_empty, player_id, gem = env.call(0, "getTileRel", [0, -1])
        self.assertIn(exists, (0, 1))
        self.assertIn(is_empty, (0, 1))
        self.assertIsInstance(player_id, int)
        self.assertIn(gem, (0, 1))

        alive, q, r, facing, length = env.call(0, "getPlayerInfo", [0])
        self.assertIn(alive, (0, 1))
        self.assertIsInstance(q, int)
        self.assertIsInstance(r, int)
        self.assertIsInstance(facing, int)
        self.assertGreaterEqual(length, 1)

    def test_session_store_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MLSessionStore(Path(tmpdir))
            session = store.new_session(seed=7, max_steps=30)
            session.env.step(["left", "right", "forward", "left", "right", "forward"])

            path = store.save_session(session.session_id, "unit_test")
            self.assertTrue(path.exists())

            files = store.list_saved()
            self.assertEqual(len(files), 1)

            payload = store.load_saved(files[0])
            self.assertIn("snapshot", payload)
            self.assertEqual(payload["snapshot"]["step"], 1)

    def test_c_core_ml_env_parity(self) -> None:
        env_py = MLTronkEnv(seed=11, max_steps=40, use_c_core=False)
        env_c = MLTronkEnv(seed=11, max_steps=40, use_c_core=True)
        if env_c.core_mode != "c":
            self.skipTest("C core unavailable")

        action_seq = [
            ["forward", "left", "right", "forward", "left", "right"],
            ["left", "left", "forward", "right", "forward", "right"],
            ["right", "forward", "left", "left", "right", "forward"],
            ["forward", "forward", "forward", "forward", "forward", "forward"],
        ]

        for i in range(12):
            actions = action_seq[i % len(action_seq)]
            env_py.step(actions)
            env_c.step(actions)

            self.assertEqual(env_py.done, env_c.done)
            self.assertEqual(sorted(env_py.gems), sorted(env_c.gems))
            for pid in range(6):
                p_py = env_py.players[pid]
                p_c = env_c.players[pid]
                self.assertEqual(p_py.alive, p_c.alive)
                self.assertEqual(p_py.head, p_c.head)
                self.assertEqual(env_py.effective_facing(p_py), env_c.effective_facing(p_c))

if __name__ == "__main__":
    unittest.main()
