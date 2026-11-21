"""Exhaustive physical checks and an independent optimality oracle."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"content"))
from kompot_rl import Config, GlassEnv, shortest_distances, train, evaluate


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = GlassEnv()

    def test_states(self):
        self.assertEqual(len(self.env.states), 231)
        self.assertEqual(len(self.env.starts), 230)
        self.assertEqual(self.env.goal, (3, 3, 3, 3))
        self.assertEqual(self.env.n_actions, 1368)

    def test_all_transitions_conserve_volume_and_match_encoding(self):
        env = self.env
        for sid, actions in enumerate(env.actions):
            self.assertEqual(len(actions), len(set(actions)))
            for action, nxt in zip(actions, env.next_states[sid]):
                source, amounts = env.decode_action(action)
                self.assertEqual(env.encode_action(source, amounts), action)
                before, after = env.states[sid], env.states[int(nxt)]
                self.assertEqual(sum(after), 12)
                self.assertTrue(all(0 <= v <= 6 for v in after))
                self.assertEqual(before[source]-after[source], sum(amounts))
                self.assertGreater(sum(amounts), 0)
                for j, amount in zip([k for k in range(4) if k != source], amounts):
                    self.assertEqual(after[j]-before[j], amount)

    def test_bfs_matches_lower_bound_for_every_state(self):
        distances = shortest_distances(self.env)
        expected = [sum(v > self.env.target for v in s) for s in self.env.states]
        np.testing.assert_array_equal(distances, expected)
        self.assertEqual(max(distances), 3)

    def test_multi_recipient_is_one_lift_and_positive_reward(self):
        env = GlassEnv()
        env.reset((6, 2, 2, 2))
        action = env.encode_action(0, (1, 1, 1))
        self.assertEqual(action, 56)
        state, reward, terminal, truncated, info = env.step(action)
        self.assertEqual(state, (3, 3, 3, 3))
        self.assertEqual(reward, 1)
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertEqual(info["lifts"], 1)

    def test_truncation_is_not_success(self):
        env = GlassEnv(Config(max_steps=1))
        env.reset((6, 2, 2, 2))
        _, reward, terminal, truncated, _ = env.step(env.encode_action(0, (1, 0, 0)))
        self.assertEqual(reward, 0)
        self.assertFalse(terminal)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_invalid_action_does_not_mutate(self):
        env = GlassEnv()
        env.reset((6, 6, 0, 0))
        with self.assertRaises(ValueError):
            env.step(env.encode_action(0, (1, 0, 0)))
        self.assertEqual(env.state, (6, 6, 0, 0))
        self.assertEqual(env.steps, 0)

    def test_training_is_reproducible_and_rewards_nonnegative(self):
        env = GlassEnv(Config(capacity=2, total=4, episodes=1000, eval_every=500))
        for method in ("q_learning", "sarsa"):
            q, history = train(env, method, seed=5)
            q2, h2 = train(env, method, seed=5)
            self.assertEqual(history["return"], h2["return"])
            for a, b in zip(q, q2):
                np.testing.assert_array_equal(a, b)
            self.assertGreaterEqual(min(history["return"]), 0)
            self.assertTrue(set(history["raw_reward"]) <= {0.0, 1.0})
            self.assertEqual(evaluate(env, q)["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
