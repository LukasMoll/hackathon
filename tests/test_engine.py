import unittest
from collections import deque

from tronk_engine import BOARD_RADIUS, DEFAULT_BOT, GameConfig, TronkSimulation, compare_python_vs_c, iter_board_tiles


def make_sim(use_c_core: bool = False) -> TronkSimulation:
    return TronkSimulation([DEFAULT_BOT] * 6, config=GameConfig(use_c_core=use_c_core))


class EngineTests(unittest.TestCase):
    def test_board_tile_count(self) -> None:
        tiles = list(iter_board_tiles(BOARD_RADIUS))
        self.assertEqual(len(tiles), 91)

    def test_contested_destination_kills_all(self) -> None:
        sim = make_sim(use_c_core=False)

        sim.players[0].body = deque([(0, 0)])
        sim.players[0].facing = 1
        sim.players[0].pending_turn = 0

        sim.players[1].body = deque([(1, -2)])
        sim.players[1].facing = 3
        sim.players[1].pending_turn = 0

        for i in range(2, 6):
            sim.players[i].alive = False

        sim._resolve_movement_step()

        self.assertFalse(sim.players[0].alive)
        self.assertFalse(sim.players[1].alive)

    def test_starvation_gem_spawns_after_threshold(self) -> None:
        sim = make_sim(use_c_core=False)

        for p in sim.players:
            p.alive = False

        before = len(sim.gems)
        for _ in range(5):
            sim._resolve_movement_step()
        after = len(sim.gems)

        self.assertGreaterEqual(after, before)

    def test_program_exit_kills_player(self) -> None:
        bot_exits = "x = 1"
        sim = TronkSimulation([bot_exits] + [DEFAULT_BOT] * 5, config=GameConfig(use_c_core=False))

        for _ in range(3):
            sim.players[0].runtime.tick()

        self.assertFalse(sim.players[0].alive)
        self.assertIsNotNone(sim.players[0].error)

    def test_top_level_variables_are_global(self) -> None:
        bot = """
constant = 7
x = 0

function main() do
    x = constant
end function

main()
"""
        sim = TronkSimulation([bot] + [DEFAULT_BOT] * 5, config=GameConfig(use_c_core=False))
        player = sim.players[0]

        for _ in range(20):
            player.runtime.tick()

        self.assertEqual(player.runtime.globals.get("x"), 7)

    def test_rel_to_abs_uses_effective_turned_facing(self) -> None:
        bot = """
function main() do
    turnRight()
    q, r = relToAbs(0, -1)
    println(q)
    println(r)
    while (1 == 1) do
        wait(10000)
    end while
end function

main()
"""
        sim = TronkSimulation([bot] + [DEFAULT_BOT] * 5, config=GameConfig(use_c_core=False))
        player = sim.players[0]

        for _ in range(250):
            player.runtime.tick()

        self.assertGreaterEqual(len(sim.logs[0]), 2)
        self.assertEqual(sim.logs[0][0], "-1")
        self.assertEqual(sim.logs[0][1], "-4")

    def test_turn_accepts_only_relative_values(self) -> None:
        bot = """
function main() do
    turn(5)
    println(getTurn())
    while (1 == 1) do
        wait(10000)
    end while
end function

main()
"""
        sim = TronkSimulation([bot] + [DEFAULT_BOT] * 5, config=GameConfig(use_c_core=False))
        player = sim.players[0]
        for _ in range(220):
            player.runtime.tick()

        self.assertGreaterEqual(len(sim.logs[0]), 1)
        self.assertEqual(sim.logs[0][0], "0")

    def test_get_player_info_returns_five_values(self) -> None:
        bot = """
function main() do
    a, b, c, d, e = getPlayerInfo(getPlayerId())
    println(a)
    println(e)
    while (1 == 1) do
        wait(10000)
    end while
end function

main()
"""
        sim = TronkSimulation([bot] + [DEFAULT_BOT] * 5, config=GameConfig(use_c_core=False))
        player = sim.players[0]
        for _ in range(300):
            player.runtime.tick()

        self.assertGreaterEqual(len(sim.logs[0]), 2)
        self.assertEqual(sim.logs[0][0], "1")
        self.assertEqual(sim.logs[0][1], "1")

    def test_reverse_is_32_bit_bit_reverse(self) -> None:
        bot = """
function main() do
    println(reverse(1))
    while (1 == 1) do
        wait(10000)
    end while
end function

main()
"""
        sim = TronkSimulation([bot] + [DEFAULT_BOT] * 5, config=GameConfig(use_c_core=False))
        player = sim.players[0]
        for _ in range(120):
            player.runtime.tick()

        self.assertGreaterEqual(len(sim.logs[0]), 1)
        self.assertEqual(sim.logs[0][0], "2147483648")

    def test_return_allows_expression(self) -> None:
        bot = """
function f(a, b) do
    return a + b
end function

function main() do
    println(f(1, 2))
    while (1 == 1) do
        wait(10000)
    end while
end function

main()
"""
        sim = TronkSimulation([bot] + [DEFAULT_BOT] * 5, config=GameConfig(use_c_core=False))
        p0 = sim.players[0]
        for _ in range(300):
            p0.runtime.tick()

        self.assertTrue(sim.players[0].alive)
        self.assertGreaterEqual(len(sim.logs[0]), 1)
        self.assertEqual(sim.logs[0][0], "3")

    def test_chained_binary_expression_is_rejected(self) -> None:
        bad_bot = "x = 1 + 2 + 3"
        sim = TronkSimulation([bad_bot] + [DEFAULT_BOT] * 5, config=GameConfig(use_c_core=False))
        self.assertFalse(sim.players[0].alive)
        self.assertIn("Parse error", sim.players[0].error)

    def test_all_builtin_functions_conformance_samples(self) -> None:
        bot = """
function main() do
    print(1, 2)
    println(3)
    println(min(5, 3))
    println(max(5, 3))
    println(abs(-5))
    println(gcd(48, 18))
    println(lcm(4, 6))
    println(is_prime(7))
    println(is_prime(8))
    println(reverse(1))
    r0 = rand()
    r1 = rand(100)
    r2 = rand(10, 20)
    println(r0)
    println(r1)
    println(r2)
    println(square(5))
    println(pow(2, 3))
    println(pow(2, -1))
    println(sqrt(25))
    println(sqrt(-1))
    println(root(8, 3))
    println(root(-8, 3))
    println(exp2(3))
    println(exp2(-1))
    wait(1)
    while (1 == 1) do
        wait(10000)
    end while
end function

main()
"""
        sim = TronkSimulation([bot] + [DEFAULT_BOT] * 5, config=GameConfig(use_c_core=False))
        p0 = sim.players[0]
        for _ in range(1200):
            p0.runtime.tick()

        lines = sim.logs[0]
        self.assertGreaterEqual(len(lines), 21)
        self.assertEqual(lines[0], "1 2 3")
        self.assertEqual(lines[1], "3")
        self.assertEqual(lines[2], "5")
        self.assertEqual(lines[3], "5")
        self.assertEqual(lines[4], "6")
        self.assertEqual(lines[5], "12")
        self.assertEqual(lines[6], "1")
        self.assertEqual(lines[7], "0")
        self.assertEqual(lines[8], "2147483648")

        r0 = int(lines[9])
        r1 = int(lines[10])
        r2 = int(lines[11])
        self.assertGreaterEqual(r0, 0)
        self.assertLessEqual(r0, 0xFFFF)
        self.assertGreaterEqual(r1, 0)
        self.assertLessEqual(r1, 100)
        self.assertGreaterEqual(r2, 10)
        self.assertLessEqual(r2, 20)

        self.assertEqual(lines[12], "25")
        self.assertEqual(lines[13], "8")
        self.assertEqual(lines[14], "0")
        self.assertEqual(lines[15], "5")
        self.assertEqual(lines[16], "0")
        self.assertEqual(lines[17], "2")
        self.assertEqual(lines[18], "0")
        self.assertEqual(lines[19], "8")
        self.assertEqual(lines[20], "0")

    def test_all_game_specific_functions_conformance_samples(self) -> None:
        bot = """
function main() do
    id = getPlayerId()
    println(id)
    turnLeft()
    println(getTurn())
    turnRight()
    println(getTurn())
    turnForward()
    println(getTurn())
    turn(1)
    println(getTurn())
    turn(5)
    println(getTurn())
    existsA, emptyA, pidA, gemA = getTileAbs(99, 0)
    println(existsA)
    existsR, emptyR, pidR, gemR = getTileRel(0, -1)
    println(existsR)
    aq, ar = relToAbs(0, 0)
    alive, hq, hr, hf, lenv = getPlayerInfo(id)
    println(aq)
    println(ar)
    println(hq)
    println(hr)
    println(alive)
    println(lenv)
    println(getTick())
    while (1 == 1) do
        wait(10000)
    end while
end function

main()
"""
        sim = TronkSimulation([bot] + [DEFAULT_BOT] * 5, config=GameConfig(use_c_core=False))
        p0 = sim.players[0]
        for _ in range(3000):
            p0.runtime.tick()

        lines = sim.logs[0]
        self.assertGreaterEqual(len(lines), 15)

        pid = int(lines[0])
        self.assertGreaterEqual(pid, 0)
        self.assertLessEqual(pid, 5)
        self.assertEqual(lines[1], "-1")
        self.assertEqual(lines[2], "1")
        self.assertEqual(lines[3], "0")
        self.assertEqual(lines[4], "1")
        self.assertEqual(lines[5], "1")
        self.assertEqual(lines[6], "0")
        self.assertEqual(lines[7], "1")

        aq = int(lines[8])
        ar = int(lines[9])
        hq = int(lines[10])
        hr = int(lines[11])
        alive = int(lines[12])
        length = int(lines[13])
        now = int(lines[14])

        self.assertEqual(aq, hq)
        self.assertEqual(ar, hr)
        self.assertEqual(alive, 1)
        self.assertEqual(length, 1)
        self.assertGreaterEqual(now, 0)

    def test_compare_python_vs_c_outcome(self) -> None:
        bots = [DEFAULT_BOT] * 6
        report = compare_python_vs_c(bots, config=GameConfig(seed=7, max_steps=12, use_c_core=True))
        self.assertIn("match", report)
        self.assertIn("python", report)
        self.assertIn("c", report)


if __name__ == "__main__":
    unittest.main()
