from __future__ import annotations

import math
import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

try:
    from tronk_ctypes import CCoreUnavailable, TronkCCore
except Exception:  # pragma: no cover - fallback when ctypes bridge is unavailable.
    CCoreUnavailable = RuntimeError  # type: ignore[assignment]
    TronkCCore = None  # type: ignore[assignment]

BOARD_RADIUS = 5
MOVE_INTERVAL = 10_000
NO_GEM_TICKS_THRESHOLD = 50_000

# Axial directions ordered clockwise.
DIRECTIONS: List[Tuple[int, int]] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
]

PLAYER_COLORS = [
    "#f5ea14",  # yellow
    "#19d9e5",  # cyan
    "#e600ff",  # magenta
    "#ff1f1f",  # red
    "#1aff1a",  # green
    "#1f2bff",  # blue
]

CORNER_SPAWNS: List[Tuple[int, int]] = [
    (0, -5),
    (5, -5),
    (5, 0),
    (0, 5),
    (-5, 5),
    (-5, 0),
]

CORNER_FACINGS: List[int] = [3, 4, 5, 0, 1, 2]

INITIAL_GEMS: List[Tuple[int, int]] = [
    (0, -2),
    (2, -2),
    (2, 0),
    (0, 2),
    (-2, 2),
    (-2, 0),
]

NUMBER_RE = re.compile(r"^(?:0b[01]+|0x[0-9a-fA-F]+|\d+)$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ParseError(Exception):
    pass


class RuntimeErrorTS(Exception):
    pass


class ReturnSignal(Exception):
    def __init__(self, values: Tuple[int, ...]):
        super().__init__("function return")
        self.values = values


@dataclass
class Expr:
    pass


@dataclass
class NumberExpr(Expr):
    value: int


@dataclass
class VarExpr(Expr):
    name: str


@dataclass
class UnaryExpr(Expr):
    op: str
    expr: Expr


@dataclass
class BinaryExpr(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class CallExpr(Expr):
    name: str
    args: List[Expr]


@dataclass
class Statement:
    pass


@dataclass
class AssignStmt(Statement):
    targets: List[str]
    expr: Expr


@dataclass
class CompoundAssignStmt(Statement):
    target: str
    op: str
    expr: Expr


@dataclass
class CallStmt(Statement):
    call: CallExpr


@dataclass
class IfStmt(Statement):
    condition: Expr
    then_block: List[Statement]
    elif_blocks: List[Tuple[Expr, List[Statement]]]
    else_block: List[Statement]


@dataclass
class WhileStmt(Statement):
    condition: Expr
    body: List[Statement]


@dataclass
class ForStmt(Statement):
    var: str
    start: Expr
    end: Expr
    body: List[Statement]


@dataclass
class ReturnStmt(Statement):
    values: List[Expr]


@dataclass
class FunctionDef:
    name: str
    params: List[str]
    body: List[Statement]


@dataclass
class Program:
    functions: Dict[str, FunctionDef]
    top_level: List[Statement]


@dataclass
class PlayerState:
    player_id: int
    color: str
    body: Deque[Tuple[int, int]]
    facing: int
    pending_turn: int = 0
    growth_pending: int = 0
    alive: bool = True
    died_tick: Optional[int] = None
    error: Optional[str] = None
    runtime: Optional["BotRuntime"] = None

    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


@dataclass
class GameConfig:
    board_radius: int = BOARD_RADIUS
    move_interval: int = MOVE_INTERVAL
    no_gem_ticks_threshold: int = NO_GEM_TICKS_THRESHOLD
    max_steps: int = 300
    seed: int = 1
    use_c_core: bool = True


@dataclass
class GameResult:
    winner_id: Optional[int]
    total_ticks: int
    steps_played: int
    frames: List[Dict[str, Any]]
    logs: Dict[int, List[str]]
    errors: Dict[int, str]


class TronkscriptParser:
    def __init__(self, source: str):
        self.lines: List[Tuple[int, str]] = []
        for lineno, raw in enumerate(source.splitlines(), start=1):
            text = raw.split("--", 1)[0].strip()
            if text:
                self.lines.append((lineno, text))
        self.idx = 0

    def parse(self) -> Program:
        functions: Dict[str, FunctionDef] = {}
        top_level: List[Statement] = []
        while self.idx < len(self.lines):
            _, text = self.lines[self.idx]
            if text.startswith("function "):
                fn = self._parse_function()
                if fn.name in functions:
                    raise ParseError(f"Duplicate function '{fn.name}'")
                functions[fn.name] = fn
            else:
                top_level.append(self._parse_statement())
        return Program(functions=functions, top_level=top_level)

    def _peek(self) -> Tuple[int, str]:
        if self.idx >= len(self.lines):
            raise ParseError("Unexpected end of file")
        return self.lines[self.idx]

    def _consume(self) -> Tuple[int, str]:
        item = self._peek()
        self.idx += 1
        return item

    def _parse_function(self) -> FunctionDef:
        lineno, text = self._consume()
        m = re.match(r"^function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*do$", text)
        if not m:
            raise ParseError(f"Line {lineno}: Invalid function declaration")
        name = m.group(1)
        params_text = m.group(2).strip()
        params = []
        if params_text:
            params = [p.strip() for p in params_text.split(",")]
            for p in params:
                if not IDENT_RE.match(p):
                    raise ParseError(f"Line {lineno}: Invalid parameter '{p}'")

        body = self._parse_block(end_tokens={"end function"})
        if self.idx >= len(self.lines) or self.lines[self.idx][1] != "end function":
            raise ParseError(f"Line {lineno}: Missing 'end function' for '{name}'")
        self.idx += 1
        return FunctionDef(name=name, params=params, body=body)

    def _parse_block(self, end_tokens: Iterable[str]) -> List[Statement]:
        out: List[Statement] = []
        end_set = set(end_tokens)
        while self.idx < len(self.lines):
            _, text = self.lines[self.idx]
            if text in end_set:
                break
            if text.startswith("else if ") or text == "else":
                break
            out.append(self._parse_statement())
        return out

    def _parse_statement(self) -> Statement:
        lineno, text = self._consume()

        if text.startswith("if "):
            return self._parse_if(lineno, text)
        if text.startswith("while "):
            return self._parse_while(lineno, text)
        if text.startswith("for "):
            return self._parse_for(lineno, text)
        if text.startswith("return"):
            return self._parse_return(lineno, text)

        m_compound = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(\+=|-=|\*=|/=)\s*(.+)$", text)
        if m_compound:
            return CompoundAssignStmt(
                target=m_compound.group(1),
                op=m_compound.group(2),
                expr=self._parse_expr(m_compound.group(3).strip()),
            )

        if "=" in text:
            left, right = text.split("=", 1)
            targets = [x.strip() for x in left.split(",")]
            if not targets or any(not IDENT_RE.match(t) for t in targets):
                raise ParseError(f"Line {lineno}: Invalid assignment target")
            return AssignStmt(targets=targets, expr=self._parse_expr(right.strip()))

        call = self._parse_call_expr(text)
        if call is not None:
            return CallStmt(call=call)

        raise ParseError(f"Line {lineno}: Could not parse statement '{text}'")

    def _parse_if(self, lineno: int, text: str) -> IfStmt:
        condition = self._extract_wrapped_expr(lineno, text, prefix="if", suffix="then")
        then_block = self._parse_block(end_tokens={"end if"})

        elif_blocks: List[Tuple[Expr, List[Statement]]] = []
        else_block: List[Statement] = []

        while self.idx < len(self.lines):
            line_no, line = self.lines[self.idx]
            if line == "end if":
                self.idx += 1
                return IfStmt(condition=condition, then_block=then_block, elif_blocks=elif_blocks, else_block=else_block)
            if line.startswith("else if "):
                self.idx += 1
                cond = self._extract_wrapped_expr(line_no, line, prefix="else if", suffix="then")
                block = self._parse_block(end_tokens={"end if"})
                elif_blocks.append((cond, block))
                continue
            if line == "else":
                self.idx += 1
                else_block = self._parse_block(end_tokens={"end if"})
                if self.idx >= len(self.lines) or self.lines[self.idx][1] != "end if":
                    raise ParseError(f"Line {lineno}: Missing 'end if'")
                self.idx += 1
                return IfStmt(condition=condition, then_block=then_block, elif_blocks=elif_blocks, else_block=else_block)
            raise ParseError(f"Line {line_no}: Invalid token in if block: '{line}'")

        raise ParseError(f"Line {lineno}: Missing 'end if'")

    def _parse_while(self, lineno: int, text: str) -> WhileStmt:
        condition = self._extract_wrapped_expr(lineno, text, prefix="while", suffix="do")
        body = self._parse_block(end_tokens={"end while"})
        if self.idx >= len(self.lines) or self.lines[self.idx][1] != "end while":
            raise ParseError(f"Line {lineno}: Missing 'end while'")
        self.idx += 1
        return WhileStmt(condition=condition, body=body)

    def _parse_for(self, lineno: int, text: str) -> ForStmt:
        m = re.match(r"^for\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)\s+to\s+(.+)\s*\)\s*do$", text)
        if not m:
            raise ParseError(f"Line {lineno}: Invalid for-loop syntax")
        var = m.group(1)
        start_expr = self._parse_expr(m.group(2).strip())
        end_expr = self._parse_expr(m.group(3).strip())
        body = self._parse_block(end_tokens={"end for"})
        if self.idx >= len(self.lines) or self.lines[self.idx][1] != "end for":
            raise ParseError(f"Line {lineno}: Missing 'end for'")
        self.idx += 1
        return ForStmt(var=var, start=start_expr, end=end_expr, body=body)

    def _parse_return(self, lineno: int, text: str) -> ReturnStmt:
        if text == "return":
            return ReturnStmt(values=[])
        m = re.match(r"^return\s+(.+)$", text)
        if not m:
            raise ParseError(f"Line {lineno}: Invalid return syntax")
        values = [self._parse_expr(part.strip()) for part in self._split_args(m.group(1).strip())]
        return ReturnStmt(values=values)

    def _extract_wrapped_expr(self, lineno: int, text: str, prefix: str, suffix: str) -> Expr:
        t = text[len(prefix) :].strip()
        if not t.endswith(suffix):
            raise ParseError(f"Line {lineno}: Expected '{suffix}'")
        t = t[: -len(suffix)].strip()
        if not (t.startswith("(") and t.endswith(")")):
            raise ParseError(f"Line {lineno}: Condition must be wrapped in parentheses")
        return self._parse_expr(t[1:-1].strip())

    def _parse_expr(self, text: str) -> Expr:
        text = self._strip_outer_parens(text.strip())
        if not text:
            raise ParseError("Empty expression")

        call = self._parse_call_expr(text)
        if call is not None:
            return call

        for op in ["==", "!=", ">=", "<=", ">", "<", "+", "-", "*", "/", "%", "&", "^", "|"]:
            split = self._split_binary(text, op)
            if split is not None:
                left, right = split
                expr = BinaryExpr(op=op, left=self._parse_expr(left), right=self._parse_expr(right))
                if self._count_binary_ops(expr) > 1:
                    raise ParseError(
                        "Binary expressions are limited to exactly two operands; split complex expressions"
                    )
                return expr

        if text[0] in ["-", "+", "~", "!"] and len(text) > 1:
            return UnaryExpr(op=text[0], expr=self._parse_expr(text[1:].strip()))

        if NUMBER_RE.match(text):
            if text.startswith("0b"):
                return NumberExpr(int(text, 2))
            if text.startswith("0x"):
                return NumberExpr(int(text, 16))
            return NumberExpr(int(text, 10))

        if IDENT_RE.match(text):
            return VarExpr(text)

        raise ParseError(f"Invalid expression '{text}'")

    def _count_binary_ops(self, expr: Expr) -> int:
        if isinstance(expr, BinaryExpr):
            return 1 + self._count_binary_ops(expr.left) + self._count_binary_ops(expr.right)
        if isinstance(expr, UnaryExpr):
            return self._count_binary_ops(expr.expr)
        if isinstance(expr, CallExpr):
            return sum(self._count_binary_ops(arg) for arg in expr.args)
        return 0

    def _parse_call_expr(self, text: str) -> Optional[CallExpr]:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", text)
        if not m:
            return None
        name = m.group(1)
        args_text = m.group(2).strip()
        args: List[Expr] = []
        if args_text:
            args = [self._parse_expr(arg.strip()) for arg in self._split_args(args_text)]
        return CallExpr(name=name, args=args)

    def _split_args(self, text: str) -> List[str]:
        parts: List[str] = []
        depth = 0
        start = 0
        for idx, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(text[start:idx])
                start = idx + 1
        parts.append(text[start:])
        return [p.strip() for p in parts if p.strip()]

    def _split_binary(self, text: str, op: str) -> Optional[Tuple[str, str]]:
        depth = 0
        op_len = len(op)
        i = 0
        while i <= len(text) - op_len:
            ch = text[i]
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth -= 1
                i += 1
                continue
            if depth == 0 and text[i : i + op_len] == op:
                left = text[:i].strip()
                right = text[i + op_len :].strip()
                if not left or not right:
                    return None
                # Do not parse unary minus as binary minus.
                if op == "-" and left == "":
                    return None
                return left, right
            i += 1
        return None

    def _strip_outer_parens(self, text: str) -> str:
        while text.startswith("(") and text.endswith(")"):
            depth = 0
            ok = True
            for idx, ch in enumerate(text):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and idx != len(text) - 1:
                        ok = False
                        break
            if ok:
                text = text[1:-1].strip()
            else:
                break
        return text


class Scope:
    def __init__(
        self,
        runtime: "BotRuntime",
        parent: Optional["Scope"] = None,
        *,
        global_scope: bool = False,
    ):
        self.runtime = runtime
        self.parent = parent
        self.global_scope = global_scope
        self.locals: Dict[str, int] = runtime.globals if global_scope else {}

    def get(self, name: str) -> int:
        if name in self.locals:
            return self.locals[name]
        if name in self.runtime.globals:
            return self.runtime.globals[name]
        return 0

    def set(self, name: str, value: int) -> None:
        if self.global_scope:
            self.runtime.globals[name] = value
            return
        if name in self.locals:
            self.locals[name] = value
            return
        if name in self.runtime.globals:
            self.runtime.globals[name] = value
            return
        self.locals[name] = value


class BotRuntime:
    def __init__(self, source: str, player_id: int, sim: "TronkSimulation"):
        parser = TronkscriptParser(source)
        self.program = parser.parse()
        self.player_id = player_id
        self.sim = sim
        self.globals: Dict[str, int] = {}
        self.cooldown = 0
        self.dead = False
        self.terminated = False
        self.logs: List[str] = []
        self._stdout_buffer = ""
        self.current_scope = Scope(runtime=self, global_scope=True)
        self.gen = self._run_top_level()

    def _run_top_level(self) -> Generator[int, None, None]:
        scope = Scope(runtime=self, global_scope=True)
        self.current_scope = scope
        try:
            yield from self._exec_block(self.program.top_level, scope)
        except ReturnSignal:
            # Return at top-level is treated as program termination.
            return

    @staticmethod
    def _div_trunc(a: int, b: int) -> int:
        if b == 0:
            return 0
        return int(a / b)

    @classmethod
    def _mod_trunc(cls, a: int, b: int) -> int:
        if b == 0:
            return 0
        return a - cls._div_trunc(a, b) * b

    def tick(self) -> None:
        if self.dead or self.terminated:
            return
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        while True:
            try:
                cost = next(self.gen)
            except StopIteration:
                self.terminated = True
                self.dead = True
                self._flush_stdout_buffer()
                self.sim.kill_player(self.player_id, "Program exited")
                return
            except Exception as exc:
                self.dead = True
                self._flush_stdout_buffer()
                self.sim.kill_player(self.player_id, f"Runtime error: {exc}")
                return

            if cost <= 0:
                continue

            self.cooldown = cost - 1
            return

    def _scalar(self, value: Any) -> int:
        if isinstance(value, tuple):
            if not value:
                return 0
            return int(value[0])
        return int(value)

    def _normalize_values(self, value: Any) -> Tuple[int, ...]:
        if isinstance(value, tuple):
            return tuple(int(v) for v in value)
        return (int(value),)

    def _truthy(self, value: Any) -> bool:
        return self._scalar(value) != 0

    def _exec_block(self, stmts: List[Statement], scope: Scope) -> Generator[int, None, None]:
        old_scope = self.current_scope
        self.current_scope = scope
        try:
            for stmt in stmts:
                if isinstance(stmt, AssignStmt):
                    value = yield from self._eval_expr(stmt.expr, scope)
                    values = self._normalize_values(value)
                    for idx, target in enumerate(stmt.targets):
                        v = values[idx] if idx < len(values) else 0
                        scope.set(target, int(v))
                    # Assignments cost 1 tick.
                    yield 1

                elif isinstance(stmt, CompoundAssignStmt):
                    left = scope.get(stmt.target)
                    right_val = yield from self._eval_expr(stmt.expr, scope)
                    right = self._scalar(right_val)
                    if stmt.op == "+=":
                        scope.set(stmt.target, left + right)
                    elif stmt.op == "-=":
                        scope.set(stmt.target, left - right)
                    elif stmt.op == "*=":
                        scope.set(stmt.target, left * right)
                    elif stmt.op == "/=":
                        scope.set(stmt.target, self._div_trunc(left, right))
                    else:
                        raise RuntimeErrorTS(f"Unsupported compound op '{stmt.op}'")
                    yield 1

                elif isinstance(stmt, CallStmt):
                    _ = yield from self._call(stmt.call, scope)

                elif isinstance(stmt, IfStmt):
                    cond = yield from self._eval_expr(stmt.condition, scope)
                    yield 1
                    if self._truthy(cond):
                        yield from self._exec_block(stmt.then_block, scope)
                        continue
                    done = False
                    for elif_cond, elif_block in stmt.elif_blocks:
                        c = yield from self._eval_expr(elif_cond, scope)
                        yield 1
                        if self._truthy(c):
                            yield from self._exec_block(elif_block, scope)
                            done = True
                            break
                    if done:
                        continue
                    if stmt.else_block:
                        yield from self._exec_block(stmt.else_block, scope)

                elif isinstance(stmt, WhileStmt):
                    while True:
                        cond = yield from self._eval_expr(stmt.condition, scope)
                        yield 1
                        if not self._truthy(cond):
                            break
                        yield from self._exec_block(stmt.body, scope)

                elif isinstance(stmt, ForStmt):
                    start_val = yield from self._eval_expr(stmt.start, scope)
                    end_val = yield from self._eval_expr(stmt.end, scope)
                    current = self._scalar(start_val)
                    end_num = self._scalar(end_val)
                    scope.set(stmt.var, current)
                    yield 1
                    while True:
                        check_value = scope.get(stmt.var)
                        cond = 1 if check_value <= end_num else 0
                        yield 1
                        if cond == 0:
                            break
                        yield from self._exec_block(stmt.body, scope)
                        scope.set(stmt.var, scope.get(stmt.var) + 1)
                        yield 1

                elif isinstance(stmt, ReturnStmt):
                    out: List[int] = []
                    for expr in stmt.values:
                        value = yield from self._eval_expr(expr, scope)
                        out.append(self._scalar(value))
                    yield 1
                    raise ReturnSignal(tuple(out))

                else:
                    raise RuntimeErrorTS(f"Unsupported statement: {type(stmt)}")
        finally:
            self.current_scope = old_scope

    def _eval_expr(self, expr: Expr, scope: Scope) -> Generator[int, None, Any]:
        if isinstance(expr, NumberExpr):
            return expr.value
        if isinstance(expr, VarExpr):
            return scope.get(expr.name)
        if isinstance(expr, UnaryExpr):
            val = yield from self._eval_expr(expr.expr, scope)
            x = self._scalar(val)
            if expr.op == "-":
                return -x
            if expr.op == "+":
                return abs(x)
            if expr.op == "~":
                return ~x
            if expr.op == "!":
                return 1 if x == 0 else 0
            raise RuntimeErrorTS(f"Unsupported unary op '{expr.op}'")
        if isinstance(expr, BinaryExpr):
            left_val = yield from self._eval_expr(expr.left, scope)
            right_val = yield from self._eval_expr(expr.right, scope)
            left = self._scalar(left_val)
            right = self._scalar(right_val)

            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                return self._div_trunc(left, right)
            if expr.op == "%":
                return self._mod_trunc(left, right)
            if expr.op == "&":
                return left & right
            if expr.op == "^":
                return left ^ right
            if expr.op == "|":
                return left | right
            if expr.op == "==":
                return 1 if left == right else 0
            if expr.op == "!=":
                return 1 if left != right else 0
            if expr.op == ">":
                return 1 if left > right else 0
            if expr.op == "<":
                return 1 if left < right else 0
            if expr.op == ">=":
                return 1 if left >= right else 0
            if expr.op == "<=":
                return 1 if left <= right else 0
            raise RuntimeErrorTS(f"Unsupported binary op '{expr.op}'")
        if isinstance(expr, CallExpr):
            return (yield from self._call(expr, scope))
        raise RuntimeErrorTS(f"Unsupported expression: {type(expr)}")

    def _call(self, call: CallExpr, scope: Scope) -> Generator[int, None, Any]:
        args: List[Any] = []
        for arg_expr in call.args:
            args.append((yield from self._eval_expr(arg_expr, scope)))
        arg_scalars = [self._scalar(v) for v in args]

        builtin = self._dispatch_builtin(call.name, arg_scalars)
        if builtin is not None:
            cost, result = builtin
            if cost > 0:
                yield cost
            return result

        if call.name not in self.program.functions:
            raise RuntimeErrorTS(f"Unknown function '{call.name}'")

        fn = self.program.functions[call.name]
        local_scope = Scope(runtime=self, parent=scope)

        for idx, param in enumerate(fn.params):
            local_scope.locals[param] = arg_scalars[idx] if idx < len(arg_scalars) else 0

        # User-defined call entry tick cost.
        yield 1

        try:
            yield from self._exec_block(fn.body, local_scope)
        except ReturnSignal as ret:
            return ret.values
        return ()

    def _dispatch_builtin(self, name: str, args: List[int]) -> Optional[Tuple[int, Any]]:
        # Logging / utility.
        if name == "print":
            text = " ".join(str(a) for a in args)
            if self._stdout_buffer:
                self._stdout_buffer = f"{self._stdout_buffer} {text}" if text else self._stdout_buffer
            else:
                self._stdout_buffer = text
            return (0, 0)
        if name == "println":
            text = " ".join(str(a) for a in args)
            if self._stdout_buffer:
                line = f"{self._stdout_buffer} {text}" if text else self._stdout_buffer
                self._stdout_buffer = ""
            else:
                line = text
            self.logs.append(line)
            self.sim.append_log(self.player_id, line)
            return (0, 0)

        if name == "min":
            if len(args) < 2:
                raise RuntimeErrorTS("min expects 2 arguments")
            return (1, min(args[0], args[1]))
        if name == "max":
            if len(args) < 2:
                raise RuntimeErrorTS("max expects 2 arguments")
            return (1, max(args[0], args[1]))
        if name == "abs":
            return (1, abs(args[0]) if args else 0)
        if name == "rand":
            if not args:
                return (1, self.sim.random.randint(0, 0xFFFF))
            if len(args) == 1:
                return (1, self.sim.random.randint(0, max(0, args[0])))
            lo, hi = args[0], args[1]
            if lo >= hi:
                return (1, 0)
            return (1, self.sim.random.randint(lo, hi))
        if name == "square":
            x = args[0] if args else 0
            return (1, x * x)
        if name == "pow":
            x = args[0] if len(args) > 0 else 0
            y = args[1] if len(args) > 1 else 0
            if y < 0:
                return (1, 0)
            if x == 0 and y == 0:
                return (1, 1)
            return (1, int(pow(x, y)))
        if name == "sqrt":
            x = args[0] if args else 0
            return (1, int(math.isqrt(max(0, x))))
        if name == "root":
            x = args[0] if len(args) > 0 else 0
            n = args[1] if len(args) > 1 else 2
            if n == 0:
                return (1, 0)
            if x < 0 or n < 0:
                return (1, 0)
            if x == 0:
                return (1, 0)
            value = int(math.floor(x ** (1.0 / n)))
            # Correct floating-point drift by adjusting neighbors.
            while (value + 1) ** n <= x:
                value += 1
            while value > 0 and value**n > x:
                value -= 1
            return (1, value)
        if name == "exp2":
            x = args[0] if args else 0
            if x < 0:
                return (1, 0)
            return (1, 1 << x)
        if name == "gcd":
            x = args[0] if len(args) > 0 else 0
            y = args[1] if len(args) > 1 else 0
            return (1, math.gcd(x, y))
        if name == "lcm":
            x = args[0] if len(args) > 0 else 0
            y = args[1] if len(args) > 1 else 0
            if x == 0 or y == 0:
                return (1, 0)
            return (1, abs(x * y) // math.gcd(x, y))
        if name == "is_prime":
            n = args[0] if args else 0
            if n < 2:
                return (1, 0)
            if n == 2:
                return (1, 1)
            if n % 2 == 0:
                return (1, 0)
            d = 3
            while d * d <= n:
                if n % d == 0:
                    return (1, 0)
                d += 2
            return (1, 1)
        if name == "reverse":
            n = args[0] if args else 0
            value = n & 0xFFFFFFFF
            rev = 0
            for _ in range(32):
                rev = (rev << 1) | (value & 1)
                value >>= 1
            return (1, rev)
        if name == "wait":
            n = args[0] if args else 0
            return (max(0, n), 0)

        # Tronk-specific API.
        if name == "getPlayerId":
            return (1, self.player_id)
        if name == "getTick":
            return (1, self.sim.tick)
        if name == "turnLeft":
            self.sim.set_player_turn(self.player_id, -1)
            return (100, 0)
        if name == "turnRight":
            self.sim.set_player_turn(self.player_id, 1)
            return (100, 0)
        if name == "turnForward":
            self.sim.set_player_turn(self.player_id, 0)
            return (100, 0)
        if name == "turn":
            desired = args[0] if args else 0
            self.sim.set_player_turn_raw(self.player_id, desired)
            return (100, 0)
        if name == "getTurn":
            return (10, self.sim.get_player_turn(self.player_id))
        if name == "relToAbs":
            q = args[0] if len(args) > 0 else 0
            r = args[1] if len(args) > 1 else 0
            aq, ar = self.sim.rel_to_abs(self.player_id, q, r)
            return (5, (aq, ar))
        if name == "getTileAbs":
            q = args[0] if len(args) > 0 else 0
            r = args[1] if len(args) > 1 else 0
            return (20, self.sim.get_tile_abs(q, r))
        if name == "getTileRel":
            q = args[0] if len(args) > 0 else 0
            r = args[1] if len(args) > 1 else 0
            aq, ar = self.sim.rel_to_abs(self.player_id, q, r)
            return (20, self.sim.get_tile_abs(aq, ar))
        if name == "getPlayerInfo":
            target_id = args[0] if args else -1
            return (50, self.sim.get_player_info(target_id))

        return None

    def _flush_stdout_buffer(self) -> None:
        if not self._stdout_buffer:
            return
        self.logs.append(self._stdout_buffer)
        self.sim.append_log(self.player_id, self._stdout_buffer)
        self._stdout_buffer = ""


class TronkSimulation:
    def __init__(self, bot_sources: Sequence[str], config: Optional[GameConfig] = None):
        if len(bot_sources) != 6:
            raise ValueError("Exactly 6 bot sources are required")
        self.config = config or GameConfig()
        self.random = random.Random(self.config.seed)
        self.c_core: Optional[TronkCCore] = None
        self.core_mode = "python"

        self.tick = 0
        self.step_count = 0
        self.logs: Dict[int, List[str]] = {i: [] for i in range(6)}
        self.errors: Dict[int, str] = {}

        self.players: List[PlayerState] = []
        for i in range(6):
            p = PlayerState(
                player_id=i,
                color=PLAYER_COLORS[i],
                body=deque([CORNER_SPAWNS[i]]),
                facing=CORNER_FACINGS[i],
            )
            self.players.append(p)

        self.gems = set(INITIAL_GEMS)
        self.no_gem_ticks = 0

        if self.config.use_c_core and TronkCCore is not None:
            try:
                self.c_core = TronkCCore()
                self.c_core.init()
                self.core_mode = "c"
                self._sync_from_c_core()
            except CCoreUnavailable:
                self.c_core = None
                self.core_mode = "python"

        for i, src in enumerate(bot_sources):
            try:
                runtime = BotRuntime(src, i, self)
                self.players[i].runtime = runtime
            except Exception as exc:
                self.kill_player(i, f"Parse error: {exc}")

        self.frames: List[Dict[str, Any]] = []
        self._capture_frame(events=["game_start"])

    def append_log(self, player_id: int, line: str) -> None:
        self.logs[player_id].append(line)

    def _sync_from_c_core(self) -> None:
        if self.c_core is None:
            return

        for pid in range(6):
            state = self.c_core.get_player_state(pid)
            p = self.players[pid]
            p.alive = bool(state["alive"])
            p.facing = int(state["facing"])
            p.pending_turn = int(state["pending"])
            p.body = deque(state["body"])
            # growth_pending is consumed inside the C step implementation and isn't part of public state.
            p.growth_pending = 0

        self.gems = set(self.c_core.get_gems())

    def alive_players(self) -> List[PlayerState]:
        return [p for p in self.players if p.alive]

    def is_inside(self, q: int, r: int) -> bool:
        s = -q - r
        limit = self.config.board_radius
        return max(abs(q), abs(r), abs(s)) <= limit

    def occupied_positions(self) -> Dict[Tuple[int, int], int]:
        occ: Dict[Tuple[int, int], int] = {}
        for p in self.players:
            for tile in p.body:
                occ[tile] = p.player_id
        return occ

    def wrap_facing(self, facing: int) -> int:
        return ((facing % 6) + 6) % 6

    def effective_facing(self, player: PlayerState) -> int:
        return self.wrap_facing(player.facing + player.pending_turn)

    def set_player_turn(self, player_id: int, rel_turn: int) -> None:
        if rel_turn < -1:
            rel_turn = -1
        if rel_turn > 1:
            rel_turn = 1
        self.players[player_id].pending_turn = rel_turn
        if self.c_core is not None:
            self.c_core.set_turn(player_id, rel_turn)

    def set_player_turn_raw(self, player_id: int, value: int) -> None:
        if value in (-1, 0, 1):
            self.players[player_id].pending_turn = value
            if self.c_core is not None:
                self.c_core.set_turn_raw(player_id, value)

    def get_player_turn(self, player_id: int) -> int:
        if self.c_core is not None:
            return self.c_core.get_turn(player_id)
        return self.players[player_id].pending_turn

    def rel_to_abs(self, player_id: int, q: int, r: int) -> Tuple[int, int]:
        if self.c_core is not None:
            return self.c_core.rel_to_abs(player_id, q, r)
        p = self.players[player_id]
        rq, rr = rotate_axial(q, r, self.effective_facing(p))
        hq, hr = p.head
        return hq + rq, hr + rr

    def get_tile_abs(self, q: int, r: int) -> Tuple[int, int, int, int]:
        if self.c_core is not None:
            return self.c_core.get_tile_abs(q, r)
        if not self.is_inside(q, r):
            return (0, 0, -1, 0)

        occupied = self.occupied_positions()
        tile = (q, r)
        if tile in occupied:
            return (1, 0, occupied[tile], 0)

        is_gem = 1 if tile in self.gems else 0
        is_empty = 1 if is_gem == 0 else 0
        return (1, is_empty, -1, is_gem)

    def get_player_info(self, target_id: int) -> Tuple[int, int, int, int, int]:
        if self.c_core is not None:
            return self.c_core.get_player_info(target_id)
        if target_id < 0 or target_id >= len(self.players):
            return (0, 0, 0, 0, 0)
        p = self.players[target_id]
        q, r = p.head
        return (
            1 if p.alive else 0,
            q,
            r,
            self.effective_facing(p),
            p.length,
        )

    def kill_player(self, player_id: int, reason: str) -> None:
        p = self.players[player_id]
        if not p.alive:
            return
        p.alive = False
        p.died_tick = self.tick
        p.error = reason
        self.errors[player_id] = reason
        if p.runtime is not None:
            p.runtime.dead = True
        if self.c_core is not None:
            self.c_core.mark_dead(player_id)

    def run(self) -> GameResult:
        max_total_ticks = self.config.max_steps * self.config.move_interval + 1

        while self.tick <= max_total_ticks:
            if len(self.alive_players()) <= 1:
                break

            if self.tick > 0 and self.tick % self.config.move_interval == 0:
                self._resolve_movement_step()
                if len(self.alive_players()) <= 1:
                    break
                if self.step_count >= self.config.max_steps:
                    break

            for player in self.players:
                if not player.alive:
                    continue
                runtime = player.runtime
                if runtime is None:
                    self.kill_player(player.player_id, "No runtime")
                    continue
                runtime.tick()

            self.tick += 1

        alive = self.alive_players()
        winner_id = alive[0].player_id if len(alive) == 1 else None
        return GameResult(
            winner_id=winner_id,
            total_ticks=self.tick,
            steps_played=self.step_count,
            frames=self.frames,
            logs=self.logs,
            errors=self.errors,
        )

    def _resolve_movement_step(self) -> None:
        if self.c_core is not None:
            self._resolve_movement_step_c()
            return

        self.step_count += 1
        events: List[str] = []
        occupied = self.occupied_positions()

        desired_face: Dict[int, int] = {}
        desired_pos: Dict[int, Tuple[int, int]] = {}

        for p in self.players:
            if not p.alive:
                continue
            move_face = self.wrap_facing(p.facing + p.pending_turn)
            dq, dr = DIRECTIONS[move_face]
            hq, hr = p.head
            destination = (hq + dq, hr + dr)
            desired_face[p.player_id] = move_face
            desired_pos[p.player_id] = destination

        doomed: Dict[int, str] = {}

        for pid, dest in desired_pos.items():
            if not self.is_inside(dest[0], dest[1]):
                doomed[pid] = "Hit wall"
                continue
            if dest in occupied:
                doomed[pid] = "Hit occupied tile"

        contested: Dict[Tuple[int, int], List[int]] = {}
        for pid, dest in desired_pos.items():
            if pid in doomed:
                continue
            contested.setdefault(dest, []).append(pid)

        for dest, pids in contested.items():
            if len(pids) > 1:
                for pid in pids:
                    doomed[pid] = f"Contested tile {dest}"

        consumed_gems: List[Tuple[int, int]] = []

        for pid, reason in doomed.items():
            self.kill_player(pid, reason)
            events.append(f"p{pid}_dead:{reason}")

        for p in self.players:
            if not p.alive:
                continue
            pid = p.player_id
            if pid in doomed:
                continue

            dest = desired_pos[pid]
            p.facing = desired_face[pid]
            p.pending_turn = 0
            p.body.appendleft(dest)

            if dest in self.gems:
                consumed_gems.append(dest)
                p.growth_pending += 1
                events.append(f"p{pid}_gem")

            if p.growth_pending > 0:
                p.growth_pending -= 1
            else:
                p.body.pop()

        gem_picked = len(consumed_gems) > 0
        if gem_picked:
            for gem in consumed_gems:
                if gem in self.gems:
                    self.gems.remove(gem)
            for _ in consumed_gems:
                self._spawn_gem()
            self.no_gem_ticks = 0
        else:
            self.no_gem_ticks += self.config.move_interval
            if self.no_gem_ticks >= self.config.no_gem_ticks_threshold:
                if self._spawn_gem():
                    events.append("extra_gem_spawn")

        self._capture_frame(events=events)

    def _resolve_movement_step_c(self) -> None:
        if self.c_core is None:
            return

        self.step_count += 1
        events: List[str] = []

        step_result = self.c_core.resolve_step()
        death_reasons = step_result["death_reasons"]
        gem_picked = step_result["gem_picked"]
        spawn_from_pickups = int(step_result["spawn_from_pickups"])
        spawn_from_starvation = int(step_result["spawn_from_starvation"])

        reason_map = {
            1: "Hit wall",
            2: "Hit occupied tile",
            3: "Contested tile",
        }

        for pid, reason_code in enumerate(death_reasons):
            if reason_code:
                reason = reason_map.get(reason_code, "C-core collision")
                self.kill_player(pid, reason)
                events.append(f"p{pid}_dead:{reason}")

        for pid, picked in enumerate(gem_picked):
            if picked:
                events.append(f"p{pid}_gem")

        spawn_requests = spawn_from_pickups + spawn_from_starvation
        for _ in range(spawn_requests):
            candidates = self.c_core.list_spawn_candidates()
            if not candidates:
                break
            q, r = self.random.choice(candidates)
            self.c_core.add_gem(q, r)

        if spawn_from_starvation > 0:
            events.append("extra_gem_spawn")

        self._sync_from_c_core()
        self._capture_frame(events=events)

    def _spawn_gem(self) -> bool:
        if self.c_core is not None:
            candidates = self.c_core.list_spawn_candidates()
            if not candidates:
                return False
            q, r = self.random.choice(candidates)
            ok = self.c_core.add_gem(q, r)
            self._sync_from_c_core()
            return ok

        occupied = set(self.occupied_positions().keys())
        candidates = [tile for tile in iter_board_tiles(self.config.board_radius) if tile not in occupied and tile not in self.gems]
        if not candidates:
            return False
        self.gems.add(self.random.choice(candidates))
        return True

    def _capture_frame(self, events: Optional[List[str]] = None) -> None:
        frame_players: List[Dict[str, Any]] = []
        for p in self.players:
            q, r = p.head
            frame_players.append(
                {
                    "id": p.player_id,
                    "alive": p.alive,
                    "q": q,
                    "r": r,
                    "facing": self.effective_facing(p),
                    "pendingTurn": p.pending_turn,
                    "length": p.length,
                    "color": p.color,
                    "body": [[x, y] for x, y in p.body],
                    "error": p.error,
                }
            )

        self.frames.append(
            {
                "tick": self.tick,
                "step": self.step_count,
                "players": frame_players,
                "gems": [[q, r] for q, r in sorted(self.gems)],
                "events": events or [],
            }
        )


def iter_board_tiles(radius: int) -> Iterable[Tuple[int, int]]:
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= radius:
                yield (q, r)


def rotate_axial(q: int, r: int, clockwise_steps: int) -> Tuple[int, int]:
    x = q
    z = r
    y = -x - z
    steps = ((clockwise_steps % 6) + 6) % 6
    for _ in range(steps):
        x, y, z = -z, -x, -y
    return x, z


DEFAULT_BOT = """-- Keep alive and go straight forever.
function main() do
    while (1 == 1) do
        wait(1000)
    end while
end function

main()
"""


def compare_python_vs_c(bot_sources: Sequence[str], config: Optional[GameConfig] = None) -> Dict[str, Any]:
    """Run both engines and report whether outcomes match."""
    base = config or GameConfig()
    py_cfg = GameConfig(
        board_radius=base.board_radius,
        move_interval=base.move_interval,
        no_gem_ticks_threshold=base.no_gem_ticks_threshold,
        max_steps=base.max_steps,
        seed=base.seed,
        use_c_core=False,
    )
    c_cfg = GameConfig(
        board_radius=base.board_radius,
        move_interval=base.move_interval,
        no_gem_ticks_threshold=base.no_gem_ticks_threshold,
        max_steps=base.max_steps,
        seed=base.seed,
        use_c_core=True,
    )

    py_result = TronkSimulation(bot_sources, config=py_cfg).run()
    c_sim = TronkSimulation(bot_sources, config=c_cfg)
    c_result = c_sim.run()

    summary_py = {
        "winner_id": py_result.winner_id,
        "total_ticks": py_result.total_ticks,
        "steps_played": py_result.steps_played,
        "errors": py_result.errors,
    }
    summary_c = {
        "winner_id": c_result.winner_id,
        "total_ticks": c_result.total_ticks,
        "steps_played": c_result.steps_played,
        "errors": c_result.errors,
        "core_mode": c_sim.core_mode,
    }

    match = summary_py["winner_id"] == summary_c["winner_id"]
    if match:
        match = summary_py["steps_played"] == summary_c["steps_played"]
    if match:
        match = summary_py["total_ticks"] == summary_c["total_ticks"]

    first_diff: Optional[Dict[str, Any]] = None
    max_frames = min(len(py_result.frames), len(c_result.frames))

    def _normalized_players(players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in players:
            cp = dict(p)
            cp.pop("error", None)
            out.append(cp)
        return out

    for idx in range(max_frames):
        a = py_result.frames[idx]
        b = c_result.frames[idx]
        if a["gems"] != b["gems"] or _normalized_players(a["players"]) != _normalized_players(b["players"]):
            first_diff = {
                "frame_index": idx,
                "python_tick": a["tick"],
                "c_tick": b["tick"],
            }
            match = False
            break

    if match and len(py_result.frames) != len(c_result.frames):
        match = False
        first_diff = {
            "frame_count_python": len(py_result.frames),
            "frame_count_c": len(c_result.frames),
        }

    return {
        "match": match,
        "python": summary_py,
        "c": summary_c,
        "first_diff": first_diff,
    }
