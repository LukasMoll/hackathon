from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rl.obs import ObsSpec, ObservationEncoder, iter_local_hex
from rl.policy import build_policy
from tronk_engine import BotRuntime, CallExpr, GameConfig, TronkSimulation
from tronk_ml import MLTronkEnv


ACTION_TO_TURN = [-1, 0, 1]
TURN_TO_ACTION = {-1: 0, 0: 1, 1: 2}


@dataclass(frozen=True)
class FeatureSchema:
    tile_coords: tuple[tuple[int, int], ...]
    include_turn: bool = True
    include_length: bool = True
    include_prev_action: bool = True
    include_player_info: bool = False
    length_cap: int = 20

    @property
    def names(self) -> List[str]:
        out: List[str] = []
        for q, r in self.tile_coords:
            out.append(f"tile_{q}_{r}")
        if self.include_turn:
            out.append("turn")
        if self.include_length:
            out.append("length")
        if self.include_prev_action:
            out.append("prev_action")
        if self.include_player_info:
            for pid in range(6):
                out.extend(
                    [
                        f"p{pid}_alive",
                        f"p{pid}_q",
                        f"p{pid}_r",
                        f"p{pid}_facing",
                        f"p{pid}_length",
                    ]
                )
        return out


def default_feature_schema(local_radius: int = 2) -> FeatureSchema:
    coords = [xy for xy in iter_local_hex(local_radius) if xy != (0, 0)]
    return FeatureSchema(tile_coords=tuple(coords))


def encode_tile_value(exists: int, is_empty: int, player_id: int, is_gem: int, self_id: int) -> int:
    # Discrete, integer-only feature values suitable for Tronkscript thresholds.
    # -2 out-of-board, -1 gem, 0 empty, 1 enemy occupied, 2 self occupied.
    if exists == 0:
        return -2
    if is_gem == 1:
        return -1
    if is_empty == 1:
        return 0
    if player_id == self_id:
        return 2
    return 1


def extract_features(env: MLTronkEnv, player_id: int, prev_action: int, schema: FeatureSchema) -> np.ndarray:
    features: List[int] = []
    for q, r in schema.tile_coords:
        exists, is_empty, occ_pid, is_gem = env.get_tile_rel(player_id, q, r)
        features.append(encode_tile_value(exists, is_empty, occ_pid, is_gem, self_id=player_id))

    if schema.include_turn:
        features.append(int(env.get_turn(player_id)))
    if schema.include_length:
        _, _, _, _, length = env.get_player_info(player_id)
        features.append(min(int(length), schema.length_cap))
    if schema.include_prev_action:
        features.append(int(prev_action))
    if schema.include_player_info:
        for pid in range(6):
            alive, head_q, head_r, head_facing, length = env.get_player_info(pid)
            features.append(int(alive))
            features.append(int(head_q))
            features.append(int(head_r))
            features.append(int(head_facing))
            features.append(min(int(length), schema.length_cap))
    return np.asarray(features, dtype=np.int16)


def infer_policy_config(policy_path: Path) -> tuple[str, int, int, int]:
    cfg_path = policy_path.parent / "train_config.json"
    arch = "planning"
    hidden_dim = 128
    mem_dim = 16
    radius = 5
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            arch = str(cfg.get("policy_arch", arch))
            hidden_dim = int(cfg.get("policy_hidden_dim", hidden_dim))
            mem_dim = int(cfg.get("policy_mem_dim", mem_dim))
            env_cfg = cfg.get("env", {})
            if isinstance(env_cfg, dict):
                radius = int(env_cfg.get("radius", radius))
        except Exception:
            pass
    return arch, hidden_dim, mem_dim, radius


def load_policy(policy_path: Path, device: torch.device) -> tuple[torch.nn.Module, ObservationEncoder]:
    arch, hidden_dim, mem_dim, radius = infer_policy_config(policy_path)
    encoder = ObservationEncoder(ObsSpec(radius=radius, include_threat_map=True))
    model = build_policy(arch, obs_dim=encoder.obs_dim, hidden_dim=hidden_dim, mem_dim=mem_dim).to(device)
    state = torch.load(policy_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, encoder


@dataclass
class DecisionTreeParams:
    max_depth: int = 8
    min_samples_split: int = 128
    min_samples_leaf: int = 64
    min_gain: float = 1e-4


@dataclass
class TreeNode:
    is_leaf: bool
    prediction: int
    counts: tuple[int, ...]
    feature_idx: int = -1
    threshold: int = 0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    depth: int = 0
    samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "is_leaf": bool(self.is_leaf),
            "prediction": int(self.prediction),
            "counts": [int(x) for x in self.counts],
            "depth": int(self.depth),
            "samples": int(self.samples),
        }
        if not self.is_leaf:
            out["feature_idx"] = int(self.feature_idx)
            out["threshold"] = int(self.threshold)
            out["left"] = self.left.to_dict() if self.left is not None else None
            out["right"] = self.right.to_dict() if self.right is not None else None
        return out

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TreeNode":
        node = TreeNode(
            is_leaf=bool(data["is_leaf"]),
            prediction=int(data["prediction"]),
            counts=tuple(int(x) for x in data["counts"]),
            feature_idx=int(data.get("feature_idx", -1)),
            threshold=int(data.get("threshold", 0)),
            depth=int(data.get("depth", 0)),
            samples=int(data.get("samples", 0)),
        )
        if not node.is_leaf:
            if data.get("left") is not None:
                node.left = TreeNode.from_dict(data["left"])
            if data.get("right") is not None:
                node.right = TreeNode.from_dict(data["right"])
        return node


class SimpleDecisionTreeClassifier:
    def __init__(self, params: Optional[DecisionTreeParams] = None):
        self.params = params or DecisionTreeParams()
        self.classes_: Optional[np.ndarray] = None
        self.root_: Optional[TreeNode] = None
        self.n_features_: int = 0

    @staticmethod
    def _gini(counts: np.ndarray) -> float:
        n = float(np.sum(counts))
        if n <= 0:
            return 0.0
        p = counts / n
        return float(1.0 - np.sum(p * p))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must be a 1D array matching x rows")
        if x.shape[0] == 0:
            raise ValueError("Cannot fit tree with zero samples")

        self.n_features_ = int(x.shape[1])
        self.classes_ = np.unique(y)
        class_to_idx = {int(c): i for i, c in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[int(v)] for v in y], dtype=np.int32)
        n_classes = len(self.classes_)

        def class_counts(indices: np.ndarray) -> np.ndarray:
            return np.bincount(y_idx[indices], minlength=n_classes).astype(np.int64)

        def majority_label(counts: np.ndarray) -> int:
            return int(self.classes_[int(np.argmax(counts))])

        def build(indices: np.ndarray, depth: int) -> TreeNode:
            counts = class_counts(indices)
            samples = int(indices.shape[0])
            pred = majority_label(counts)
            impurity = self._gini(counts)

            if depth >= self.params.max_depth:
                return TreeNode(True, pred, tuple(int(c) for c in counts), depth=depth, samples=samples)
            if samples < self.params.min_samples_split:
                return TreeNode(True, pred, tuple(int(c) for c in counts), depth=depth, samples=samples)
            if samples < (self.params.min_samples_leaf * 2):
                return TreeNode(True, pred, tuple(int(c) for c in counts), depth=depth, samples=samples)
            if impurity <= 1e-12:
                return TreeNode(True, pred, tuple(int(c) for c in counts), depth=depth, samples=samples)

            best_gain = -1.0
            best_feature = -1
            best_threshold = 0
            best_left: Optional[np.ndarray] = None
            best_right: Optional[np.ndarray] = None

            x_sub = x[indices]
            for feature_idx in range(self.n_features_):
                values = x_sub[:, feature_idx]
                uniq = np.unique(values)
                if uniq.shape[0] <= 1:
                    continue
                # Integer thresholds only: split on <= unique_value.
                for threshold in uniq[:-1]:
                    left_mask = values <= threshold
                    left_count = int(np.sum(left_mask))
                    right_count = samples - left_count
                    if left_count < self.params.min_samples_leaf or right_count < self.params.min_samples_leaf:
                        continue

                    left_indices = indices[left_mask]
                    right_indices = indices[~left_mask]
                    c_left = class_counts(left_indices)
                    c_right = class_counts(right_indices)

                    g_left = self._gini(c_left)
                    g_right = self._gini(c_right)
                    weighted = (left_count / samples) * g_left + (right_count / samples) * g_right
                    gain = impurity - weighted

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = int(threshold)
                        best_left = left_indices
                        best_right = right_indices

            if best_feature < 0 or best_left is None or best_right is None:
                return TreeNode(True, pred, tuple(int(c) for c in counts), depth=depth, samples=samples)
            if best_gain < self.params.min_gain:
                return TreeNode(True, pred, tuple(int(c) for c in counts), depth=depth, samples=samples)

            left_node = build(best_left, depth + 1)
            right_node = build(best_right, depth + 1)
            return TreeNode(
                is_leaf=False,
                prediction=pred,
                counts=tuple(int(c) for c in counts),
                feature_idx=best_feature,
                threshold=best_threshold,
                left=left_node,
                right=right_node,
                depth=depth,
                samples=samples,
            )

        root = build(np.arange(x.shape[0], dtype=np.int32), 0)
        self.root_ = root

    def _predict_one_node(self, row: np.ndarray, node: TreeNode) -> int:
        cur = node
        while not cur.is_leaf:
            if row[cur.feature_idx] <= cur.threshold:
                if cur.left is None:
                    break
                cur = cur.left
            else:
                if cur.right is None:
                    break
                cur = cur.right
        return int(cur.prediction)

    def predict_one(self, row: np.ndarray) -> int:
        if self.root_ is None:
            raise RuntimeError("Tree has not been fit")
        return self._predict_one_node(row, self.root_)

    def predict_with_scores_one(self, row: np.ndarray) -> tuple[int, Dict[int, int]]:
        pred = int(self.predict_one(row))
        scores = {-1: 0, 0: 0, 1: 0}
        if pred in scores:
            scores[pred] = 1
        else:
            pred = 0
            scores[0] = 1
        return pred, scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Tree has not been fit")
        out = np.zeros((x.shape[0],), dtype=np.int16)
        for i in range(x.shape[0]):
            out[i] = self._predict_one_node(x[i], self.root_)
        return out

    def to_dict(self) -> Dict[str, Any]:
        if self.root_ is None or self.classes_ is None:
            raise RuntimeError("Tree has not been fit")
        return {
            "classes": [int(v) for v in self.classes_],
            "n_features": int(self.n_features_),
            "params": {
                "max_depth": int(self.params.max_depth),
                "min_samples_split": int(self.params.min_samples_split),
                "min_samples_leaf": int(self.params.min_samples_leaf),
                "min_gain": float(self.params.min_gain),
            },
            "root": self.root_.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SimpleDecisionTreeClassifier":
        params_data = data.get("params", {})
        params = DecisionTreeParams(
            max_depth=int(params_data.get("max_depth", 8)),
            min_samples_split=int(params_data.get("min_samples_split", 128)),
            min_samples_leaf=int(params_data.get("min_samples_leaf", 64)),
            min_gain=float(params_data.get("min_gain", 1e-4)),
        )
        clf = SimpleDecisionTreeClassifier(params=params)
        clf.classes_ = np.asarray([int(v) for v in data["classes"]], dtype=np.int32)
        clf.n_features_ = int(data["n_features"])
        clf.root_ = TreeNode.from_dict(data["root"])
        return clf

    def max_depth(self) -> int:
        if self.root_ is None:
            return 0

        def walk(node: TreeNode) -> int:
            if node.is_leaf:
                return node.depth
            left_depth = walk(node.left) if node.left is not None else node.depth
            right_depth = walk(node.right) if node.right is not None else node.depth
            return max(left_depth, right_depth)

        return int(walk(self.root_))

    def leaf_count(self) -> int:
        if self.root_ is None:
            return 0

        def walk(node: TreeNode) -> int:
            if node.is_leaf:
                return 1
            return (walk(node.left) if node.left is not None else 0) + (
                walk(node.right) if node.right is not None else 0
            )

        return int(walk(self.root_))


class TreeEnsemble:
    def __init__(self, trees: Sequence[SimpleDecisionTreeClassifier]):
        self.trees: List[SimpleDecisionTreeClassifier] = list(trees)
        if not self.trees:
            raise ValueError("TreeEnsemble requires at least one tree")

    def predict_one(self, row: np.ndarray) -> int:
        pred, _ = self.predict_with_scores_one(row)
        return int(pred)

    def predict_with_scores_one(self, row: np.ndarray) -> tuple[int, Dict[int, int]]:
        votes = {-1: 0, 0: 0, 1: 0}
        for tree in self.trees:
            pred = int(tree.predict_one(row))
            if pred not in votes:
                pred = 0
            votes[pred] += 1
        # Deterministic tie-break: forward (0), then left (-1), then right (1).
        best = 0
        best_count = votes[0]
        if votes[-1] > best_count:
            best = -1
            best_count = votes[-1]
        if votes[1] > best_count:
            best = 1
        return int(best), votes

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((x.shape[0],), dtype=np.int16)
        for i in range(x.shape[0]):
            out[i] = self.predict_one(x[i])
        return out

    def max_depth(self) -> int:
        return max(int(tree.max_depth()) for tree in self.trees)

    def leaf_count(self) -> int:
        return int(sum(int(tree.leaf_count()) for tree in self.trees))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "ensemble",
            "n_estimators": int(len(self.trees)),
            "trees": [tree.to_dict() for tree in self.trees],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TreeEnsemble":
        trees = [SimpleDecisionTreeClassifier.from_dict(td) for td in data["trees"]]
        return TreeEnsemble(trees)


def train_bagged_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int,
    params: DecisionTreeParams,
    seed: int,
) -> TreeEnsemble:
    if n_estimators <= 1:
        tree = SimpleDecisionTreeClassifier(params=params)
        tree.fit(x, y)
        return TreeEnsemble([tree])

    rng = np.random.default_rng(seed)
    n = x.shape[0]
    trees: List[SimpleDecisionTreeClassifier] = []
    for _ in range(n_estimators):
        sample_idx = rng.integers(0, n, size=n, dtype=np.int32)
        xt = x[sample_idx]
        yt = y[sample_idx]
        tree = SimpleDecisionTreeClassifier(params=params)
        tree.fit(xt, yt)
        trees.append(tree)
    return TreeEnsemble(trees)


def _emit_tree_node(
    lines: List[str],
    node: TreeNode,
    feature_vars: Sequence[str],
    action_var: str,
    indent: str,
) -> None:
    if node.is_leaf:
        lines.append(f"{indent}{action_var} = {int(node.prediction)}")
        return
    feature = feature_vars[node.feature_idx]
    threshold = int(node.threshold)
    lines.append(f"{indent}if ({feature} <= {threshold}) then")
    if node.left is not None:
        _emit_tree_node(lines, node.left, feature_vars, action_var, indent + "    ")
    else:
        lines.append(f"{indent}    {action_var} = {int(node.prediction)}")
    lines.append(f"{indent}else")
    if node.right is not None:
        _emit_tree_node(lines, node.right, feature_vars, action_var, indent + "    ")
    else:
        lines.append(f"{indent}    {action_var} = {int(node.prediction)}")
    lines.append(f"{indent}end if")


def generate_tronkscript(
    model: Any,
    schema: FeatureSchema,
    *,
    include_main: bool,
    infer_function_name: str = "infer_action",
    wait_ticks: int = 1200,
    safety_fallback: bool = True,
) -> str:
    if isinstance(model, SimpleDecisionTreeClassifier):
        if model.root_ is None:
            raise RuntimeError("Tree has not been fit")
        trees: List[SimpleDecisionTreeClassifier] = [model]
    elif isinstance(model, TreeEnsemble):
        trees = list(model.trees)
    else:
        raise TypeError(f"Unsupported model type for Tronkscript generation: {type(model)}")

    lines: List[str] = []
    lines.append("-- Auto-generated policy tree from ML model surrogate.")
    lines.append("-- Feature encoding: tile_code = {-2 out, -1 gem, 0 empty, 1 enemy, 2 self}.")
    lines.append("my_id = getPlayerId()")
    lines.append("mem_prev_action = 0")
    lines.append("")
    lines.append("function tile_code(q, r) do")
    lines.append("    exists, isEmpty, playerId, isGem = getTileRel(q, r)")
    lines.append("    if (exists == 0) then")
    lines.append("        value = -2")
    lines.append("    else if (isGem == 1) then")
    lines.append("        value = -1")
    lines.append("    else if (isEmpty == 1) then")
    lines.append("        value = 0")
    lines.append("    else if (playerId == my_id) then")
    lines.append("        value = 2")
    lines.append("    else")
    lines.append("        value = 1")
    lines.append("    end if")
    lines.append("    return value")
    lines.append("end function")
    lines.append("")
    lines.append(f"function {infer_function_name}() do")

    feature_vars: List[str] = []
    idx = 0
    for q, r in schema.tile_coords:
        var = f"f{idx}"
        lines.append(f"    {var} = tile_code({q}, {r})")
        feature_vars.append(var)
        idx += 1

    if schema.include_turn:
        var = f"f{idx}"
        lines.append(f"    {var} = getTurn()")
        feature_vars.append(var)
        idx += 1

    if schema.include_length:
        var = f"f{idx}"
        lines.append("    alive, headQ, headR, headFacing, myLength = getPlayerInfo(my_id)")
        lines.append(f"    {var} = myLength")
        lines.append(f"    if ({var} > {schema.length_cap}) then")
        lines.append(f"        {var} = {schema.length_cap}")
        lines.append("    end if")
        feature_vars.append(var)
        idx += 1

    if schema.include_prev_action:
        var = f"f{idx}"
        lines.append(f"    {var} = mem_prev_action")
        feature_vars.append(var)
        idx += 1

    if schema.include_player_info:
        for pid in range(6):
            lines.append(f"    p{pid}_alive, p{pid}_q, p{pid}_r, p{pid}_facing, p{pid}_length = getPlayerInfo({pid})")
            var_alive = f"f{idx}"
            lines.append(f"    {var_alive} = p{pid}_alive")
            feature_vars.append(var_alive)
            idx += 1

            var_q = f"f{idx}"
            lines.append(f"    {var_q} = p{pid}_q")
            feature_vars.append(var_q)
            idx += 1

            var_r = f"f{idx}"
            lines.append(f"    {var_r} = p{pid}_r")
            feature_vars.append(var_r)
            idx += 1

            var_face = f"f{idx}"
            lines.append(f"    {var_face} = p{pid}_facing")
            feature_vars.append(var_face)
            idx += 1

            var_len = f"f{idx}"
            lines.append(f"    {var_len} = p{pid}_length")
            lines.append(f"    if ({var_len} > {schema.length_cap}) then")
            lines.append(f"        {var_len} = {schema.length_cap}")
            lines.append("    end if")
            feature_vars.append(var_len)
            idx += 1

    lines.append("    action = 0")
    if len(trees) == 1:
        _emit_tree_node(lines, trees[0].root_, feature_vars, "action", "    ")
        lines.append("    score_left = 0")
        lines.append("    score_fwd = 0")
        lines.append("    score_right = 0")
        lines.append("    if (action == -1) then")
        lines.append("        score_left = 1")
        lines.append("    else if (action == 0) then")
        lines.append("        score_fwd = 1")
        lines.append("    else")
        lines.append("        score_right = 1")
        lines.append("    end if")
    else:
        lines.append("    vote_left = 0")
        lines.append("    vote_fwd = 0")
        lines.append("    vote_right = 0")
        for i, tree in enumerate(trees):
            action_var = f"tree_action_{i}"
            lines.append(f"    {action_var} = 0")
            _emit_tree_node(lines, tree.root_, feature_vars, action_var, "    ")
            lines.append(f"    if ({action_var} == -1) then")
            lines.append("        vote_left += 1")
            lines.append(f"    else if ({action_var} == 0) then")
            lines.append("        vote_fwd += 1")
            lines.append("    else")
            lines.append("        vote_right += 1")
            lines.append("    end if")
        lines.append("    if (vote_left > vote_fwd) then")
        lines.append("        if (vote_left > vote_right) then")
        lines.append("            action = -1")
        lines.append("        else")
        lines.append("            action = 1")
        lines.append("        end if")
        lines.append("    else")
        lines.append("        if (vote_fwd >= vote_right) then")
        lines.append("            action = 0")
        lines.append("        else")
        lines.append("            action = 1")
        lines.append("        end if")
        lines.append("    end if")
        lines.append("    score_left = vote_left")
        lines.append("    score_fwd = vote_fwd")
        lines.append("    score_right = vote_right")

    if safety_fallback:
        lines.append("    safe_left = 0")
        lines.append("    safe_fwd = 0")
        lines.append("    safe_right = 0")
        lines.append("    eL, emL, pL, gL = getTileRel(-1, 0)")
        lines.append("    if (eL == 1) then")
        lines.append("        if (emL == 1) then")
        lines.append("            safe_left = 1")
        lines.append("        else if (gL == 1) then")
        lines.append("            safe_left = 1")
        lines.append("        end if")
        lines.append("    end if")
        lines.append("    eF, emF, pF, gF = getTileRel(0, -1)")
        lines.append("    if (eF == 1) then")
        lines.append("        if (emF == 1) then")
        lines.append("            safe_fwd = 1")
        lines.append("        else if (gF == 1) then")
        lines.append("            safe_fwd = 1")
        lines.append("        end if")
        lines.append("    end if")
        lines.append("    eR, emR, pR, gR = getTileRel(1, -1)")
        lines.append("    if (eR == 1) then")
        lines.append("        if (emR == 1) then")
        lines.append("            safe_right = 1")
        lines.append("        else if (gR == 1) then")
        lines.append("            safe_right = 1")
        lines.append("        end if")
        lines.append("    end if")
        lines.append("    chosen_safe = 0")
        lines.append("    if (action == -1) then")
        lines.append("        if (safe_left == 1) then")
        lines.append("            chosen_safe = 1")
        lines.append("        end if")
        lines.append("    else if (action == 0) then")
        lines.append("        if (safe_fwd == 1) then")
        lines.append("            chosen_safe = 1")
        lines.append("        end if")
        lines.append("    else if (action == 1) then")
        lines.append("        if (safe_right == 1) then")
        lines.append("            chosen_safe = 1")
        lines.append("        end if")
        lines.append("    end if")
        lines.append("    if (chosen_safe == 0) then")
        lines.append("        best_action = action")
        lines.append("        best_score = -999999")
        lines.append("        if (safe_fwd == 1) then")
        lines.append("            best_action = 0")
        lines.append("            best_score = score_fwd")
        lines.append("        end if")
        lines.append("        if (safe_left == 1) then")
        lines.append("            if (score_left > best_score) then")
        lines.append("                best_action = -1")
        lines.append("                best_score = score_left")
        lines.append("            end if")
        lines.append("        end if")
        lines.append("        if (safe_right == 1) then")
        lines.append("            if (score_right > best_score) then")
        lines.append("                best_action = 1")
        lines.append("                best_score = score_right")
        lines.append("            end if")
        lines.append("        end if")
        lines.append("        if (best_score > -999999) then")
        lines.append("            action = best_action")
        lines.append("        end if")
        lines.append("    end if")
    lines.append("    return action")
    lines.append("end function")

    if include_main:
        lines.append("")
        lines.append("function main() do")
        lines.append("    while (1 == 1) do")
        lines.append(f"        action = {infer_function_name}()")
        lines.append("        turn(action)")
        lines.append("        mem_prev_action = action")
        lines.append(f"        wait({max(1, int(wait_ticks))})")
        lines.append("    end while")
        lines.append("end function")
        lines.append("")
        lines.append("main()")

    return "\n".join(lines) + "\n"


class _MLRuntimeBridge:
    def __init__(self, env: MLTronkEnv, player_id: int):
        self.env = env
        self.player_id = player_id
        self.tick = 0
        self.random = random.Random(0)

    def bind(self, env: MLTronkEnv, player_id: int) -> None:
        self.env = env
        self.player_id = player_id
        self.tick = int(env.get_tick())

    def append_log(self, player_id: int, line: str) -> None:
        _ = (player_id, line)

    def kill_player(self, player_id: int, reason: str) -> None:
        _ = (player_id, reason)

    def set_player_turn(self, player_id: int, rel_turn: int) -> None:
        _ = (player_id, rel_turn)

    def set_player_turn_raw(self, player_id: int, value: int) -> None:
        _ = (player_id, value)

    def get_player_turn(self, player_id: int) -> int:
        return int(self.env.get_turn(player_id))

    def rel_to_abs(self, player_id: int, q: int, r: int) -> Tuple[int, int]:
        return self.env.rel_to_abs(player_id, q, r)

    def get_tile_abs(self, q: int, r: int) -> Tuple[int, int, int, int]:
        return self.env.get_tile_abs(q, r)

    def get_player_info(self, target_id: int) -> Tuple[int, int, int, int, int]:
        return self.env.get_player_info(target_id)


def _drain_generator(gen: Any) -> tuple[Any, int]:
    cost_total = 0
    while True:
        try:
            cost = next(gen)
            if int(cost) > 0:
                cost_total += int(cost)
        except StopIteration as exc:
            return exc.value, cost_total


class TronkscriptInferencePolicy:
    def __init__(self, source_no_main: str, infer_function_name: str = "infer_action"):
        self.source = source_no_main
        self.infer_function_name = infer_function_name
        self._bridges: List[_MLRuntimeBridge] = []
        self._runtimes: List[BotRuntime] = []
        self.reset(seed=0, max_steps=1, use_c_core=False)

    def reset(self, seed: int, max_steps: int, use_c_core: bool) -> None:
        dummy_env = MLTronkEnv(seed=seed, max_steps=max_steps, use_c_core=use_c_core)
        self._bridges = []
        self._runtimes = []
        for pid in range(6):
            bridge = _MLRuntimeBridge(dummy_env, pid)
            runtime = BotRuntime(self.source, pid, bridge)  # type: ignore[arg-type]
            # Execute top-level to initialize globals once.
            _, _ = _drain_generator(runtime.gen)
            runtime.globals["mem_prev_action"] = 0
            self._bridges.append(bridge)
            self._runtimes.append(runtime)

    def act(self, env: MLTronkEnv, player_id: int) -> tuple[int, int]:
        runtime = self._runtimes[player_id]
        bridge = self._bridges[player_id]
        bridge.bind(env, player_id)
        gen = runtime._call(CallExpr(self.infer_function_name, []), runtime.current_scope)
        result, tick_cost = _drain_generator(gen)
        if isinstance(result, tuple):
            action = int(result[0]) if result else 0
        else:
            action = int(result)
        if action not in (-1, 0, 1):
            action = 0
        runtime.globals["mem_prev_action"] = int(action)
        return int(action), int(tick_cost)


def collect_teacher_dataset(
    *,
    policy_path: Path,
    schema: FeatureSchema,
    target_samples: int,
    seed: int,
    max_steps: int,
    use_c_core: bool,
    require_c_core: bool,
    deterministic_teacher: bool = True,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if target_samples <= 0:
        raise ValueError("target_samples must be > 0")

    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoder = load_policy(policy_path, device=device)
    hidden_dim = int(getattr(model, "hidden_dim", 128))

    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    episodes = 0
    steps = 0

    while len(x_rows) < target_samples:
        env = MLTronkEnv(
            seed=int(rng.integers(0, 2_000_000_000)),
            max_steps=max_steps,
            randomize_starts=True,
            randomize_facings=True,
            use_c_core=use_c_core,
            require_c_core=require_c_core,
        )
        hidden = [np.zeros((hidden_dim,), dtype=np.float32) for _ in range(6)]
        prev_actions = [0] * 6

        while not env.done and len(x_rows) < target_samples:
            actions = [0] * 6
            for pid in range(6):
                if not env.players[pid].alive:
                    continue
                obs = encoder.encode(env, pid)
                action_idx, _, _, new_h = model.act(obs, hidden[pid], device=device, deterministic=deterministic_teacher)
                hidden[pid] = new_h
                turn = int(ACTION_TO_TURN[action_idx])

                feat = extract_features(env, pid, prev_actions[pid], schema)
                x_rows.append(feat)
                y_rows.append(turn)
                actions[pid] = turn
                prev_actions[pid] = turn
            env.step(actions)
            steps += 1
        episodes += 1

    x = np.asarray(x_rows[:target_samples], dtype=np.int16)
    y = np.asarray(y_rows[:target_samples], dtype=np.int8)
    meta = {
        "episodes": int(episodes),
        "env_steps": int(steps),
        "samples": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
    }
    return x, y, meta


def collect_dagger_dataset(
    *,
    policy_path: Path,
    schema: FeatureSchema,
    predictor: Any,
    target_samples: int,
    seed: int,
    max_steps: int,
    use_c_core: bool,
    require_c_core: bool,
    deterministic_teacher: bool = True,
    teacher_mix_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if target_samples <= 0:
        raise ValueError("target_samples must be > 0")

    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoder = load_policy(policy_path, device=device)
    hidden_dim = int(getattr(model, "hidden_dim", 128))
    mix_prob = float(max(0.0, min(1.0, teacher_mix_prob)))

    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    episodes = 0
    steps = 0
    teacher_action_count = 0
    student_action_count = 0

    while len(x_rows) < target_samples:
        env = MLTronkEnv(
            seed=int(rng.integers(0, 2_000_000_000)),
            max_steps=max_steps,
            randomize_starts=True,
            randomize_facings=True,
            use_c_core=use_c_core,
            require_c_core=require_c_core,
        )
        hidden = [np.zeros((hidden_dim,), dtype=np.float32) for _ in range(6)]
        prev_actions = [0] * 6

        while not env.done and len(x_rows) < target_samples:
            actions = [0] * 6
            for pid in range(6):
                if not env.players[pid].alive:
                    continue

                feat = extract_features(env, pid, prev_actions[pid], schema)
                teacher_obs = encoder.encode(env, pid)
                action_idx, _, _, new_h = model.act(
                    teacher_obs,
                    hidden[pid],
                    device=device,
                    deterministic=deterministic_teacher,
                )
                hidden[pid] = new_h
                teacher_turn = int(ACTION_TO_TURN[action_idx])

                x_rows.append(feat)
                y_rows.append(teacher_turn)

                student_turn_raw, student_scores = _predict_action_with_scores(predictor, feat)
                student_turn = _apply_safety_fallback(env, pid, student_turn_raw, student_scores)

                use_teacher = bool(rng.random() < mix_prob)
                selected_turn = int(teacher_turn if use_teacher else student_turn)
                actions[pid] = selected_turn
                prev_actions[pid] = selected_turn
                if use_teacher:
                    teacher_action_count += 1
                else:
                    student_action_count += 1
            env.step(actions)
            steps += 1
        episodes += 1

    x = np.asarray(x_rows[:target_samples], dtype=np.int16)
    y = np.asarray(y_rows[:target_samples], dtype=np.int8)
    meta = {
        "episodes": int(episodes),
        "env_steps": int(steps),
        "samples": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "teacher_action_count": int(teacher_action_count),
        "student_action_count": int(student_action_count),
        "teacher_mix_prob": float(mix_prob),
    }
    return x, y, meta


def _predict_action_with_scores(predictor: Any, row: np.ndarray) -> tuple[int, Dict[int, int]]:
    if hasattr(predictor, "predict_with_scores_one"):
        pred, scores = predictor.predict_with_scores_one(row)
        action = int(pred)
        cleaned = {-1: 0, 0: 0, 1: 0}
        for k in (-1, 0, 1):
            cleaned[k] = int(scores.get(k, 0))
        if action not in cleaned:
            action = 0
        return action, cleaned

    pred = int(predictor.predict_one(row))
    scores = {-1: 0, 0: 0, 1: 0}
    if pred in scores:
        scores[pred] = 1
    else:
        pred = 0
        scores[0] = 1
    return pred, scores


def _is_walkable_tile(tile: tuple[int, int, int, int]) -> bool:
    exists, is_empty, _occ_pid, is_gem = tile
    return int(exists) == 1 and (int(is_empty) == 1 or int(is_gem) == 1)


def _apply_safety_fallback(
    env: MLTronkEnv,
    player_id: int,
    action: int,
    scores: Dict[int, int],
) -> int:
    safe = {
        -1: _is_walkable_tile(env.get_tile_rel(player_id, -1, 0)),
        0: _is_walkable_tile(env.get_tile_rel(player_id, 0, -1)),
        1: _is_walkable_tile(env.get_tile_rel(player_id, 1, -1)),
    }
    if safe.get(action, False):
        return int(action)

    best_action: Optional[int] = None
    best_score = -10**9
    # Tie-break order: forward, left, right.
    for candidate in (0, -1, 1):
        if not safe[candidate]:
            continue
        candidate_score = int(scores.get(candidate, 0))
        if candidate_score > best_score:
            best_score = candidate_score
            best_action = candidate
    if best_action is None:
        return int(action)
    return int(best_action)


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.shape[0] == 0:
        return {"accuracy": 0.0}
    correct = int(np.sum(y_true == y_pred))
    acc = float(correct / y_true.shape[0])
    return {"accuracy": acc}


def benchmark_tronkscript_vs_model(
    *,
    policy_path: Path,
    tronkscript_source_no_main: str,
    schema: FeatureSchema,
    predictor: Any,
    matches: int,
    seed: int,
    max_steps: int,
    use_c_core: bool,
    require_c_core: bool,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoder = load_policy(policy_path, device=device)
    model_hidden_dim = int(getattr(model, "hidden_dim", 128))

    ts_policy = TronkscriptInferencePolicy(tronkscript_source_no_main, infer_function_name="infer_action")

    rank_sum = {"model": 0.0, "ts": 0.0}
    wins = {"model": 0, "ts": 0}
    top2 = {"model": 0, "ts": 0}
    surv_sum = {"model": 0.0, "ts": 0.0}
    counts = {"model": 0, "ts": 0}
    ts_tick_cost_total = 0
    ts_tick_calls = 0
    tree_parity_matches = 0
    tree_parity_total = 0

    for _ in range(matches):
        env = MLTronkEnv(
            seed=int(rng.integers(0, 2_000_000_000)),
            max_steps=max_steps,
            randomize_starts=True,
            randomize_facings=True,
            use_c_core=use_c_core,
            require_c_core=require_c_core,
        )
        ts_policy.reset(seed=int(rng.integers(0, 2_000_000_000)), max_steps=max_steps, use_c_core=use_c_core)

        seats = np.arange(6)
        rng.shuffle(seats)
        seat_group = [""] * 6
        for i, pid in enumerate(seats):
            seat_group[int(pid)] = "model" if i < 3 else "ts"

        model_hidden = [np.zeros((model_hidden_dim,), dtype=np.float32) for _ in range(6)]
        tree_prev = [0] * 6

        while not env.done:
            actions = [0] * 6
            for pid in range(6):
                if not env.players[pid].alive:
                    continue
                if seat_group[pid] == "model":
                    obs = encoder.encode(env, pid)
                    action_idx, _, _, new_h = model.act(obs, model_hidden[pid], device=device, deterministic=True)
                    model_hidden[pid] = new_h
                    actions[pid] = int(ACTION_TO_TURN[action_idx])
                else:
                    ts_action, ts_cost = ts_policy.act(env, pid)
                    actions[pid] = int(ts_action)
                    ts_tick_cost_total += int(ts_cost)
                    ts_tick_calls += 1

                    feat = extract_features(env, pid, tree_prev[pid], schema)
                    tree_action_raw, tree_scores = _predict_action_with_scores(predictor, feat)
                    tree_action = _apply_safety_fallback(env, pid, tree_action_raw, tree_scores)
                    tree_parity_total += 1
                    if tree_action == ts_action:
                        tree_parity_matches += 1

                    tree_prev[pid] = int(ts_action)
            env.step(actions)

        ranks = env.compute_ranks()
        for pid, rank in enumerate(ranks):
            grp = seat_group[pid]
            counts[grp] += 1
            rank_sum[grp] += float(rank)
            if rank <= 1.0:
                wins[grp] += 1
            if rank <= 2.0:
                top2[grp] += 1
            death = env.death_step[pid]
            surv = float(death if death is not None else env.step_count)
            surv_sum[grp] += surv

    def summarize(group: str) -> Dict[str, float]:
        n = max(1, counts[group])
        return {
            "agents": float(counts[group]),
            "avg_rank": float(rank_sum[group] / n),
            "win_rate_per_agent": float(wins[group] / n),
            "top2_rate_per_agent": float(top2[group] / n),
            "avg_survival_steps": float(surv_sum[group] / n),
        }

    model_stats = summarize("model")
    ts_stats = summarize("ts")
    out = {
        "matches": int(matches),
        "model": model_stats,
        "tronkscript": ts_stats,
        "delta_tronkscript_minus_model": {
            "avg_rank": float(ts_stats["avg_rank"] - model_stats["avg_rank"]),
            "win_rate_per_agent": float(ts_stats["win_rate_per_agent"] - model_stats["win_rate_per_agent"]),
            "top2_rate_per_agent": float(ts_stats["top2_rate_per_agent"] - model_stats["top2_rate_per_agent"]),
            "avg_survival_steps": float(ts_stats["avg_survival_steps"] - model_stats["avg_survival_steps"]),
        },
        "tronkscript_runtime": {
            "avg_tick_cost_per_infer_call": float(ts_tick_cost_total / max(1, ts_tick_calls)),
            "infer_calls": int(ts_tick_calls),
            "python_tree_parity_rate": float(tree_parity_matches / max(1, tree_parity_total)),
            "python_tree_parity_samples": int(tree_parity_total),
        },
    }
    return out


def validate_tronkscript_bot(source_with_main: str, *, steps: int, seeds: Iterable[int]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    fatal_total = 0
    fatal_prefixes = ("Parse error", "Runtime error", "Program exited", "No runtime")
    for seed in seeds:
        sim = TronkSimulation(
            [source_with_main] * 6,
            config=GameConfig(max_steps=int(steps), seed=int(seed), use_c_core=True),
        )
        result = sim.run()
        fatal_errors = {pid: msg for pid, msg in result.errors.items() if msg.startswith(fatal_prefixes)}
        run = {
            "seed": int(seed),
            "winner_id": result.winner_id,
            "steps_played": result.steps_played,
            "total_ticks": result.total_ticks,
            "errors": dict(result.errors),
            "fatal_errors": fatal_errors,
            "engine_used": sim.core_mode,
        }
        fatal_total += len(fatal_errors)
        runs.append(run)
    return {"runs": runs, "fatal_error_count": int(fatal_total)}


def write_tree_artifacts(
    *,
    output_dir: Path,
    model: Any,
    schema: FeatureSchema,
    wait_ticks: int,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tree_json = output_dir / "tree_model.json"
    lib_ts = output_dir / "tree_policy_lib.tronkscript"
    bot_ts = output_dir / "tree_policy_bot.tronkscript"

    if isinstance(model, SimpleDecisionTreeClassifier):
        model_data = {"model_type": "tree", "tree": model.to_dict()}
    elif isinstance(model, TreeEnsemble):
        model_data = {"model_type": "ensemble", "ensemble": model.to_dict()}
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    model_payload = {
        "model": model_data,
        "schema": {
            "tile_coords": [[int(q), int(r)] for q, r in schema.tile_coords],
            "include_turn": bool(schema.include_turn),
            "include_length": bool(schema.include_length),
            "include_prev_action": bool(schema.include_prev_action),
            "include_player_info": bool(schema.include_player_info),
            "length_cap": int(schema.length_cap),
            "names": schema.names,
        },
    }
    tree_json.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    lib_code = generate_tronkscript(
        model,
        schema,
        include_main=False,
        infer_function_name="infer_action",
        wait_ticks=wait_ticks,
    )
    bot_code = generate_tronkscript(
        model,
        schema,
        include_main=True,
        infer_function_name="infer_action",
        wait_ticks=wait_ticks,
    )
    lib_ts.write_text(lib_code, encoding="utf-8")
    bot_ts.write_text(bot_code, encoding="utf-8")
    return {"tree_json": tree_json, "lib_tronkscript": lib_ts, "bot_tronkscript": bot_ts}
