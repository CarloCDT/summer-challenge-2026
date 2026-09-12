"""AlphaZero-style MCTS: PUCT selection, network-guided expansion (no rollouts), backprop.

Single-agent framing (see training plan): our agent only ever controls player 0, the opponent's
move is baked into GameSimulator's turn resolution, so there is no sign-flipping in backprop -
the value head predicts "how good is this state for player 0" directly, and every visited node
on a simulation's path gets the exact same value added to its total.
"""
import math
from typing import Dict, List, Tuple

import numpy as np
import torch

from .simulator import Action, GameSimulator

# Converts a final score margin (player 0 - player 1) into the [-1, 1] range the value head
# is trained against. This is a tunable hyperparameter, not a derived constant - based on the
# ~300-700 point score ranges observed in early test games, chosen so that a decisive win still
# saturates close to +-1 without every ordinary game washing out near 0.
SCORE_NORM = 200.0

ACTION_KIND_CHANNEL = {"PLACE": 0, "DISRUPT": 1}

# Temporarily applied to a node the instant it's selected during batch collection, so other
# simulations *in the same batch* don't all pile onto the identical highest-PUCT leaf before its
# real value is known (nothing has been backpropagated for it yet). Undone once the batch's
# evaluation completes and the real value is backpropagated instead.
VIRTUAL_LOSS = 1.0


def score_margin_to_value(score_diff: float) -> float:
    return max(-1.0, min(1.0, score_diff / SCORE_NORM))


class Node:
    __slots__ = ("parent", "prior", "visit_count", "total_value", "children", "expanded")

    def __init__(self, parent: "Node", prior: float):
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[Action, "Node"] = {}
        self.expanded = False

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0


class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        c_puct: float = 1.5,
        num_simulations: int = 200,
        eval_batch_size: int = 32,
    ):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.eval_batch_size = eval_batch_size

    def run(self, root_simulator: GameSimulator) -> Dict[Action, int]:
        """Runs `num_simulations` simulations from `root_simulator` (never mutated - every
        simulation clones it) and returns the root's {action: visit_count}, the "improved
        policy" distribution AlphaZero trains the network's policy head toward.

        Simulations are collected in batches of `eval_batch_size`: each batch's leaves are
        evaluated with ONE network forward pass instead of one-per-simulation - the single
        biggest MCTS speedup available without changing the algorithm, since a naive
        one-at-a-time loop makes hundreds of tiny GPU calls per real turn decision. Virtual loss
        keeps simulations within a batch from all selecting the same still-unresolved leaf."""
        root = Node(parent=None, prior=1.0)
        self._expand_or_terminal(root, root_simulator)

        remaining = self.num_simulations
        while remaining > 0:
            batch_n = min(self.eval_batch_size, remaining)
            remaining -= batch_n

            pending_paths: List[List[Node]] = []
            pending_sims: List[GameSimulator] = []
            resolved: List[Tuple[List[Node], float]] = []

            for _ in range(batch_n):
                sim = root_simulator.clone()
                node = root
                path: List[Node] = [node]

                while node.expanded and not sim.is_game_over():
                    action, node = self._select_child(node)
                    sim.apply(action)
                    path.append(node)
                    node.visit_count += 1
                    node.total_value -= VIRTUAL_LOSS

                if sim.is_game_over():
                    score_diff = sim.game_state.scores[0] - sim.game_state.scores[1]
                    resolved.append((path, score_margin_to_value(score_diff)))
                elif not sim.get_legal_actions():
                    resolved.append((path, 0.0))  # unreachable in practice, guarded defensively
                else:
                    pending_paths.append(path)
                    pending_sims.append(sim)

            for path, value in resolved:
                self._undo_virtual_loss(path)
                self._backpropagate(path, value)

            if pending_sims:
                values = self._batched_expand(pending_paths, pending_sims)
                for path, value in zip(pending_paths, values):
                    self._undo_virtual_loss(path)
                    self._backpropagate(path, value)

        return {action: child.visit_count for action, child in root.children.items()}

    @staticmethod
    def _undo_virtual_loss(path: List[Node]):
        for node in path[1:]:  # path[0] is root, which selection never applies virtual loss to
            node.visit_count -= 1
            node.total_value += VIRTUAL_LOSS

    def _batched_expand(self, paths: List[List[Node]], sims: List[GameSimulator]) -> List[float]:
        """One network forward pass for a whole batch of leaves at once. Populates each leaf
        node's children (priors masked to that leaf's own legal actions) and returns each
        leaf's value-head estimate, in the same order as `paths`/`sims`."""
        encoded_batch = np.stack([sim.get_encoded_state() for sim in sims])
        mask_batch = np.stack([sim.get_action_mask() for sim in sims])

        x = torch.from_numpy(encoded_batch).to(self.device)
        m = torch.from_numpy(mask_batch).to(self.device)

        self.model.eval()
        with torch.no_grad():
            policy_logits, values = self.model(x, m)  # (B, 2, H, W), (B, 1)

            # Illegal cells are already -1e9 (masked by the model), so softmax over the WHOLE
            # (2, H, W) tensor per batch item gives exactly the same normalized distribution
            # over legal cells as computing softmax restricted to just the legal subset -
            # e^-1e9 underflows to exact 0.0 in float32, contributing nothing to the
            # normalizer. This lets priors be read off with plain indexing below instead of
            # gathering an explicit "legal logits" tensor via a per-action Python loop first.
            flat_probs = torch.softmax(policy_logits.flatten(1), dim=1).view_as(policy_logits)

        probs_np = flat_probs.cpu().numpy()
        values_np = values.cpu().numpy()

        results = []
        for i, (path, sim) in enumerate(zip(paths, sims)):
            node = path[-1]
            legal_actions = sim.get_legal_actions()
            pad_top, pad_left = sim.pad_offsets
            item_probs = probs_np[i]

            for kind, x, y in legal_actions:
                prior = float(item_probs[ACTION_KIND_CHANNEL[kind], pad_top + y, pad_left + x])
                node.children[(kind, x, y)] = Node(parent=node, prior=prior)
            node.expanded = True

            results.append(float(values_np[i, 0]))

        return results

    def _select_child(self, node: Node) -> Tuple[Action, Node]:
        """Standard PUCT: argmax_a [ Q(a) + c_puct * P(a) * sqrt(sum_siblings_N) / (1 + N(a)) ]."""
        sqrt_total = math.sqrt(sum(child.visit_count for child in node.children.values()))

        best_score, best_action, best_child = -float("inf"), None, None
        for action, child in node.children.items():
            score = child.q_value + self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            if score > best_score:
                best_score, best_action, best_child = score, action, child

        return best_action, best_child

    def _expand_or_terminal(self, node: Node, sim: GameSimulator) -> float:
        """Evaluates a freshly-reached node with a single-item batch. Only used for the root's
        initial expansion before the batched loop in run() takes over; if the real game has
        already ended here, use the actual outcome instead of calling the network."""
        if sim.is_game_over():
            node.expanded = True  # no children to expand - nothing further to search here
            score_diff = sim.game_state.scores[0] - sim.game_state.scores[1]
            return score_margin_to_value(score_diff)

        if not sim.get_legal_actions():
            # Unreachable in practice - GameSimulator's phase cascading guarantees either a
            # fresh legal action exists or the game is over - but guarded defensively.
            node.expanded = True
            return 0.0

        return self._batched_expand([[node]], [sim])[0]

    @staticmethod
    def _backpropagate(path: List[Node], value: float):
        for node in path:
            node.visit_count += 1
            node.total_value += value


def visit_count_policy(visit_counts: Dict[Action, int], temperature: float = 1.0) -> Dict[Action, float]:
    """Converts root visit counts into a probability distribution. temperature=1.0 is
    proportional to visit count (used during self-play for exploration); temperature -> 0
    approaches greedy argmax (used for evaluation/deployment).

    Computed in log-space (`count ** (1/temperature)` overflows for small temperature and
    large counts, e.g. 20 ** 1000) rather than direct exponentiation - mathematically
    equivalent after normalization, just numerically stable."""
    if not visit_counts:
        return {}

    if temperature < 1e-3:
        best_action = max(visit_counts, key=visit_counts.get)
        return {action: (1.0 if action == best_action else 0.0) for action in visit_counts}

    actions = list(visit_counts.keys())
    counts = np.array([visit_counts[a] for a in actions], dtype=np.float64)
    log_scaled = np.log(counts + 1e-10) / temperature
    log_scaled -= log_scaled.max()
    scaled = np.exp(log_scaled)
    probs = scaled / scaled.sum()
    return dict(zip(actions, probs.tolist()))
