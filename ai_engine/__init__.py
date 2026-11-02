"""BLOCK BLAST AI ULTIMATE — AI Decision Engine Package.

Provides high-performance heuristic board evaluation, baseline solvers (Random, Greedy),
multi-depth Beam Search solver, unified AIDecisionEngine API, and Strategy Arena.
"""

from ai_engine.baseline_solver import (
    BaseSolver,
    GreedySolver,
    RandomSolver,
)
from ai_engine.beam_search import (
    BeamSearchSolver,
    SearchNode,
)
from ai_engine.decision import (
    AIDecisionEngine,
    DecisionResult,
    DecisionStatus,
    PlannedMove,
)
from ai_engine.evaluator import (
    BoardEvaluator,
    BoardFeatures,
    EvaluationWeights,
)
from ai_engine.strategy_arena import (
    GameRecord,
    StrategyArena,
    StrategyStats,
)

__all__ = [
    # Evaluator
    "EvaluationWeights",
    "BoardFeatures",
    "BoardEvaluator",
    # Solvers
    "BaseSolver",
    "RandomSolver",
    "GreedySolver",
    "BeamSearchSolver",
    "SearchNode",
    # Decision API
    "DecisionStatus",
    "PlannedMove",
    "DecisionResult",
    "AIDecisionEngine",
    # Strategy Arena
    "GameRecord",
    "StrategyStats",
    "StrategyArena",
]
