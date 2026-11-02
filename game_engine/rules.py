"""Rule Specification and Configuration Taxonomy for BLOCK BLAST AI ULTIMATE.

CRITICAL REQUIREMENT:
- Rules are strictly decoupled from simulation and engine logic.
- Every rule carries an explicit epistemic status:
  - CONFIRMED: Proven fact from disassembly, game mechanics, or official store rules.
  - PROVISIONAL: Empirical hypothesis from audited clones/repos; NOT independently verified.
  - UNKNOWN: Speculative or dynamic game server behaviors (e.g. piece dealing DDA).
- Never label an unverified rule as confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Epistemic Rule Status
# ---------------------------------------------------------------------------

class RuleStatus(Enum):
    """Epistemic confidence status for game rules and mechanics."""

    CONFIRMED = auto()    # Verified fact from direct gameplay observation / mechanics.
    PROVISIONAL = auto()  # Derived from cloned repositories / empirical tests; unproven on official Play Store build.
    UNKNOWN = auto()      # Unknown or closed-source dynamic game server logic (e.g. anti-loss DDA algorithms).


# ---------------------------------------------------------------------------
# Configurable Game Rules Object
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GameRules:
    """Configurable rule set governing Block Blast simulation, scoring, and dealing.

    All rules can be modified or customized dynamically without altering
    the underlying engine or bitboard routines.
    """

    # Grid & Physical Mechanics
    board_size: int = 8
    allow_rotation: bool = False
    simultaneous_line_clears: bool = True
    tray_capacity: int = 3

    # Combo Mechanics
    # In official Android Block Blast, empirical tests indicate players receive
    # a grace window of up to 3 non-clearing moves (combo breaks on 4th consecutive non-clearing move).
    combo_window_moves: int = 3

    # Scoring Constants (Provisional Hypothesis)
    # Δscore = cells_placed * points_per_cell + lines_cleared * points_per_line + combo_bonus + all_clear_bonus
    points_per_cell: int = 1
    points_per_line: int = 10
    combo_multiplier_base: int = 10
    all_clear_bonus: int = 300

    # Dealing Strategy: "uniform", "weighted", or "provisional_dda"
    deal_strategy: str = "uniform"

    # Rule Metadata Mapping: tracks status of every individual rule
    _statuses: dict[str, RuleStatus] = field(
        default_factory=lambda: {
            "board_size": RuleStatus.CONFIRMED,
            "allow_rotation": RuleStatus.CONFIRMED,
            "simultaneous_line_clears": RuleStatus.CONFIRMED,
            "tray_capacity": RuleStatus.CONFIRMED,
            "combo_window_moves": RuleStatus.PROVISIONAL,
            "points_per_cell": RuleStatus.PROVISIONAL,
            "points_per_line": RuleStatus.PROVISIONAL,
            "combo_multiplier_base": RuleStatus.PROVISIONAL,
            "all_clear_bonus": RuleStatus.PROVISIONAL,
            "deal_strategy": RuleStatus.UNKNOWN,
        }
    )

    def get_status(self, rule_name: str) -> RuleStatus:
        """Return the epistemic status of a specific rule."""
        if rule_name not in self._statuses:
            return RuleStatus.UNKNOWN
        return self._statuses[rule_name]

    def is_confirmed(self, rule_name: str) -> bool:
        """Return True only if the rule is verified as CONFIRMED."""
        return self.get_status(rule_name) == RuleStatus.CONFIRMED

    def get_confirmed_rules(self) -> dict[str, Any]:
        """Return dictionary of all confirmed rules and their values."""
        return {
            k: getattr(self, k)
            for k, status in self._statuses.items()
            if status == RuleStatus.CONFIRMED and hasattr(self, k)
        }

    def get_provisional_rules(self) -> dict[str, Any]:
        """Return dictionary of all provisional rules and their values."""
        return {
            k: getattr(self, k)
            for k, status in self._statuses.items()
            if status == RuleStatus.PROVISIONAL and hasattr(self, k)
        }

    def get_unknown_rules(self) -> dict[str, Any]:
        """Return dictionary of all unknown rules and their values."""
        return {
            k: getattr(self, k)
            for k, status in self._statuses.items()
            if status == RuleStatus.UNKNOWN and hasattr(self, k)
        }

    def with_updates(
        self,
        new_statuses: Mapping[str, RuleStatus] | None = None,
        **updates: Any,
    ) -> GameRules:
        """Return a new GameRules instance with specified updates applied.

        Maintains immutability while allowing arbitrary rule customization.
        """
        current_data = {
            "board_size": self.board_size,
            "allow_rotation": self.allow_rotation,
            "simultaneous_line_clears": self.simultaneous_line_clears,
            "tray_capacity": self.tray_capacity,
            "combo_window_moves": self.combo_window_moves,
            "points_per_cell": self.points_per_cell,
            "points_per_line": self.points_per_line,
            "combo_multiplier_base": self.combo_multiplier_base,
            "all_clear_bonus": self.all_clear_bonus,
            "deal_strategy": self.deal_strategy,
        }
        current_statuses = dict(self._statuses)

        for k, v in updates.items():
            if k in current_data:
                current_data[k] = v
            else:
                raise AttributeError(f"Unknown game rule parameter: '{k}'")

        if new_statuses:
            current_statuses.update(new_statuses)

        return GameRules(
            **current_data,
            _statuses=current_statuses,
        )


# Default game rules instance
DEFAULT_RULES: GameRules = GameRules()
