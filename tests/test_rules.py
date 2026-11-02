"""Unit tests for Rule Configuration and Epistemic Status (game_engine/rules.py)."""

import unittest
from dataclasses import FrozenInstanceError

from game_engine.rules import (
    DEFAULT_RULES,
    GameRules,
    RuleStatus,
)


class TestGameRules(unittest.TestCase):
    """Test epistemic rule status taxonomy and rule customization."""

    def test_epistemic_rule_statuses(self):
        """CRITICAL: Confirmed, Provisional, and Unknown rules must be strictly segregated."""
        rules = DEFAULT_RULES

        # Confirmed physical rules
        self.assertEqual(rules.get_status("board_size"), RuleStatus.CONFIRMED)
        self.assertEqual(rules.get_status("allow_rotation"), RuleStatus.CONFIRMED)
        self.assertEqual(rules.get_status("simultaneous_line_clears"), RuleStatus.CONFIRMED)
        self.assertEqual(rules.get_status("tray_capacity"), RuleStatus.CONFIRMED)
        self.assertTrue(rules.is_confirmed("board_size"))
        self.assertTrue(rules.is_confirmed("allow_rotation"))

        # Provisional rules (unverified empirical hypotheses)
        self.assertEqual(rules.get_status("combo_window_moves"), RuleStatus.PROVISIONAL)
        self.assertEqual(rules.get_status("points_per_cell"), RuleStatus.PROVISIONAL)
        self.assertEqual(rules.get_status("points_per_line"), RuleStatus.PROVISIONAL)
        self.assertEqual(rules.get_status("combo_multiplier_base"), RuleStatus.PROVISIONAL)
        self.assertEqual(rules.get_status("all_clear_bonus"), RuleStatus.PROVISIONAL)
        self.assertFalse(rules.is_confirmed("combo_window_moves"))
        self.assertFalse(rules.is_confirmed("points_per_line"))

        # Unknown / server-side dynamic rules
        self.assertEqual(rules.get_status("deal_strategy"), RuleStatus.UNKNOWN)
        self.assertFalse(rules.is_confirmed("deal_strategy"))

        # Unregistered rule returns UNKNOWN
        self.assertEqual(rules.get_status("non_existent_rule"), RuleStatus.UNKNOWN)

    def test_rules_immutability(self):
        """GameRules must be frozen against direct mutation."""
        rules = DEFAULT_RULES
        with self.assertRaises(FrozenInstanceError):
            rules.combo_window_moves = 5  # type: ignore

    def test_with_updates(self):
        """Customizing rules via with_updates returns a new independent instance."""
        base_rules = DEFAULT_RULES
        custom_rules = base_rules.with_updates(
            combo_window_moves=0,  # Test strict instant-reset mode
            points_per_line=25,
            all_clear_bonus=500,
        )

        # Base rules unchanged
        self.assertEqual(base_rules.combo_window_moves, 3)
        self.assertEqual(base_rules.points_per_line, 10)
        self.assertEqual(base_rules.all_clear_bonus, 300)

        # Custom rules reflect updates
        self.assertEqual(custom_rules.combo_window_moves, 0)
        self.assertEqual(custom_rules.points_per_line, 25)
        self.assertEqual(custom_rules.all_clear_bonus, 500)

        # Invalid rule parameter raises AttributeError
        with self.assertRaises(AttributeError):
            base_rules.with_updates(invalid_param_xyz=123)

    def test_status_group_dictionaries(self):
        """Confirmed, provisional, and unknown dictionaries must partition rules accurately."""
        rules = DEFAULT_RULES
        confirmed = rules.get_confirmed_rules()
        provisional = rules.get_provisional_rules()
        unknown = rules.get_unknown_rules()

        self.assertIn("board_size", confirmed)
        self.assertIn("allow_rotation", confirmed)
        self.assertIn("combo_window_moves", provisional)
        self.assertIn("points_per_line", provisional)
        self.assertIn("deal_strategy", unknown)

        # Ensure no overlap
        self.assertTrue(set(confirmed.keys()).isdisjoint(set(provisional.keys())))
        self.assertTrue(set(confirmed.keys()).isdisjoint(set(unknown.keys())))
        self.assertTrue(set(provisional.keys()).isdisjoint(set(unknown.keys())))


if __name__ == "__main__":
    unittest.main()
