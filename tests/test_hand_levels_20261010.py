"""Hand-level cache regressions; use fake sessions and installed static assets."""
import json
import re
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from strategy import Card, Joker, RunState, HAND_LEVELS, PLANET_HAND, hand_name, score_hand
from hand_levels import fresh_levels, after_discard, after_play, sync_cache, needs_observation, secret_candidates
from consumables import Consumable, consumed_engine
from controller import collect_levels, collect_secret_levels, collect_position
from perception import read_hand_preview_level
from recovery import begin_action, finish_action, state_from_dict
from tests.test_runtime_recovery import FakeSession, level_words

ROOT = Path(__file__).resolve().parents[1]


class HandLevelTests(unittest.TestCase):
    def test_all_twelve_increments_match_installed_game(self):
        with ZipFile(ROOT / 'Balatro/Balatro.exe') as archive:
            source = archive.read('game.lua').decode()
        found = {}
        for line in source.splitlines():
            if 'l_chips =' not in line or 's_mult =' not in line:
                continue
            name = re.search(r'\["([^"]+)"\]', line)
            if name and hand_name(name[1]) in HAND_LEVELS:
                values = {k: int(v) for k, v in re.findall(r'(s_chips|s_mult|l_chips|l_mult)\s*=\s*(\d+)', line)}
                found[hand_name(name[1])] = tuple(values[k] for k in ('s_chips', 's_mult', 'l_chips', 'l_mult'))
        self.assertEqual(found, HAND_LEVELS)

    def test_fresh_run_resets_every_category_without_sharing_dictionary(self):
        first = fresh_levels(RunState(levels={'pair': 12}, uncertain_levels=('pair',), hand_history_dirty=True))
        second = fresh_levels(first)
        self.assertEqual(first.levels, dict.fromkeys(HAND_LEVELS, 1))
        self.assertFalse(needs_observation(first))
        first.levels['pair'] = 5
        self.assertEqual(second.level('pair'), 1)

    def test_normal_recollection_ignores_old_forced_refresh_without_opening_panel(self):
        state = fresh_levels(RunState(round_resources_observed=True))
        found = [(Card(i+2, 'S'), 1., (600+i*100, 725)) for i in range(8)]
        with tempfile.TemporaryDirectory() as d:
            session = FakeSession(d)
            session.policy_state = state
            with patch('controller.observed_state', return_value=(state, 8)), \
                    patch('controller.collect_jokers', return_value=state), \
                    patch('controller.collect_cards', return_value=found), \
                    patch('controller.collect_levels', side_effect=AssertionError('Unnecessary level panel')):
                result, cards = collect_position(session, state, refresh_levels=True)
            self.assertEqual(result.levels, state.levels)
            self.assertEqual(cards, found)
            self.assertEqual(session.taps, [])

    def test_invisible_secret_hand_level_is_read_from_preview_and_selection_restored(self):
        state = replace(fresh_levels(RunState()), uncertain_levels=('flush_five',))
        found = [(Card(7, 'S'), 1., (600+i*100, 725)) for i in range(5)]
        raised = [(c, confidence, (x, y-50)) for c, confidence, (x, y) in found]
        with tempfile.TemporaryDirectory() as d:
            session = FakeSession(d)
            with patch('controller.select_verified', return_value=session.frame) as select, \
                    patch('controller.read_hand', side_effect=[raised, found]), \
                    patch('perception.read_hand_preview_level', return_value=('flush_five', 4)):
                result, restored = collect_secret_levels(session, state, found)
            self.assertEqual(result.level('flush_five'), 4)
            self.assertEqual(result.uncertain_levels, ())
            self.assertEqual(select.call_args_list[1].args[2], set())
            self.assertEqual(restored, found)

    def test_each_planet_and_black_hole_update_without_invalidation(self):
        state = fresh_levels(RunState())
        for planet, category in PLANET_HAND.items():
            after = consumed_engine(state, Consumable(planet, 'Planet'))
            self.assertEqual(after.level(category), 2)
            self.assertEqual(sum(v == 2 for v in after.levels.values()), 1)
            self.assertFalse(needs_observation(after))
        after = consumed_engine(state, Consumable('Black Hole', 'Spectral'))
        self.assertEqual(after.levels, dict.fromkeys(HAND_LEVELS, 2))
        self.assertFalse(needs_observation(after))

    def test_burnt_first_discard_copies_and_debuff(self):
        state = fresh_levels(RunState(jokers=(Joker('Blueprint'), Joker('Burnt Joker')),
                                     round_discards_used=0))
        cards = [Card(7, 'S'), Card(7, 'H')]
        after = after_discard(state, replace(state, round_discards_used=1), cards)
        self.assertEqual(after.level('pair'), 3)
        self.assertFalse(needs_observation(after))
        second = after_discard(after, replace(after, round_discards_used=2), cards)
        self.assertEqual(second.level('pair'), 3)
        debuffed = replace(state, jokers=(Joker('Burnt Joker', debuffed=True),))
        self.assertEqual(after_discard(debuffed, debuffed, cards).level('pair'), 1)

    def test_arm_permanent_reduction_floor_and_disabled_boss(self):
        state = fresh_levels(RunState(boss='The Arm'))
        state = replace(state, levels={**state.levels, 'pair': 4})
        after = after_play(state, state, 'pair')
        self.assertEqual(after.level('pair'), 3)
        self.assertFalse(needs_observation(after))
        normal = replace(after, boss='')
        self.assertEqual(score_hand([Card(2, 'S'), Card(2, 'H')], normal).score, (40+4)*4)
        low = after_play(fresh_levels(RunState(boss='The Arm')), fresh_levels(RunState()), 'pair')
        self.assertEqual(low.level('pair'), 1)
        for disabled in (replace(state, boss_disabled=True), replace(state, jokers=(Joker('Chicot'),))):
            self.assertEqual(after_play(disabled, disabled, 'pair').level('pair'), 4)

    def test_space_never_accumulates_expected_trigger(self):
        state = fresh_levels(RunState(jokers=(Joker('Space Joker'),)))
        after = after_play(state, state, 'straight')
        self.assertEqual(after.level('straight'), 1)
        self.assertEqual(after.uncertain_levels, ('straight',))
        self.assertTrue(needs_observation(after))

    def test_arm_reduces_before_space_and_cannot_cancel_upgrade_at_level_one(self):
        cards = [Card(2, 'S'), Card(2, 'H')]
        state = fresh_levels(RunState(boss='The Arm', jokers=(Joker('Space Joker'),)))
        # No trigger: (10+4)*2=28; trigger: (25+4)*3=87.
        self.assertAlmostEqual(score_hand(cards, state).score, .75*28 + .25*87)
        state = replace(state, levels={**state.levels, 'pair': 3})
        self.assertAlmostEqual(score_hand(cards, state).score, .75*87 + .25*176)

    def test_boss_rejected_hand_does_not_trigger_space(self):
        state = fresh_levels(RunState(boss='The Mouth', boss_target_hand='straight',
                                     jokers=(Joker('Space Joker'),)))
        after = after_play(state, state, 'pair', [Card(2, 'S'), Card(2, 'H')])
        self.assertFalse(needs_observation(after))

    def test_probability_row_does_not_require_unaffected_rows(self):
        state = fresh_levels(RunState())
        state = replace(state, levels={**state.levels, 'pair': 6}, uncertain_levels=('straight',))
        words = [('Back', .99, (1032, 1027)), ('Lvl.3', .99, (641, 614)),
                 ('Straight', .99, (915, 614)), ('2', .99, (1440, 614))]
        with tempfile.TemporaryDirectory() as d:
            session = FakeSession(d)
            sync_cache(session, state)
            with patch('controller.read_text', return_value=words), patch('controller.hand_counter', return_value=(8, 8)):
                result = collect_levels(session, state)
            self.assertEqual(result.level('straight'), 3)
            self.assertEqual(result.level('pair'), 6)
            self.assertFalse(needs_observation(result))
            self.assertEqual(len(session.taps), 2)

    def test_uncertain_row_cannot_be_filled_from_old_cache(self):
        state = replace(fresh_levels(RunState()), uncertain_levels=('straight',))
        with tempfile.TemporaryDirectory() as d:
            session = FakeSession(d)
            sync_cache(session, state)
            session.levels_dirty = False  # Even a legacy flag cannot authorize this.
            with patch('controller.read_text', return_value=level_words((4,))), \
                    patch('controller.hand_counter', return_value=(8, 8)), patch('controller.time.sleep'):
                with self.assertRaisesRegex(RuntimeError, 'straight'):
                    collect_levels(session, state)

    def test_hidden_burnt_keeps_invisible_secret_levels_uncertain(self):
        state = fresh_levels(RunState(jokers=(Joker('Burnt Joker'),), round_discards_used=0))
        after = after_discard(state, state, [Card(7, 'S', face_down=True)])
        with tempfile.TemporaryDirectory() as d:
            session = FakeSession(d)
            with patch('controller.read_text', return_value=level_words()), patch('controller.hand_counter', return_value=(8, 8)):
                result = collect_levels(session, after)
        self.assertEqual(set(result.uncertain_levels), {'five_kind', 'flush_house', 'flush_five'})
        self.assertFalse(needs_observation(result))
        self.assertEqual(secret_candidates(result, [Card(7, 'S'), Card(8, 'H')]), {})
        self.assertIn('flush_five', secret_candidates(result, [Card(7, 'S')]*5))

    def test_hidden_play_marks_history_for_actual_readback(self):
        after = after_play(fresh_levels(RunState()), fresh_levels(RunState()), 'unknown')
        self.assertTrue(after.hand_history_dirty)
        self.assertTrue(needs_observation(after))
        state = fresh_levels(RunState(boss='The Mouth', boss_target_hand='straight',
                                     jokers=(Joker('Space Joker'),)))
        # Guessed hidden identities must not rule out a real legal hand.
        after = after_play(state, state, 'unknown', [Card(2, 'S', face_down=True)])
        self.assertEqual(set(after.uncertain_levels), set(HAND_LEVELS))

    def test_confirmed_planet_journal_updates_once_and_not_before_receipt(self):
        state = fresh_levels(RunState(consumables_held=1))
        after = consumed_engine(state, Consumable('Mercury', 'Planet'))
        with tempfile.TemporaryDirectory() as d:
            session = FakeSession(d)
            session.policy_state = state
            begin_action(session, 'consume', {'consumable_count': 1}, after_state=after,
                         name='Mercury', consumable_kind='planet', levels_changed=True)
            self.assertEqual(session.policy_state.level('pair'), 1)
            finish_action(session, True)
            finish_action(session, True)
            self.assertEqual(session.policy_state.level('pair'), 2)
            self.assertFalse(session.refresh_levels)
            saved = state_from_dict(json.loads((Path(d)/'policy-state.json').read_text(encoding='utf-8')))
            self.assertFalse(needs_observation(saved))
            self.assertEqual(saved.level('pair'), 2)

    def test_unapplied_action_leaves_levels_unchanged(self):
        state = fresh_levels(RunState())
        with tempfile.TemporaryDirectory() as d:
            session = FakeSession(d)
            session.policy_state = state
            begin_action(session, 'consume', {}, after_state=consumed_engine(state, Consumable('Mercury', 'Planet')))
            finish_action(session, False)
            self.assertEqual(session.policy_state.level('pair'), 1)

    def test_pending_action_recovery_replays_absolute_level_idempotently(self):
        state = fresh_levels(RunState(jokers=(Joker('Burnt Joker'),), round_discards_used=0))
        after = after_discard(state, replace(state, round_discards_used=1), [Card(4, 'C')])
        with tempfile.TemporaryDirectory() as d:
            session = FakeSession(d)
            session.policy_state = state
            begin_action(session, 'discard', {'discards_left': 3}, after_state=after)
            pending = dict(session.pending_action)
            finish_action(session, True)
            # Crash after state commit but before retiring the same journal.
            session.pending_action = pending
            finish_action(session, True)
            self.assertEqual(session.policy_state.level('high_card'), 2)

    def test_uncertainty_survives_checkpoint_and_legacy_requires_observation(self):
        after = after_play(fresh_levels(RunState(jokers=(Joker('Space Joker'),))), fresh_levels(RunState()), 'pair')
        restored = state_from_dict(json.loads(json.dumps(asdict(after), default=list)))
        self.assertEqual(restored.uncertain_levels, ('pair',))
        self.assertTrue(needs_observation(restored))
        self.assertTrue(needs_observation(state_from_dict({'levels': {'pair': 8}})))

    def test_preview_level_requires_explicit_unambiguous_label(self):
        words = [('Flush Five', .99, (340, 565)), ('Lvl.7', .99, (340, 610))]
        with patch('perception.read_text', return_value=words):
            self.assertEqual(read_hand_preview_level(None), ('flush_five', 7))
        for bad in (words[:1], words+[('Lvl.3', .99, (340, 630))]):
            with patch('perception.read_text', return_value=bad):
                with self.assertRaises(RuntimeError):
                    read_hand_preview_level(None)


if __name__ == '__main__':
    unittest.main()
