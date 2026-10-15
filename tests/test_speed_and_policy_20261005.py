"""Offline checks for roster caching and the two completed games' shop faults."""
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from bot import Session
from controller import collect_jokers
from joker_cache import (JokerCache, STABLE_WITHIN_BLIND, roster_patch,
                         round_token, cached_joker, same_slot_art)
from perception import read_text
from planning import (ShopForecast, RoundForecast, engine_profile,
                      replacement_reduces_survival, advance_projected_play)
from recovery import begin_action, finish_action
from strategy import (Card, Joker, RunState, ShopItem, decide_play,
                      decide_discard, decide_shop, score_hand)
from tests.test_runtime_recovery import FakeSession


def forecast(safe=12000, target=10000, sustainable=None, boss_safe=None):
    state = RunState(ante=4)
    def round_result(score, target):
        return RoundForecast(target, '', score, score, float(score >= target),
                             score / 4, 4, 3, (), (score,))
    return ShopForecast(round_result(safe, target),
                        round_result(boss_safe, 40000) if boss_safe is not None else None,
                        safe if sustainable is None else sustainable, engine_profile(state))


class RosterCacheTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(5).integers(
            0, 256, (1191, 2064, 3), dtype=np.uint8)
        self.state = RunState(ante=4, blind_index=0, jokers=(Joker('Scholar'), Joker('Square Joker', 4)))
        self.cache = JokerCache(round_token(self.state), 5, self.state.jokers,
                                tuple(roster_patch(self.frame, 2, i) for i in range(2)))

    def test_same_roster_and_small_wobble_reuse_but_colour_change_does_not(self):
        self.assertTrue(self.cache.reusable(self.frame, self.state, 2, 5))
        shifted = self.frame.copy()
        shifted[128:313, 790:900] = self.cache.patches[0]
        self.assertTrue(same_slot_art(shifted, 2, 0, self.cache.patches[0]))
        changed = self.frame.copy()
        changed[105:330, 767:917] = 0
        self.assertFalse(self.cache.reusable(changed, self.state, 2, 5))
        blank = np.zeros_like(self.frame)
        self.assertFalse(same_slot_art(blank, 2, 0, roster_patch(blank, 2, 0)))

    def test_changed_lower_emblem_cannot_reuse_an_identical_upper_face(self):
        changed = self.frame.copy()
        changed[265:305, 802:882] = 0
        self.assertFalse(same_slot_art(changed, 2, 0, self.cache.patches[0]))

    def test_count_capacity_round_and_boss_effects_invalidate(self):
        self.assertFalse(self.cache.reusable(self.frame, self.state, 1, 5))
        self.assertFalse(self.cache.reusable(self.frame, self.state, 2, 6))
        self.assertFalse(self.cache.reusable(self.frame, replace(self.state, blind_index=1), 2, 5))
        for boss in ('The Crimson Heart', 'The Amber Acorn', 'crimson_heart', 'amber_acorn'):
            state = replace(self.state, boss=boss)
            cache = replace(self.cache, token=round_token(state))
            self.assertFalse(cache.reusable(self.frame, state, 2, 5))
            disabled = replace(state, boss_disabled=True)
            self.assertTrue(replace(cache, token=round_token(disabled)).reusable(self.frame, disabled, 2, 5))

    def test_static_slot_skips_tooltip_but_growth_slot_is_reobserved(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = self.frame
            session.joker_cache = self.cache
            with patch('controller.joker_counter', return_value=(2, 5)), \
                 patch('controller.read_jokers') as atlas, \
                 patch('controller.read_text', return_value=[]), \
                 patch('controller.identify_joker_text', return_value=('square_joker', 1)), \
                 patch('controller.joker_from_definition', return_value=Joker('Square Joker')), \
                 patch('controller.parse_joker_tooltip', return_value=Joker('Square Joker', 8)) as tooltip:
                actual = collect_jokers(session, self.frame, self.state)
            atlas.assert_not_called()
            tooltip.assert_called_once()
            self.assertEqual(actual.jokers, (Joker('Scholar'), Joker('Square Joker', 8)))
            log = session.events[-1]
            self.assertEqual(log['reused_slots'], [0])
            self.assertEqual(log['refreshed_slots'], [1])

    def test_unknown_and_counter_jokers_are_never_assumed_static(self):
        for key in ('green_joker', 'square_joker', 'ramen', 'loyalty_card',
                    'lucky_cat', 'vampire', 'yorick', 'flash_card', 'unknown_future_joker'):
            self.assertNotIn(key, STABLE_WITHIN_BLIND)

    def test_money_bonus_updates_without_tooltip_and_keeps_edition(self):
        bull = Joker('Bull', 120, edition='foil')
        current = cached_joker(bull, replace(self.state, money=59))
        self.assertEqual(current.value, 118)
        self.assertEqual(current.edition, 'foil')
        hand = [Card(10, 'H')]
        self.assertEqual(score_hand(hand, replace(self.state, money=59, jokers=(current,))),
                         score_hand(hand, replace(self.state, money=59, jokers=(bull,))))

    def test_same_count_transaction_invalidates_cache(self):
        for kind in ('buy', 'sell', 'consume'):
            with tempfile.TemporaryDirectory() as directory:
                session = FakeSession(directory)
                session.joker_cache = self.cache
                begin_action(session, kind, {}, after_state=self.state)
                finish_action(session, True)
                self.assertIsNone(session.joker_cache)

    def test_forced_order_verification_bypasses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.joker_cache = self.cache
            with patch('controller.joker_counter', return_value=(0, 5)), \
                 patch('controller.read_jokers', return_value=[]) as atlas:
                self.assertEqual(collect_jokers(session, self.frame, self.state, force=True).jokers, ())
            atlas.assert_called_once()


class SpeedTests(unittest.TestCase):
    def test_foreground_window_returns_without_restore_or_sleep(self):
        session = Session.__new__(Session)
        session.hwnd = 7
        with patch('bot.win32gui.IsWindow', return_value=True), \
             patch('bot.win32gui.GetForegroundWindow', return_value=7), \
             patch('bot.win32gui.IsIconic', return_value=False), \
             patch('bot.win32gui.ShowWindow') as restore, \
             patch('bot.time.sleep') as sleep:
            session.focus()
        restore.assert_not_called()
        sleep.assert_not_called()

    def test_exact_pixel_ocr_cache_respects_changed_pixels_mode_and_region(self):
        result = SimpleNamespace(txts=['Joker'], scores=[.99], boxes=None)
        engine = Mock(return_value=result)
        frame = np.zeros((20, 20, 3), np.uint8)
        with patch('perception.ocr_engine', return_value=engine):
            a = read_text(frame)
            a.clear()
            self.assertEqual(len(read_text(frame.copy())), 1)
            self.assertEqual(engine.call_count, 1)
            frame[0, 0] = 1
            read_text(frame)
            read_text(frame, line=True)
            read_text(frame, (1, 1, 10, 10))
            self.assertEqual(engine.call_count, 4)

    def test_reusing_play_plan_keeps_discard_decision_identical(self):
        hand = [Card(r, 'H') for r in (10, 11, 12, 13, 14)] + [Card(2, 'S')]
        state = RunState(target_score=1000, hands_left=2)
        current = decide_play(hand, state)
        original = decide_discard(hand, state)
        with patch('strategy.decide_play', side_effect=AssertionError('duplicate search')):
            reused = decide_discard(hand, state, current=current)
        self.assertEqual(original, reused)


class ShopSurvivalTests(unittest.TestCase):
    def test_arm_downgrade_persists_between_forecast_plays_and_stops_at_one(self):
        hand = [Card(r, 'H' if r == 2 else 'S') for r in (2, 3, 4, 5, 6)]
        state = RunState(boss='The Arm', levels={'straight': 3})
        first = score_hand(hand, state)
        after = advance_projected_play(state, hand, tuple(range(5)), first)
        self.assertEqual(after.level('straight'), 2)
        second = score_hand(hand, after)
        self.assertLess(second.score, first.score)
        after = advance_projected_play(after, hand, tuple(range(5)), second)
        self.assertEqual(after.level('straight'), 1)
        after = advance_projected_play(after, hand, tuple(range(5)), score_hand(hand, after))
        self.assertEqual(after.level('straight'), 1)
        disabled = replace(state, boss_disabled=True)
        self.assertEqual(advance_projected_play(disabled, hand, tuple(range(5)),
                                               score_hand(hand, disabled)).level('straight'), 3)

    def test_expiring_popcorn_support_is_weak_even_when_next_blind_is_covered(self):
        old = forecast(safe=5914, target=5000, sustainable=1879)
        self.assertTrue(old.weak)
        self.assertAlmostEqual(old.coverage, 1879 / 5000)

    def test_replacing_only_polychrome_for_easier_blind_is_rejected(self):
        old = forecast(safe=28399, target=30000, boss_safe=29200)
        new = forecast(safe=31241, target=30000, boss_safe=27200)
        self.assertTrue(replacement_reduces_survival(old, new))
        # Even if sustainable support is the worst bottleneck, retain the
        # independent Boss constraint rather than comparing only the minimum.
        self.assertTrue(replacement_reduces_survival(
            replace(old, sustainable_score=10000), replace(new, sustainable_score=11000)))
        self.assertFalse(replacement_reduces_survival(old, replace(new, boss_round=old.boss_round)))

    def test_shop_enforces_replacement_survival_gate(self):
        state = RunState(ante=6, money=59, total_joker_slots=1,
                         jokers=(Joker('Smiley Face', edition='polychrome'),))
        old = forecast(safe=28399, target=30000, boss_safe=29200)
        new = forecast(safe=31241, target=30000, boss_safe=27200)
        with patch('planning.shop_forecast', side_effect=[old, new]):
            decision = decide_shop([ShopItem('joker', 'Joker', 3, joker=Joker('Joker', 4))], state)
        self.assertEqual(decision.action, 'leave')

    def test_editions_and_baseball_synergy_are_counted(self):
        state = RunState(ante=6, jokers=(Joker('Smiley Face', edition='polychrome'),))
        self.assertEqual(engine_profile(state).multiplicative_mult, 1)
        self.assertNotIn('no_multiplier_or_retrigger', engine_profile(state).weaknesses)
        baseball = Joker('Baseball Card', rarity='rare')
        self.assertEqual(engine_profile(replace(state, jokers=(baseball,))).multiplicative_mult, 0)
        bull = Joker('Bull', rarity='uncommon')
        self.assertEqual(engine_profile(replace(state, jokers=(baseball, bull))).multiplicative_mult, 1)
        self.assertEqual(engine_profile(replace(state, jokers=(Joker('Drunkard', edition='foil'),))).chips, 1)

    def test_critical_reroll_can_cost_twelve_but_has_money_and_count_limits(self):
        state = RunState(ante=6, money=59, jokers=(Joker('Bull'),), shop_rerolls=2)
        with patch('planning.shop_forecast', return_value=forecast(safe=12451, target=40000)):
            self.assertEqual(decide_shop([], state, reroll_cost=12).action, 'reroll')
            self.assertEqual(decide_shop([], replace(state, shop_rerolls=4), reroll_cost=12).action, 'leave')
            self.assertEqual(decide_shop([], replace(state, money=22), reroll_cost=12).action, 'leave')
            self.assertEqual(decide_shop([], state, reroll_cost=21).action, 'leave')


if __name__ == '__main__':
    unittest.main()
