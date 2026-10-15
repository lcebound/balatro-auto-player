"""Offline regressions for missed upgrades in the ten-game batch."""
import json
import tempfile
import unittest
import cv2
import numpy as np
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from perception import shop_offer_item, joker_from_definition, parse_joker_tooltip, parse_card_tooltip, read_hand_preview
from recovery import ObservationError, action_outcome, begin_action, confirm_pending, state_from_dict
from runner import shop, sell_shop_joker, verify_selected_purchase, collect_shop_jokers
from controller import collect_position, collect_cards, verify_play_preview
from strategy import (Card, Joker, RunState, ShopItem, decide_shop, decide_play,
                      decide_discard, score_hand, shop_strength)
from tests.test_runtime_recovery import FakeSession
from strategy import _shop_scores, joker_hand_size_delta
from perception import shop_tooltip_region, tooltip_rows, read_text, identify_consumable_text
from controller import wait_consumable_hand
from controller import selection_flags
from vision import read_hand, classify_scene, hand_slots


def offer(name, price):
    from strategy import joker_name
    return ShopItem('joker', name, price, joker_from_definition(joker_name(name)))


class ScoreImprovementTests(unittest.TestCase):
    def test_actual_popcorn_score_uses_decayed_bonus(self):
        case = json.loads((Path(__file__).parent / 'fixtures/popcorn_dynamic_20261004.json').read_text('utf8'))
        state = state_from_dict(case['state'])
        joker = parse_joker_tooltip(joker_from_definition('popcorn'), case['tooltip'])
        state = replace(state, jokers=(joker,))
        cards = [Card(**card) for card in case['cards']]
        indices = case['indices']
        result = score_hand([cards[i] for i in indices], state,
                            [card for i, card in enumerate(cards) if i not in indices])
        self.assertEqual((joker.value, result.score), (16, case['actual_score']))
        self.assertNotEqual(result.score, case['old_score'])

    def test_decay_joker_direct_effects_preserve_zero_and_reject_ambiguity(self):
        cases = [('popcorn', ['Popcorn', '+0 Mult', '-4 Mult per round played'], 0),
                 ('ice_cream', ['Ice Cream', '+35 Chips', '-5 Chips for every hand played'], 35),
                 ('ramen', ['Ramen', 'X1.72 Mult,', 'loses X0.01 Mult per card discarded'], 1.72)]
        for key, labels, value in cases:
            joker = parse_joker_tooltip(joker_from_definition(key), labels)
            self.assertEqual(joker.value, value)
            self.assertEqual(joker.get('observed_value'), 1)
        plain = score_hand([Card(14, 'S')]).score
        for key, unit in [('popcorn', 'Mult'), ('ice_cream', 'Chips')]:
            joker = parse_joker_tooltip(joker_from_definition(key), ['+0 ' + unit])
            self.assertEqual(score_hand([Card(14, 'S')], RunState(jokers=(joker,))).score, plain)
            with self.assertRaises(ObservationError):
                parse_joker_tooltip(joker_from_definition(key), ['Unknown'])
        with self.assertRaises(ObservationError):
            parse_joker_tooltip(joker_from_definition('popcorn'), ['+12 Mult', 'Holographic +10 Mult'])

    def test_ramen_discard_simulations_charge_each_discarded_card(self):
        hand = [Card(rank, 'S') for rank in (14, 12, 9, 6, 2)]
        unseen = tuple(Card(rank, 'H') for rank in range(2, 15))
        for hands in (1, 2):
            state = RunState(hands_left=hands, discards_left=1, target_score=10000,
                             jokers=(Joker('Ramen', 2),), remaining_deck=unseen, draw_pile_size=13)
            seen = []
            def estimate(cards, sampled):
                discarded = len(hand) - sum(card in hand for card in cards)
                self.assertAlmostEqual(sampled.jokers[0].value, 2 - .01 * discarded)
                seen.append(discarded)
                return 1
            with patch('strategy._draw_estimate', side_effect=estimate):
                decide_discard(hand, state)
            self.assertTrue(any(count > 1 for count in seen))

    def test_recorded_missed_shop_jokers_are_recognized(self):
        cases = [
            ('drunkard', ['Not Discovered', '+1 discard', 'each round', 'Common']),
            ('ice_cream', ['Not Discovered', '+100 Chips', '-5 Chips for', 'every yhand played', 'Common']),
            ('mystic_summit', ['Not Discovered', '+15 Mult when', '0 discards', 'remaining', 'Common', 'Holographic']),
            ('juggler', ['Not t Discovered', '+1 hand size', 'Common']),
            ('juggler', ['Not Discovered', '+1 hand d size', 'Common']),
            ('abstract_joker', ['+3 Mult for', 'each Joker card', '[Currently +9 mult]']),
            ('walkie_talkie', ['Each played 10 or', 'gives +10 Chips s and', '+4 Mult when scored']),
        ]
        for key, labels in cases:
            with self.subTest(joker=key, labels=labels):
                item, confidence = shop_offer_item(labels, 4)
                self.assertEqual((item.kind, item.joker.key), ('joker', key))
                self.assertGreaterEqual(confidence, .88)
                if key == 'mystic_summit':
                    self.assertEqual(item.joker.edition, 'holographic')

    def test_short_joker_descriptions_keep_distinguishing_numbers(self):
        from item_identity import identify_description
        for labels in (['+1 hand'], ['hand size'], ['+2 hand size'],
                       ['-1 hand size'], ['+15 Mult when 1 discards remaining'],
                       ['+1 hand size', '+1 discard each round']):
            with self.subTest(labels=labels):
                with self.assertRaises(RuntimeError):
                    identify_description(labels)

    def test_face_down_placeholders_do_not_trigger_rank_condition_jokers(self):
        hidden = [Card(0, '?', face_down=True)] * 4
        plain = score_hand([Card(7, 'H'), *hidden])
        for name in ('Jolly Joker', 'Zany Joker', 'Mad Joker', 'The Duo', 'The Trio', 'The Family'):
            with self.subTest(joker=name):
                result = score_hand([Card(7, 'H'), *hidden], RunState(jokers=(Joker(name),)))
                self.assertEqual(result.score, plain.score)

    def test_last_hand_does_not_play_recorded_49_point_shortfall_with_four_discards(self):
        case = json.loads((Path(__file__).parent / 'fixtures/batch_20261003_strategy.json').read_text())[-1]
        state = state_from_dict(case['state'])
        hand = [Card(**value) for value in case['cards']]
        current = decide_play(hand, state)
        self.assertEqual((current.estimate.score, state.score_needed), (3852, 3901))
        discard = decide_discard(hand, state)
        self.assertTrue(1 <= len(discard.indices) <= 5)
        self.assertIn('last hand needs 3901', discard.reason)
        self.assertGreater(discard.utility, 0)

    def test_last_hand_single_draw_out_and_empty_deck(self):
        hand = [Card(14, 'S'), Card(14, 'H'), Card(13, 'C'), Card(12, 'D'), Card(2, 'S')]
        state = RunState(hands_left=1, discards_left=1, target_score=180,
                         draw_pile_size=1, remaining_deck=(Card(14, 'D'),))
        discard = decide_discard(hand, state)
        self.assertTrue(discard.indices)
        self.assertGreaterEqual(discard.expected_score, 180)
        self.assertEqual(decide_discard(hand, replace(state, draw_pile_size=0)).indices, ())

    def test_hidden_play_requires_unknown_preview_and_does_not_claim_category(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.read_text', return_value=[('????', .95, (310, 600))]):
                verify_play_preview(session, None, 'high_card', hidden=True)
            self.assertEqual(session.events[-1]['category'], 'unknown')
            with patch('controller.read_text', return_value=[('High Card', .99, (310, 600))]), patch('controller.time.sleep'):
                with self.assertRaisesRegex(ObservationError, 'hidden-hand preview'):
                    verify_play_preview(session, None, 'high_card', hidden=True)

    def test_house_face_down_animation_frames_are_readable_without_fabricated_identity(self):
        for number in range(1, 4):
            path = Path(__file__).parent / f'fixtures/house_face_down_{number}.png'
            image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                               (2064, 1191), interpolation=cv2.INTER_NEAREST)
            with self.subTest(frame=number):
                found = read_hand(image)
                self.assertEqual(len(found), 8)
                self.assertTrue(all(c.face_down and c.rank == 0 and c.suit == '?' for c, _, _ in found))
                self.assertEqual(selection_flags(found), set())
                with tempfile.TemporaryDirectory() as directory:
                    session = FakeSession(directory)
                    with patch('controller.read_text') as ocr, patch('controller.read_hand', return_value=found):
                        collected = collect_cards(session, image, 8)
                    ocr.assert_not_called()
                    self.assertTrue(all(c.face_down for c, _, _ in collected))
                    self.assertEqual(session.taps, [])

    def test_manacle_full_seven_card_hand_is_a_playable_scene(self):
        path = Path(__file__).parent / 'fixtures/manacle_seven_cards.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        self.assertEqual(classify_scene(image), 'hand')
        identities = [Card(r, s) for r, s in [(11, 'D'), (5, 'S'), (5, 'H'),
                                            (4, 'S'), (3, 'C'), (2, 'C'), (2, 'D')]]
        found = read_hand(image, 7, identities=identities)
        self.assertEqual(len(found), 7)
        self.assertEqual(selection_flags(found), set())
        self.assertEqual([c.identity for c, _, _ in found], [c.identity for c in identities])

    def test_partial_hand_uses_capacity_and_changed_count_blocks_selection(self):
        self.assertEqual(hand_slots(7, 7), [560, 718, 876, 1034, 1192, 1350, 1508])
        self.assertEqual(hand_slots(6, 8), [695, 831, 966, 1102, 1237, 1373])
        self.assertEqual(hand_slots(1, 8), [1034])
        with patch('perception.hand_counter', return_value=(7, 7)):
            with self.assertRaisesRegex(ObservationError, 'Hand changed'):
                read_hand(np.zeros((1191, 2064, 3), dtype=np.uint8), 8)

    def test_hidden_hand_bonus_jokers_have_distinct_descriptions(self):
        cases = [('Pair', 'Mult', 'jolly_joker'), ('Three of a Kind', 'Mult', 'zany_joker'),
                 ('Two Pair', 'Mult', 'mad_joker'), ('Straight', 'Mult', 'crazy_joker'),
                 ('Flush', 'Mult', 'droll_joker'), ('Pair', 'Chips', 'sly_joker'),
                 ('Three of a Kind', 'Chips', 'wily_joker'), ('Two Pair', 'Chips', 'clever_joker'),
                 ('Straight', 'Chips', 'devious_joker'), ('Flush', 'Chips', 'crafty_joker')]
        for hand, bonus, expected in cases:
            with self.subTest(joker=expected):
                item, _ = shop_offer_item(['Not Discovered', '+100 ' + bonus + ' if played',
                                           'hand contains', 'a ' + hand, 'Common'], 4)
                self.assertEqual(item.joker.key, expected)

    def test_selected_tilted_nine_is_not_located_at_its_body_pips(self):
        path = Path(__file__).parent / 'fixtures/selected_duplicate_diamonds.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        identities = [Card(r, s) for r, s in [(14, 'D'), (12, 'D'), (11, 'D'), (11, 'D'),
                                            (10, 'C'), (9, 'S'), (9, 'D'), (5, 'C')]]
        found = read_hand(image, 8, identities=identities)
        self.assertEqual(selection_flags(found), {0, 1, 2, 3, 6})
        self.assertEqual([card.identity for card, _, _ in found], [card.identity for card in identities])

    def test_real_neighbour_card_name_cannot_replace_shop_tooltip_identity(self):
        root = Path(__file__).parent / 'fixtures/shop_identity'
        for file, x, key in [('judgement-hover', 1163, 'judgement'),
                             ('star-hover', 1351, 'the_star'),
                             ('judgement-selected', 1163, 'judgement')]:
            with self.subTest(file=file):
                image = cv2.imdecode(np.fromfile(root / (file + '.png'), dtype=np.uint8), 1)
                image = cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)
                labels = tooltip_rows(read_text(image, shop_tooltip_region(image, x)))
                self.assertEqual(identify_consumable_text(labels)[0], key)
                if key == 'judgement':
                    with self.assertRaises(ObservationError):
                        verify_selected_purchase(image, ShopItem('tarot', 'The Star', 3), x)
                else:
                    self.assertEqual(shop_offer_item(labels, 3)[0].name, 'The Star')

    def test_hidden_consumable_descriptions_distinguish_suits_and_enhancements(self):
        for suit, key in [('Diamonds', 'the_star'), ('Clubs', 'the_moon'),
                          ('Hearts', 'the_sun'), ('Spades', 'the_world')]:
            self.assertEqual(identify_consumable_text(['Not Discovered', 'Converts up to',
                             '3 selected cards', 'to ' + suit])[0], key)
        for name, key in [('Mult Cards', 'the_empress'), ('Bonus Cards', 'the_hierophant'),
                          ('Lucky Cards', 'the_magician')]:
            self.assertEqual(identify_consumable_text(['Not Discovered', 'Enhances 2',
                             'selected cards to', name])[0], key)

    def test_consumable_waits_for_animation_before_recollecting_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.hand_counter', side_effect=[ObservationError('animation'), (8, 8)]), patch('controller.read_hand'), patch('controller.time.sleep'):
                image, count = wait_consumable_hand(session, session.frame)
            self.assertEqual(count, 8)
            self.assertEqual(session.taps, [])
            self.assertEqual(sum(e['event'] == 'consumable_animation_wait' for e in session.events), 1)

    def test_preview_joins_split_titles_without_matching_partial_category(self):
        cases = [('Two Pair', 'two_pair'), ('Straight Flush', 'straight_flush'),
                 ('Full House', 'full_house'), ('Four of a Kind', 'four_kind'),
                 ('High Card', 'high_card'), ('Flush', 'flush')]
        for title, expected in cases:
            with self.subTest(title=title):
                words = [(word, .99, (210 + i * 55, 604 + i % 2))
                         for i, word in enumerate(title.split())]
                words.append(('luL.1', .9, (490, 602)))
                with patch('perception.read_text', return_value=words):
                    self.assertEqual(read_hand_preview(None), expected)
        with patch('perception.read_text', return_value=[('Pair', .99, (240, 600)), ('Flush', .99, (240, 640))]):
            with self.assertRaises(ObservationError):
                read_hand_preview(None)

    def test_three_real_identity_errors_reproduce_and_choose_correct_cards(self):
        fixtures = json.loads((Path(__file__).parent / 'fixtures/score_identity_residuals.json').read_text(encoding='utf-8'))
        for case, category, score in zip(fixtures, ('two_pair', 'straight_flush', 'straight'), (100, 1080, 1140)):
            with self.subTest(time=case['action_time']):
                state = state_from_dict(case['state'])
                corrected = []
                for record in case['tooltips']:
                    card = parse_card_tooltip(Card(**record['old']), record['retry_labels'] or record['labels'], require_identity=True)
                    self.assertEqual(card, Card(**record['corrected']))
                    corrected.append(card)
                previous = score_hand([corrected[i] for i in case['old_indices']], state,
                                      [c for i, c in enumerate(corrected) if i not in case['old_indices']])
                self.assertEqual(previous.score, case['actual_old_score'])
                self.assertEqual(previous.category, case['original_preview_category'])
                choice = decide_play(corrected, state)
                self.assertEqual((choice.estimate.category, choice.estimate.score), (category, score))

    def test_counter_lines_cannot_fabricate_a_card_title(self):
        card = parse_card_tooltip(Card(6, 'C'), ['3/5', '0/2', 'of Clubs', 'King', '+10 chips', 'King of Clubs'], require_identity=True)
        self.assertEqual(card.identity, (13, 'C'))
        queen = parse_card_tooltip(Card(11, 'S'), ['Qveen of', 'Spades', '+10 chips', 'Qveen of Spades'], require_identity=True)
        self.assertEqual(queen.identity, (12, 'S'))

    def test_partial_tooltip_requires_retry_instead_of_a_guessed_rank(self):
        missing = ['of', 'Hearts', 'chips', 'K', 'of Hearts']
        with self.assertRaises(ObservationError):
            parse_card_tooltip(Card(2, 'H'), missing, require_identity=True)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            located = [(Card(2, 'H'), .8, (1000, 740))]
            reread = [(Card(6, 'H'), .9, (1000, 740))]
            first = [('of Hearts', .99, (1000, 550)), ('chips', .99, (1000, 590))]
            second = [('6 of Hearts', .99, (1000, 550)), ('+6 chips', .99, (1000, 590))]
            with patch('controller.read_hand', side_effect=[located, reread]), patch('controller.read_text', side_effect=[first, second]), patch('controller.time.sleep'):
                result = collect_cards(session, session.frame, 1)
            self.assertEqual(result[0][0].rank, 6)
            self.assertEqual(sum(e['event'] == 'card_identity_retry' for e in session.events), 1)

    def test_wrong_hand_preview_blocks_play_before_spending_a_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.read_hand_preview', return_value='flush'), patch('controller.time.sleep'):
                with self.assertRaises(ObservationError):
                    verify_play_preview(session, session.frame, 'straight_flush')
            self.assertEqual(session.taps, [])
            self.assertIsNone(session.pending_action)
            with patch('controller.read_hand_preview', return_value='straight_flush'):
                verify_play_preview(session, session.frame, 'straight_flush')
            self.assertEqual(session.events[-1]['event'], 'play_preview_verified')

    def test_historical_ancient_offer_is_affordable_and_bought(self):
        state = RunState(money=26, ante=2, target_score=600, jokers=(Joker('Joker', 4),))
        item, _ = shop_offer_item(['Ancient Joker', 'Each played card with',
                                 'Spade suit gives', 'X1.5 Mult when scored', 'Rare'], 8)
        self.assertEqual(decide_shop([item], state).action, 'buy')

    def test_historical_stencil_improves_two_joker_engine(self):
        state = RunState(money=35, ante=2, target_score=800,
                         jokers=(Joker('Joker', 4), Joker('Green Joker', 4)))
        self.assertEqual(decide_shop([offer('Joker Stencil', 8)], state).action, 'buy')

    def test_raised_fist_beats_unpowered_constellation(self):
        state = RunState(money=25, ante=2, target_score=800)
        decision = decide_shop([offer('Constellation', 6), offer('Raised Fist', 5)], state)
        self.assertEqual((decision.action, decision.item_index), ('buy', 1))

    def test_weak_engine_does_not_buy_pack_income_as_its_only_upgrade(self):
        state = RunState(money=25, ante=2, target_score=800)
        for name, price in [('Hallucination', 4), ('Constellation', 6), ('Golden Joker', 6)]:
            with self.subTest(name=name):
                self.assertEqual(decide_shop([offer(name, price)], state).action, 'leave')

    def test_seltzer_and_supernova_no_longer_use_generic_low_value(self):
        self.assertEqual(decide_shop([offer('Seltzer', 6)], RunState(money=25, ante=2, target_score=800)).action, 'buy')
        state = RunState(money=41, ante=2, target_score=1200,
                         jokers=(Joker('Gros Michel', 15),), played_types={'straight': 3, 'two_pair': 2})
        self.assertEqual(decide_shop([offer('Supernova', 5)], state).action, 'buy')

    def test_regular_planet_beats_unused_straight_flush_planet(self):
        state = RunState(money=26, ante=2, target_score=600,
                         jokers=(Joker('Joker', 4),), played_types={'straight': 3, 'pair': 2})
        decision = decide_shop([ShopItem('planet', 'Neptune', 3), ShopItem('planet', 'Saturn', 3)], state)
        self.assertEqual((decision.action, decision.item_index), ('buy', 1))

    def test_consumable_slots_and_budget_are_real_constraints(self):
        planet = ShopItem('planet', 'Saturn', 3)
        self.assertEqual(decide_shop([planet], RunState(money=25, consumables_held=2)).action, 'leave')
        self.assertEqual(decide_shop([offer('Cavendish', 10)], RunState(money=0)).action, 'leave')

    def test_full_slots_replace_non_scoring_joker_not_scoring_core(self):
        state = RunState(money=30, ante=3, target_score=2000,
                         jokers=(Joker('Hallucination'), Joker('Golden Joker'),
                                 Joker('Joker', 4), Joker('Gros Michel', 15), Joker('Gift Card')))
        result = decide_shop([offer('Cavendish', 10)], state)
        self.assertEqual(result.action, 'sell_then_buy')
        self.assertEqual(state.jokers[result.sell_index].name, 'Hallucination')
        eternal = replace(state, jokers=(replace(state.jokers[0], data=(('eternal', 1),)), *state.jokers[1:]))
        self.assertNotEqual(decide_shop([offer('Cavendish', 10)], eternal).sell_index, 0)

    def test_negative_joker_does_not_require_selling_a_full_slot(self):
        state = RunState(money=30, jokers=(Joker('Joker', 4),) * 5)
        item = ShopItem('joker', 'Cavendish', 10, replace(offer('Cavendish', 10).joker, edition='negative'))
        self.assertEqual(decide_shop([item], state).action, 'buy')

    def test_rerolls_use_purchase_cash_and_bounded_rich_weak_budget(self):
        weak = RunState(money=50, ante=2, target_score=800)
        self.assertEqual(decide_shop([], weak, reroll_cost=5).action, 'reroll')
        self.assertEqual(decide_shop([], replace(weak, money=20), reroll_cost=5).action, 'reroll')
        self.assertEqual(decide_shop([], replace(weak, shop_rerolls=2), reroll_cost=5).action, 'reroll')
        # Critical searches can now spend up to a quarter of available cash,
        # while retaining purchase cash and the existing four-reroll limit.
        self.assertEqual(decide_shop([], weak, reroll_cost=12).action, 'reroll')
        for state, cost in [(replace(weak, shop_rerolls=4), 5), (replace(weak, money=8), 5), (weak, 13)]:
            self.assertEqual(decide_shop([], state, reroll_cost=cost).action, 'leave')
        strong = replace(weak, jokers=(Joker('Gros Michel', 15), Joker('Cavendish'), Joker('Stuntman')))
        self.assertFalse(shop_strength(strong)['weak_engine'])
        self.assertEqual(decide_shop([], strong, reroll_cost=5).action, 'leave')

    def test_shop_does_not_value_next_round_using_spent_action_counters(self):
        state = RunState(money=50, ante=2, target_score=800,
                         jokers=(Joker('Acrobat'), Joker('Banner')))
        exhausted = replace(state, hands_left=1, discards_left=0)
        self.assertEqual(shop_strength(state), shop_strength(exhausted))
        self.assertEqual(decide_shop([offer('Burglar', 6)], state),
                         decide_shop([offer('Burglar', 6)], exhausted))

    def test_shop_comparison_accounts_for_realizable_joker_order(self):
        state = RunState(money=35, ante=2, target_score=800,
                         jokers=(Joker('Joker Stencil'), Joker('Joker', 4)))
        reversed_state = replace(state, jokers=tuple(reversed(state.jokers)))
        self.assertEqual(shop_strength(state), shop_strength(reversed_state))
        self.assertEqual(decide_shop([offer('Green Joker', 4)], state),
                         decide_shop([offer('Green Joker', 4)], reversed_state))

    def test_round_capacity_observation_preserves_deck_and_boss_rules(self):
        cases = [
            (RunState(hands_left=5, discards_left=3, deck_type='Blue Deck'), (5, 3)),
            (RunState(hands_left=7, discards_left=0, jokers=(Joker('Burglar'),)), (4, 3)),
            (RunState(hands_left=1, discards_left=3, boss='The Needle'), (4, 3)),
            (RunState(hands_left=4, discards_left=0, boss='The Water'), (4, 3)),
            (RunState(hands_left=1, discards_left=0, round_resources_observed=True), (4, 3)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            for observed, expected in cases:
                # Resource capacities are independent of level-panel OCR.
                observed = replace(observed, levels_initialized=True)
                with self.subTest(boss=observed.boss, jokers=observed.jokers):
                    with patch('controller.observed_state', return_value=(observed, 8)), patch('controller.collect_jokers', side_effect=lambda s, im, st: st), patch('controller.collect_cards', return_value=[]):
                        collected, _ = collect_position(session, observed)
                    self.assertEqual((collected.base_hands_per_round, collected.base_discards_per_round), expected)
                    self.assertTrue(collected.round_resources_observed)

    def test_sale_refuses_a_different_selected_joker(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            state = RunState(money=30, jokers=(Joker('Hallucination'), Joker('Gros Michel', 15)))
            with patch('runner.read_text', return_value=[('Gros Michel', .99, (700, 350)), ('Sell', .99, (800, 280))]), patch('runner.identify_joker_text', return_value=('gros_michel', 1.)):
                with self.assertRaises(ObservationError):
                    sell_shop_joker(session, state, 0)
            self.assertIsNone(session.pending_action)
            self.assertEqual(len(session.taps), 1)
            self.assertFalse(any('sell weak Joker' in tap[2] for tap in session.taps))

    def test_selected_purchase_must_still_have_the_expected_identity(self):
        selected = [('Mercury', .99, (1450, 600)), ('Planet', .99, (1450, 710))]
        with patch('runner.shop_tooltip_region', return_value=(535,350,1670,900)), patch('runner.read_text', return_value=selected):
            with self.assertRaises(ObservationError):
                verify_selected_purchase(None, offer('Joker', 2), 1160)

    def test_card_sharp_does_not_activate_for_an_unplayed_category(self):
        # A pair is followed by a flush without any pair. Card Sharp cannot
        # triple the flush just because this is the second sampled hand.
        pair = tuple(Card(r, s) for r, s in [(14, 'S'), (14, 'H'), (13, 'D'), (10, 'C'),
                                           (8, 'D'), (6, 'S'), (4, 'H'), (2, 'C')])
        flush = tuple(Card(r, 'H') for r in (2, 4, 6, 8, 10, 12, 13, 14))
        state = RunState(jokers=(Joker('Gros Michel', 15), Joker('Card Sharp')))
        without = replace(state, jokers=(Joker('Gros Michel', 15),))
        with patch('strategy._shop_probe_hands', return_value=(pair, flush, flush, flush)):
            scores = _shop_scores(state)
            baseline = _shop_scores(without)
        self.assertEqual(scores[0], baseline[0])
        self.assertEqual(scores[1], baseline[1])
        self.assertEqual(scores[2], baseline[2] * 3)

    def test_shop_hand_size_tracks_purchase_sale_and_turtle_decay(self):
        turtle5 = parse_joker_tooltip(Joker('Turtle Bean'), ['Turtle Bean', '+5 hand size', 'reduces by 1 each round'])
        turtle4 = parse_joker_tooltip(Joker('Turtle Bean'), ['Turtle Bean', '+4 hand size', 'reduces by 1 each round'])
        self.assertEqual(joker_hand_size_delta((turtle5,), (turtle4,)), -1)
        self.assertEqual(joker_hand_size_delta((Joker('Stuntman'),), ()), 2)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            state = RunState(hand_size=13, jokers=(turtle5,))
            with patch('runner.collect_jokers', return_value=replace(state, jokers=(turtle4, Joker('Juggler')))):
                observed = collect_shop_jokers(session, session.frame, state)
            self.assertEqual(observed.hand_size, 13)

    def test_existing_same_name_joker_cannot_hide_a_wrong_purchase(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            initial = RunState(money=20, ante=3, target_score=3000, jokers=(Joker('Joker', 4),))
            session.policy_state = initial
            labels = [('Joker', .99, (1450, 600)), ('+4 Mult', .99, (1450, 650)), ('Common', .99, (1450, 710))]
            reads = [[('$2', .99, (1160, 460))], labels, [('Buy', .99, (1160, 710))], labels]
            wrong_roster = replace(initial, jokers=(*initial.jokers, Joker('Green Joker')))
            with patch('runner.read_numbers', side_effect=[{'money': 20}, {'money': 18}]), patch('runner.collect_jokers', side_effect=[initial, wrong_roster]), patch('runner.read_jokers', return_value=[]), patch('runner.shop_tooltip_region', return_value=(535,350,1670,900)), patch('runner.read_text', side_effect=reads), patch('runner.read_reroll_button', return_value=None), patch('runner.consumable_counter', return_value=(0, 2)), patch('runner.time.sleep'), patch('vision.classify_scene', return_value='shop'), patch('perception.read_numbers', return_value={'money': 18}), patch('perception.consumable_counter', return_value=(0, 2)), patch('perception.joker_counter', return_value=(2, 5)):
                with self.assertRaises(ObservationError):
                    shop(session, session.frame)
            self.assertEqual(session.policy_state.jokers[-1].name, 'Green Joker')
            self.assertIsNone(session.pending_action)
            self.assertEqual(sum(e['event'] == 'action_confirmed' for e in session.events), 1)
            self.assertTrue(any(e['event'] == 'purchase_identity_mismatch' for e in session.events))

    def test_mouth_prefers_repeatable_pair_to_near_clear_straight(self):
        hand = [Card(r, s) for r, s in [(2, 'S'), (3, 'H'), (4, 'D'), (5, 'C'),
                                      (6, 'S'), (14, 'H'), (14, 'D'), (8, 'C')]]
        state = RunState(boss='The Mouth', jokers=(Joker('Gros Michel', 15),), target_score=1050,
                         discards_left=0, remaining_deck=(Card(14, 'C'), Card(14, 'S'),
                                                         Card(13, 'C'), Card(13, 'D'), Card(12, 'D')))
        decision = decide_play(hand, state)
        self.assertEqual(decision.estimate.category, 'pair')
        self.assertIn('repeat probability', decision.reason)
        # If a straight already clears, no speculative repeatability can beat it.
        clear = decide_play(hand, replace(state, target_score=1000))
        self.assertEqual(clear.estimate.category, 'straight')
        self.assertGreaterEqual(clear.estimate.score, 1000)

    def test_no_discard_when_draw_pile_is_empty(self):
        hand = [Card(14, 'S'), Card(13, 'H'), Card(7, 'D'), Card(2, 'C')]
        self.assertEqual(decide_discard(hand, RunState(draw_pile_size=0)).indices, ())

    def test_offer_tooltip_preserves_suit_edition_and_eternal(self):
        item, _ = shop_offer_item(['Ancient Joker', 'Heart suit gives X1.5 Mult', 'Rare', 'Polychrome'], 8)
        self.assertEqual(item.joker.get('suit'), 1)
        self.assertEqual(item.joker.edition, 'polychrome')
        self.assertEqual(parse_joker_tooltip(Joker('Joker'), ['Joker', 'Eternal']).get('eternal'), 1)
        planet, _ = shop_offer_item(['Saturn', 'Planet', 'Level up Straight'], 3)
        self.assertEqual((planet.kind, planet.name), ('planet', 'Saturn'))
        # Random Joker creation now has an observed-roster workflow. Random
        # destructive Spectrals still require an unsupported use workflow.
        judgement, _ = shop_offer_item(['Judgement', 'Tarot'], 3)
        self.assertEqual((judgement.kind, judgement.name), ('tarot', 'Judgement'))
        with self.assertRaises(RuntimeError):
            shop_offer_item(['Ectoplasm', 'Spectral'], 3)

    def test_sale_and_reroll_outcomes_are_confirmed_by_real_counters(self):
        sale = {'kind': 'sell', 'before': {'joker_count': 5, 'money': 20}}
        self.assertTrue(action_outcome(sale, 'shop', {'joker_count': 4, 'money': 20}))
        self.assertIsNone(action_outcome(sale, 'unknown', {'joker_count': 4, 'money': 22}))
        reroll = {'kind': 'reroll', 'before': {'money': 25, 'reroll_cost': 5}}
        self.assertTrue(action_outcome(reroll, 'shop', {'money': 20}))
        self.assertFalse(action_outcome(reroll, 'shop', {'money': 25, 'reroll_cost': 5}))
        free = {'kind': 'reroll', 'before': {'money': 25, 'reroll_cost': 0}}
        self.assertTrue(action_outcome(free, 'shop', {'money': 25, 'reroll_cost': 5}))

    def test_reroll_counter_survives_confirmation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            before = RunState(money=25)
            begin_action(session, 'reroll', {'money': 25, 'reroll_cost': 5},
                         after_state=replace(before, money=20, shop_rerolls=1))
            with patch('vision.classify_scene', return_value='shop'), patch('perception.read_numbers', return_value={'money': 20}), patch('perception.read_reroll_button', return_value=((750, 630), 6)):
                confirm_pending(session)
                confirm_pending(session)
            saved = json.loads((session.run_dir / 'policy-state.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['shop_rerolls'], 1)
            self.assertEqual(saved['money'], 20)
            self.assertEqual(sum(e['event'] == 'action_confirmed' for e in session.events), 1)

    def test_full_joker_slots_do_not_block_a_planet_purchase(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = RunState(money=20, jokers=(Joker('Joker', 4),) * 5,
                                            played_types={'pair': 10}, ante=3, target_score=3000)
            text_reads = [[('$3', .99, (1160, 460))],
                          [('Mercury', .99, (1450, 600)), ('Planet', .99, (1450, 710))],
                          [('Buy', .99, (1160, 710))],
                          [('Mercury', .99, (1450, 600)), ('Planet', .99, (1450, 710))], []]
            with patch('runner.read_numbers', side_effect=[{'money': 20}, {'money': 17}]), patch('runner.collect_jokers', side_effect=lambda s, im, st: st), patch('runner.read_jokers', return_value=[]), patch('runner.shop_tooltip_region', return_value=(535,350,1670,900)), patch('runner.read_text', side_effect=text_reads), patch('runner.read_reroll_button', return_value=None), patch('runner.consumable_counter', side_effect=[(0, 2), (0, 2), (1, 2)]), patch('runner.time.sleep'), patch('vision.classify_scene', return_value='shop'), patch('perception.read_numbers', return_value={'money': 17}), patch('perception.consumable_counter', return_value=(1, 2)), patch('perception.joker_counter', return_value=(5, 5)):
                shop(session, session.frame)
            self.assertEqual(session.policy_state.consumables_held, 1)
            self.assertEqual(session.policy_state.money, 17)
            self.assertEqual(len(session.policy_state.jokers), 5)
            self.assertEqual(sum(e['event'] == 'action_confirmed' for e in session.events), 1)
            self.assertTrue(any('planet purchase' in tap[2] for tap in session.taps))


if __name__ == '__main__':
    unittest.main()
