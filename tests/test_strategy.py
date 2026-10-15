"""Regressions for observed failures and game-source scoring rules."""
import unittest
import json
from pathlib import Path
from dataclasses import replace
from strategy import (Card, Joker, RunState, classify, score_hand, decide_play,
                      decide_discard, optimize_play_order, joker_name,
                      resolve_abilities, decide_shop, ShopItem)
from strategy import optimize_joker_order
from strategy import locked_mouth_hand
from recovery import state_from_dict
from perception import parse_card_tooltip, parse_joker_tooltip, parse_levels
from perception import joker_definitions, identify_joker_text, joker_aliases, joker_slots
from controller import selection_flags
from consumables import Consumable, transformed_hand, decide_consumable


def cards(text):
    ranks = {"A": 14, "K": 13, "Q": 12, "J": 11}
    return [Card(ranks.get(token[:-1], int(token[:-1]) if token[:-1].isdigit() else 0), token[-1])
            for token in text.split()]


class StrategyTests(unittest.TestCase):
    def test_observed_straight_flush(self):
        hand = cards("AD KD QH 7H 6H 5H 4H 3H")
        result = decide_play(hand, RunState(jokers=(Joker("Joker"),), target_score=600))
        self.assertEqual(result.estimate.category, "straight_flush")
        self.assertEqual(result.indices, (3, 4, 5, 6, 7))
        self.assertEqual(result.estimate.score, 1500)

    def test_four_kind_over_full_house(self):
        result = decide_play(cards("AS AH AC AD KH KC KD"))
        self.assertEqual(result.estimate.category, "four_kind")
        self.assertEqual(result.estimate.score, 728)

    def test_thirteen_cards_find_flush(self):
        hand = cards("KH KD QH JH JC 10S 5H 5D 4C 3C 2S 2H 2C")
        result = decide_play(hand, RunState(hand_size=13, hands_left=1))
        self.assertEqual(result.estimate.category, "flush")
        self.assertEqual(len(result.indices), 5)

    def test_all_twelve_categories(self):
        examples = {
            "high_card": "AS", "pair": "AS AH", "two_pair": "AS AH KS KH",
            "three_kind": "AS AH AD", "straight": "2S 3H 4D 5C 6S",
            "flush": "2S 4S 6S 8S JS", "full_house": "AS AH AD KS KH",
            "four_kind": "AS AH AD AC", "straight_flush": "AS 2S 3S 4S 5S",
            "five_kind": "AS AH AD AC AS", "flush_house": "AS AS AS KS KS",
            "flush_five": "AS AS AS AS AS"}
        for expected, text in examples.items():
            with self.subTest(expected=expected):
                self.assertEqual(classify(cards(text)).category, expected)

    def test_four_fingers_composite_straight_flush(self):
        state = RunState(jokers=(Joker("Four Fingers"),))
        result = classify(cards("2H 3H 4H 5C 9H"), state)
        self.assertEqual(result.category, "straight_flush")
        self.assertEqual(result.scoring, (0, 1, 2, 3, 4))

    def test_four_fingers_duplicate_straight_scores_all(self):
        result = classify(cards("2H 3D 4S 5C 5H"), RunState(jokers=(Joker("Four Fingers"),)))
        self.assertEqual(result.category, "straight")
        self.assertEqual(len(result.scoring), 5)

    def test_shortcut_and_low_ace(self):
        state = RunState(jokers=(Joker("Shortcut"),))
        self.assertEqual(classify(cards("AS 2H 4C 6D 8S"), state).category, "straight")
        self.assertEqual(classify(cards("QS KH AC 2D 3S"), state).category, "high_card")

    def test_duplicated_cards_are_legal(self):
        self.assertEqual(classify(cards("AS AS AS AS AS")).category, "flush_five")

    def test_enhancements(self):
        expected = {"bonus": 37, "mult": 35, "glass": 14, "stone": 55, "lucky": 35}
        for enhancement, score in expected.items():
            with self.subTest(enhancement=enhancement):
                self.assertEqual(score_hand([Card(2, "S", enhancement=enhancement)]).score, score)

    def test_wild_completes_flush(self):
        hand = cards("AS 2H 3H 4H 5H")
        hand[0] = replace(hand[0], enhancement="wild")
        self.assertEqual(classify(hand).category, "straight_flush")

    def test_red_glass_retriggers_chips_and_multiplier(self):
        self.assertEqual(score_hand([Card(14, "S", enhancement="glass", seal="red")]).score, 108)

    def test_held_red_steel_with_mime_adds_retrigger(self):
        score = score_hand(cards("AS"), RunState(jokers=(Joker("Mime"),)),
                           [Card(13, "S", enhancement="steel", seal="red")]).score
        self.assertEqual(score, 54)

    def test_blueprint_mime_baron(self):
        state = RunState(jokers=(Joker("Blueprint"), Joker("Mime"), Joker("Baron")))
        score = score_hand(cards("AS"), state, [Card(13, "S", enhancement="steel", seal="red")]).score
        self.assertAlmostEqual(score, 16 * 1.5 ** 8)

    def test_copier_cycles_do_not_recurse(self):
        self.assertEqual(resolve_abilities((Joker("Blueprint"), Joker("Brainstorm"))), (None, None))

    def test_blueprint_does_not_copy_passive(self):
        state = RunState(jokers=(Joker("Blueprint"), Joker("Four Fingers")))
        self.assertEqual(state.effect_count("four_fingers"), 1)

    def test_blueprint_keeps_own_edition(self):
        state = RunState(jokers=(Joker("Blueprint", edition="holographic"), Joker("Joker", edition="foil")))
        score = score_hand(cards("AS"), state)
        self.assertEqual(score.score, (16 + 50) * (1 + 10 + 4 + 4))

    def test_joker_holographic_precedes_ability(self):
        state = RunState(jokers=(Joker("The Duo", edition="holographic"),))
        self.assertEqual(score_hand(cards("2S 2H"), state).score, 336)

    def test_joker_order_is_observable(self):
        pair = cards("2S 2H")
        good = RunState(jokers=(Joker("Joker"), Joker("The Duo")))
        bad = replace(good, jokers=tuple(reversed(good.jokers)))
        self.assertGreater(score_hand(pair, good).score, score_hand(pair, bad).score)

    def test_photograph_and_hanging_chad(self):
        state = RunState(jokers=(Joker("Photograph"), Joker("Hanging Chad")))
        self.assertEqual(score_hand(cards("QH"), state).score, 280)

    def test_pareidolia_face_effects(self):
        state = RunState(jokers=(Joker("Pareidolia"), Joker("Scary Face")))
        self.assertEqual(score_hand(cards("2H"), state).score, 37)

    def test_stone_does_not_trigger_scholar(self):
        state = RunState(jokers=(Joker("Scholar"),))
        self.assertEqual(score_hand([Card(14, "S", enhancement="stone")], state).score, 55)

    def test_raised_fist_retriggers_in_held_phase(self):
        held = [Card(6, "H", enhancement="steel", seal="red")]
        state = RunState(jokers=(Joker("Raised Fist"),))
        self.assertEqual(score_hand(cards("AS"), state, held).score, 516)

    def test_held_queen_king_order(self):
        state = RunState(jokers=(Joker("Shoot the Moon"), Joker("Baron")))
        self.assertEqual(score_hand(cards("AS"), state, cards("QH KS")).score, 336)
        self.assertEqual(score_hand(cards("AS"), state, cards("KS QH")).score, 232)

    def test_flower_pot_requires_four_distinct_cards(self):
        state = RunState(jokers=(Joker("Flower Pot"),))
        wild = Card(14, "H", enhancement="wild")
        self.assertEqual(score_hand([wild], state).score, 16)
        hand = cards("AS AH AD AC")
        self.assertEqual(score_hand(hand, state).score, 728 * 3)

    def test_seeing_double_is_not_a_clubs_flush(self):
        state = RunState(jokers=(Joker("Seeing Double"),))
        self.assertEqual(score_hand(cards("2C 2H"), state).score, 56)
        self.assertEqual(score_hand(cards("2C 2C"), state).score, 28)

    def test_card_sharp_only_counts_current_round(self):
        state = RunState(jokers=(Joker("Card Sharp"),), played_types={"pair": 4})
        self.assertEqual(score_hand(cards("2C 2H"), state).score, 28)
        self.assertEqual(score_hand(cards("2C 2H"), replace(state, round_played_types={"pair": 1})).score, 84)

    def test_scalers_apply_before_current_hand(self):
        for joker, expected in [(Joker("Green Joker", 0), 14), (Joker("Ride the Bus", 5), 49),
                                (Joker("Supernova"), 14)]:
            with self.subTest(joker=joker.name):
                self.assertEqual(score_hand(cards("2S"), RunState(jokers=(joker,))).score, expected)
        self.assertEqual(score_hand(cards("KS"), RunState(jokers=(Joker("Ride the Bus", 5),))).score, 15)

    def test_late_small_hands_depend_on_real_engine(self):
        hand = cards("AS AH AC AD KH KC KD")
        state = RunState(levels={"pair": 20})
        result = decide_play(hand, state)
        self.assertEqual(result.estimate.category, "pair")
        self.assertGreater(result.estimate.score, 728)

    def test_made_hand_on_pace_preserves_discards(self):
        state = RunState(target_score=300, hands_left=4, discards_left=3)
        result = decide_discard(cards("AS AH KS KH 2C 3D 7S 9H"), state)
        self.assertFalse(result.indices)

    def test_banner_after_discard_has_one_fewer_discard(self):
        state = RunState(jokers=(Joker("Banner"),), discards_left=3)
        self.assertEqual(score_hand(cards("AS"), state).score - score_hand(cards("AS"), replace(state, discards_left=2)).score, 30)

    def test_psychic_requires_five_cards(self):
        result = decide_play(cards("AS AH 2H 3C 4D 6S 7H 8C"), RunState(boss="The Psychic"))
        self.assertEqual(len(result.indices), 5)

    def test_eye_and_mouth_restrict_categories(self):
        for boss in ("The Eye", "The Mouth"):
            state = RunState(boss=boss, boss_seen_types=frozenset({"pair"}))
            result = decide_play(cards("AS AH 2H 3C 4D"), state)
            self.assertEqual(result.estimate.category == "pair", boss == "The Mouth")

    def test_mouth_actual_stall_uses_zero_score_cycle_and_keeps_straight_core(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/mouth_no_scoring_hand.json').read_text(encoding='utf-8'))
        state = state_from_dict(fixture['state'])
        hand = [Card(**card) for card in fixture['cards']]
        result = decide_play(hand, state)
        self.assertEqual(result.estimate.score, 0)
        self.assertIn('zero-score cycle', result.reason)
        self.assertTrue(1 <= len(result.indices) <= 5)
        retained_ranks = {card.rank for i, card in enumerate(hand) if i not in result.indices}
        self.assertTrue({6, 7, 8} <= retained_ranks)
        self.assertEqual(locked_mouth_hand(state), 'straight')

    def test_rejected_mouth_hand_never_unlocks_another_category(self):
        state = RunState(boss='The Mouth', boss_target_hand='straight',
                         boss_seen_types=frozenset({'straight', 'pair'}),
                         round_played_types={'straight': 1, 'pair': 1})
        self.assertEqual(score_hand(cards('AS AH'), state).score, 0)
        self.assertGreater(score_hand(cards('6S 7H 8D 9C 10S'), state).score, 0)
        # Old checkpoints migrate from the first round-history entry.
        legacy = replace(state, boss_target_hand='')
        self.assertEqual(locked_mouth_hand(legacy), 'straight')
        self.assertEqual(score_hand(cards('AS AH'), legacy).score, 0)

    def test_zero_score_cycle_preserves_known_one_card_straight_draw(self):
        hand = cards('AH AD 10D 8C 8D 7S 6C 3S')
        state = RunState(boss='The Mouth', boss_target_hand='straight',
                         hands_left=3, discards_left=0, remaining_deck=(Card(9, 'H'),),
                         draw_pile_size=1, target_score=1600, round_score=1539,
                         jokers=(Joker('Gros Michel', 15),))
        result = decide_play(hand, state)
        retained = [c for i, c in enumerate(hand) if i not in result.indices]
        next_hand = [*retained, Card(9, 'H')]
        follow_up = decide_play(next_hand, replace(state, hands_left=2))
        self.assertEqual(result.estimate.score, 0)
        self.assertEqual(follow_up.estimate.category, 'straight')
        self.assertGreaterEqual(follow_up.estimate.score, state.score_needed)

    def test_zero_score_final_hand_empty_deck_and_forced_card_are_playable(self):
        hand = cards('AS AH 8D 6S')
        for state in (
            RunState(boss='The Mouth', boss_target_hand='straight', hands_left=1, discards_left=0),
            RunState(boss='The Mouth', boss_target_hand='straight', draw_pile_size=0),
            RunState(boss='The Mouth', boss_target_hand='straight', forced_card_index=2),
            RunState(boss='The Eye', boss_seen_types=frozenset({'pair', 'high_card'})),
            RunState(boss='The Psychic'),
        ):
            with self.subTest(state=state):
                result = decide_play(hand, state)
                self.assertEqual(result.estimate.score, 0)
                self.assertTrue(1 <= len(result.indices) <= 5)
                if state.forced_card_index >= 0:
                    self.assertIn(state.forced_card_index, result.indices)

    def test_flint_halves_base_not_jokers(self):
        state = RunState(boss="The Flint", jokers=(Joker("Joker"),))
        self.assertEqual(score_hand(cards("2S 2H"), state).score, 45)

    def test_arm_decreases_level(self):
        state = RunState(boss="The Arm", levels={"pair": 2})
        self.assertEqual(score_hand(cards("2S 2H"), state).score, 28)

    def test_wild_is_debuffed_by_each_suit_boss(self):
        hand = [Card(14, "S", enhancement="wild")]
        for boss in ("The Head", "The Club", "The Goad", "The Window"):
            self.assertEqual(score_hand(hand, RunState(boss=boss)).score, 5)

    def test_forced_card_inclusion(self):
        result = decide_play(cards("AS AH AC AD KH 2D"), RunState(forced_card_index=5))
        self.assertIn(5, result.indices)

    def test_blue_seal_requires_free_space_and_round_clear(self):
        state = RunState(target_score=10, consumables_held=2)
        held = [Card(2, "H", seal="blue")]
        self.assertEqual(score_hand(cards("AS"), state, held).cash, 0)
        self.assertEqual(score_hand(cards("AS"), replace(state, consumables_held=1), held).cash, 3)

    def test_random_scores_have_conservative_alternative(self):
        hand = [Card(2, "H", enhancement="lucky")]
        state = RunState(jokers=(Joker("Oops! All 6s"),))
        self.assertEqual(score_hand(hand, state).score, 63)
        self.assertEqual(score_hand(hand, replace(state, score_mode="conservative")).score, 7)

    def test_plasma_balances_final_chips_and_mult(self):
        state = RunState(deck_type="Plasma Deck")
        self.assertEqual(score_hand(cards("AS"), state).score, 8.5 ** 2)

    def test_card_order_improves_photo_chad(self):
        hand = cards("AS KH QS JS 10S")
        state = RunState(jokers=(Joker("Photograph"), Joker("Hanging Chad")))
        result = optimize_play_order(hand, state, decide_play(hand, state))
        self.assertEqual(hand[result.play_order[0]].rank, 13)
        self.assertGreater(result.estimate.score, score_hand(hand, state).score)

    def test_shop_respects_budget_and_full_slots(self):
        items = [ShopItem("joker", "Cavendish", 10, Joker("Cavendish"))]
        self.assertEqual(decide_shop(items, RunState(money=0)).action, "leave")

    def test_normalized_real_joker_names(self):
        self.assertEqual(joker_name("Oops! All 6s"), "oops_all_6s")
        self.assertEqual(joker_name("Driver's License"), "drivers_license")
        self.assertEqual(joker_name("Séance"), "seance")

    def test_modifier_tooltip(self):
        card = parse_card_tooltip(Card(13, "H"), ["Steel Card", "Red Seal", "Polychrome", "+5 extra chips"])
        self.assertEqual((card.enhancement, card.seal, card.edition, card.bonus_chips), ("steel", "red", "polychrome", 5))

    def test_scaling_tooltip_preserves_full_xmult(self):
        joker = parse_joker_tooltip(Joker("Campfire"), ["Campfire", "Currently X2.5 Mult"])
        self.assertEqual(joker.value, 2.5)
        self.assertEqual(score_hand(cards("AS"), RunState(jokers=(joker,))).score, 40)

    def test_level_row_parser(self):
        words = [("Lvl. 7", .99, (600, 400)), ("Pair", .99, (800, 400)),
                 ("100", .99, (1100, 400)), ("8", .99, (1200, 400)), ("13", .99, (1500, 400))]
        levels, played = parse_levels(words)
        self.assertEqual(levels, {"pair": 7})
        self.assertEqual(played, {"pair": 13})

    def test_death_copies_entire_right_card(self):
        hand = [Card(2, "S"), Card(13, "H", enhancement="steel", edition="foil", seal="red")]
        result = transformed_hand(hand, Consumable("Death", "Tarot"), (0, 1))
        self.assertEqual(result[0], result[1])

    def test_strength_wraps_ace_to_two(self):
        result = transformed_hand(cards("AS"), Consumable("Strength", "Tarot"), (0,))
        self.assertEqual(result[0].rank, 2)

    def test_cryptid_duplicates_special_card(self):
        card = Card(13, "H", enhancement="steel", seal="red")
        self.assertEqual(transformed_hand([card], Consumable("Cryptid", "Spectral"), (0,)), [card] * 3)

    def test_observatory_can_prefer_holding_planet(self):
        state = RunState(levels={"high_card": 20}, observatory=True, planet_cards={"high_card": 1})
        self.assertEqual(decide_consumable(Consumable("Pluto", "Planet"), cards("AS"), state).action, "hold")

    def test_ankh_does_not_destroy_developed_engine(self):
        state = RunState(jokers=(Joker("Photograph"), Joker("Hanging Chad")))
        self.assertEqual(decide_consumable(Consumable("Ankh", "Spectral"), cards("AS"), state).action, "hold")

    def test_all_150_joker_names_are_identifiable(self):
        definitions = joker_definitions()
        self.assertEqual(len(definitions), 150)
        for key, definition in definitions.items():
            with self.subTest(joker=definition.name):
                self.assertEqual(identify_joker_text([definition.name])[0], key)

    def test_internal_names_differ_from_display_names(self):
        self.assertEqual(identify_joker_text(["Canio"])[0], "canio")
        self.assertEqual(identify_joker_text(["Caino"])[0], "canio")
        self.assertEqual(identify_joker_text(["Driver's License"])[0], "drivers_license")
        self.assertEqual(identify_joker_text(["Driver"])[0], "drivers_license")

    def test_chinese_joker_names_map_to_same_policy_key(self):
        self.assertEqual(identify_joker_text(["小丑"])[0], "joker")
        self.assertGreater(len(joker_aliases()), 150)

    def test_ocr_tolerance_and_unknown_rejection(self):
        self.assertEqual(identify_joker_text(["Bloodst0ne"])[0], "bloodstone")
        with self.assertRaises(RuntimeError):
            identify_joker_text(["nothing readable here"])

    def test_duplicate_jokers_keep_two_physical_slots(self):
        self.assertEqual(joker_slots(2), [842, 1198])
        state = RunState(jokers=(Joker("Joker"), Joker("Joker")))
        self.assertEqual(score_hand(cards("AS"), state).score, 144)

    def test_canio_and_drivers_license_score_by_display_key(self):
        state = RunState(jokers=(Joker("Canio", 3), Joker("Driver's License", data=(("enhanced_count", 16),))))
        self.assertEqual(score_hand(cards("AS"), state).score, 144)

    def test_space_joker_expected_score_uses_full_distribution(self):
        state = RunState(jokers=(Joker("Space Joker"),))
        self.assertEqual(score_hand(cards("AS"), state).score, 25)
        self.assertEqual(score_hand(cards("AS"), replace(state, score_mode="conservative")).score, 16)

    def test_hiker_bonus_starts_on_next_trigger(self):
        state = RunState(jokers=(Joker("Hiker"),))
        self.assertEqual(score_hand(cards("AS"), state).score, 16)
        self.assertEqual(score_hand([Card(14, "S", seal="red")], state).score, 32)

    def test_observed_stone_and_steel_joker_values(self):
        stone = parse_joker_tooltip(Joker("Stone Joker"), ["Currently +75 Chips"])
        steel = parse_joker_tooltip(Joker("Steel Joker"), ["Currently X1.6 Mult"])
        self.assertEqual(score_hand(cards("AS"), RunState(jokers=(stone,))).score, 91)
        self.assertEqual(score_hand(cards("AS"), RunState(jokers=(steel,))).score, 25.6)

    def test_selection_readback_handles_large_hand_and_forced_card(self):
        hand = [(Card(2, "H"), 1, (600 + i * 80, 725 if i != 8 else 655)) for i in range(13)]
        self.assertEqual(selection_flags(hand), {8})

    def test_joker_reordering_puts_flat_mult_before_xmult(self):
        hand = cards("2S 2H")
        state = RunState(jokers=(Joker("The Duo"), Joker("Joker")))
        order, score = optimize_joker_order(hand, state, decide_play(hand, state))
        self.assertEqual(order, (1, 0))
        self.assertEqual(score, 168)

    def test_joker_reordering_finds_blueprint_target(self):
        hand = cards("AS AH")
        state = RunState(jokers=(Joker("Blueprint"), Joker("Four Fingers"), Joker("The Duo")))
        order, score = optimize_joker_order(hand, state, decide_play(hand, state))
        self.assertEqual(state.jokers[order[order.index(0) + 1]].key, "the_duo")
        self.assertEqual(score, 256)


if __name__ == "__main__":
    unittest.main()
