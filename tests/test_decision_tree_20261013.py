"""Log-backed purchase priorities and bounded conditional suit planning."""
import unittest
from dataclasses import replace
from unittest.mock import patch

from planning import (forecast_hold, diverse_draw_candidates,
                      shop_forecast, forecast_round, _FORECAST_CACHE)
from strategy import (Card,Joker,RunState,ShopItem,ScoreEstimate,decide_shop,score_hand,
                      suit_coverage,flower_pot_cores,_discard_candidates)
from tests.test_speed_and_policy_20261005 import forecast


class ShopPriorityTests(unittest.TestCase):
    def setUp(self):
        self.state=RunState(ante=5,money=17,jokers=(Joker('Joker',4),),objective='survival')
        self.pack=ShopItem('celestial_pack','Jumbo Celestial Pack',6)

    def test_lethal_boss_search_does_not_require_a_missing_role_label(self):
        f=forecast(safe=5000)
        f=replace(f,profile=replace(f.profile,weaknesses=()))
        with patch('planning.shop_forecast',return_value=f):
            self.assertEqual(decide_shop([self.pack],self.state,reroll_cost=5).action,'reroll')

    def test_low_value_tarot_cannot_consume_the_last_search_budget(self):
        item=ShopItem('tarot','The Magician',3)
        with patch('planning.shop_forecast',return_value=forecast(safe=6500)), \
             patch('strategy._item_value',return_value=9.2):
            self.assertEqual(decide_shop([item],replace(self.state,money=11),reroll_cost=5).action,'reroll')
            self.assertNotEqual(decide_shop([item],replace(self.state,money=10),reroll_cost=5).action,'reroll')

    def test_known_scoring_rescue_keeps_priority_over_unknown_search(self):
        item=ShopItem('joker','Popcorn',5,Joker('Popcorn',20))
        with patch('planning.shop_forecast',side_effect=lambda s:forecast(safe=15000 if s.has('popcorn') else 5000)):
            d=decide_shop([item,self.pack],self.state,reroll_cost=5)
        self.assertEqual((d.action,d.item_index),('buy',0))

    def test_supported_feeding_keeps_its_existing_priority(self):
        state=replace(self.state,jokers=(Joker('Constellation',1.5),),money=16)
        with patch('planning.shop_forecast',return_value=forecast(safe=7000)):
            self.assertEqual(decide_shop([ShopItem('celestial_pack','Celestial Pack',4)],state,reroll_cost=5).action,'buy')

    def test_small_late_cash_can_search_before_coverage_becomes_lethal(self):
        state=replace(self.state,ante=6)
        with patch('planning.shop_forecast',return_value=forecast(safe=11200)):
            self.assertEqual(decide_shop([self.pack],state,reroll_cost=5).action,'reroll')
            self.assertNotEqual(decide_shop([self.pack],replace(state,money=10),reroll_cost=5).action,'reroll')
            self.assertNotEqual(decide_shop([self.pack],replace(state,shop_rerolls=2),reroll_cost=5).action,'reroll')

    def test_cash_scoring_jokers_are_not_spent_down_for_pressure_search(self):
        for name in ('Bull','Bootstraps'):
            state=replace(self.state,jokers=(Joker(name),),ante=6)
            with patch('planning.shop_forecast',return_value=forecast(safe=11200)):
                self.assertEqual(decide_shop([],state,reroll_cost=5).action,'leave')

    def test_safe_engine_does_not_reroll_just_because_it_has_a_label_gap(self):
        with patch('planning.shop_forecast',return_value=forecast(safe=25000)):
            self.assertEqual(decide_shop([],self.state,reroll_cost=5).action,'leave')

    def test_preemptive_search_is_available_at_borderline_safe_coverage_once(self):
        with patch('planning.shop_forecast',return_value=forecast(safe=12500)):
            self.assertEqual(decide_shop([],self.state,reroll_cost=5).action,'reroll')
            self.assertEqual(decide_shop([],replace(self.state,shop_rerolls=1),reroll_cost=5).action,'leave')


class ConditionalSuitTests(unittest.TestCase):
    def setUp(self):
        self.hand=[Card(r,s) for r,s in ((14,'S'),(14,'S'),(13,'S'),(12,'S'),
                                       (2,'H'),(3,'D'),(4,'C'),(6,'C'))]
        self.state=RunState(jokers=(Joker('Flower Pot'),Joker('Splash')))

    def test_rainbow_hold_is_generated_alongside_rank_and_flush_routes(self):
        cores=flower_pot_cores(self.hand,self.state)
        self.assertLessEqual(len(cores),8)
        self.assertTrue(any(suit_coverage([self.hand[i] for i in c],self.state)==4 for c in cores))
        plans=_discard_candidates(self.hand,self.state)
        self.assertTrue(any('Flower Pot' in why for why in plans.values()))
        hold=forecast_hold(self.hand,self.state,'flush')
        self.assertEqual(suit_coverage([self.hand[i] for i in hold],self.state),4)

    def test_one_wild_or_smeared_card_cannot_fill_multiple_suit_slots(self):
        self.assertEqual(suit_coverage([Card(2,'H',enhancement='wild')],self.state),1)
        smeared=replace(self.state,jokers=self.state.jokers+(Joker('Smeared Joker'),))
        self.assertEqual(suit_coverage([Card(2,'S')],smeared),1)
        self.assertEqual(suit_coverage([Card(2,'S'),Card(3,'C')],smeared),2)
        for c in flower_pot_cores([Card(2,'H',enhancement='wild')],self.state):
            self.assertEqual(len(c),len(set(c)))

    def test_hidden_debuffed_and_stone_cards_cannot_supply_suits(self):
        cards=[Card(2,'S',face_down=True),Card(3,'H',debuffed=True),Card(4,'D',enhancement='stone')]
        self.assertEqual(suit_coverage(cards,self.state),0)
        self.assertFalse(flower_pot_cores(cards,self.state))

    def test_without_splash_rainbow_high_card_is_not_scored_as_x3(self):
        state=replace(self.state,jokers=(Joker('Flower Pot'),))
        selected=[Card(14,'S'),Card(2,'H'),Card(3,'D'),Card(4,'C')]
        self.assertEqual(score_hand(selected,state).score,score_hand(selected,RunState()).score)

    def test_large_hand_pruning_preserves_a_conditional_suit_candidate(self):
        hand=self.hand+[Card(r,'S') for r in (5,7,8,9,10)]
        plans=_discard_candidates(hand,self.state)
        reduced=diverse_draw_candidates(hand,self.state,plans,limit=14)
        self.assertLessEqual(len(reduced),14)
        self.assertTrue(any('Flower Pot' in why for why in reduced.values()))

    def test_disabled_flower_pot_adds_no_extra_routes(self):
        state=replace(self.state,jokers=(Joker('Flower Pot',debuffed=True),))
        self.assertFalse(flower_pot_cores(self.hand,state))


class NeedleForecastTests(unittest.TestCase):
    def setUp(self):
        _FORECAST_CACHE.clear()

    def test_next_needle_uses_more_samples_without_changing_round_resources(self):
        state=RunState(ante=5,blind_index=1,upcoming_boss='The Needle',
                       upcoming_boss_ante=5,upcoming_boss_target=11000)
        f=forecast(safe=15000)
        with patch('planning.forecast_round',return_value=f.next_round) as calculate:
            shop_forecast(state)
        self.assertEqual(calculate.call_args.kwargs['samples'],12)
        self.assertEqual(state.hands_left,4)

    def test_upcoming_needle_is_stressed_separately_from_ordinary_next_blind(self):
        state=RunState(ante=5,blind_index=0,upcoming_boss='The Needle',
                       upcoming_boss_ante=5,upcoming_boss_target=11000)
        f=forecast(safe=15000)
        with patch('planning.forecast_round',return_value=f.next_round) as calculate:
            shop_forecast(state)
        self.assertEqual([c.kwargs['samples'] for c in calculate.call_args_list],[6,12])

    def test_single_hand_safe_score_cannot_average_away_the_failed_draw(self):
        scores=iter((100,1000,1000,1000))
        def choose(hand,state):
            score=next(scores) if len(hand)==8 else 1000
            return (0,),ScoreEstimate(score,'high_card',10,score/10,(0,))
        with patch('planning._draw_best_plan',side_effect=choose):
            f=forecast_round(RunState(base_discards_per_round=0),'The Needle',1000,samples=4)
        self.assertEqual(f.safe_score,100)
        self.assertEqual(f.clear_rate,.75)


if __name__=='__main__':
    unittest.main()
