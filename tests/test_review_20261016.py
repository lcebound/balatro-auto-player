"""Real Stone-discovery failures and capital needed to build an Ante-8 engine."""
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from economy import reroll_investment_value
from perception import parse_card_tooltip
from recovery import ObservationError, state_from_dict
from strategy import Card, Joker, RunState, ShopItem, decide_shop
from vision import read_hand, hand_slots, stone_position
from controller import selection_flags
from tools.replay_observation_speed import load_frame
from tools.review_growth_20261015 import items_from_event
from tests.test_speed_and_policy_20261005 import forecast

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=json.loads((ROOT/'tests/fixtures/decision_review_20261016.json').read_text('utf8'))


class StoneDiscoveryTests(unittest.TestCase):
    def test_six_fatal_frames_discover_the_ninth_stone_without_identity_hints(self):
        for case in FIXTURE['failures']:
            if '8 of 9' not in case['event']['error']:continue
            image=load_frame(case['proof']['path'])
            found=read_hand(image,9,allow_uncertain=True)
            self.assertEqual(len(found),9)
            # allow_uncertain=True supplies provisional corners to collect_cards;
            # it does not certify rank/suit before the required tooltip read.
            self.assertFalse(any(c.is_stone or c.face_down for c,_,_ in found[:8]))
            self.assertTrue(found[-1][0].is_stone)
            self.assertGreaterEqual(found[-1][1],.85)
            self.assertEqual(selection_flags(found),set())

    def test_transient_empty_hand_is_still_rejected(self):
        case=next(c for c in FIXTURE['failures'] if '0 of 9' in c['event']['error'])
        with self.assertRaises(ObservationError):
            read_hand(load_frame(case['proof']['path']),9,allow_uncertain=True)

    def test_corrected_identities_relocate_all_nine_cards_on_the_same_frame(self):
        image=load_frame(FIXTURE['failures'][-1]['proof']['path'])
        identities=[Card(r,s) for r,s in ((13,'D'),(11,'C'),(10,'H'),(9,'H'),
                                          (8,'D'),(7,'H'),(6,'H'),(6,'C'))]
        identities.append(Card(0,'?',enhancement='stone',rankless=True))
        found=read_hand(image,9,identities=identities)
        self.assertEqual([c for c,_,_ in found],identities)
        self.assertEqual(selection_flags(found),set())

    def test_normal_corners_do_not_match_the_stone_discovery_threshold(self):
        case=FIXTURE['failures'][-1]
        image=load_frame(case['proof']['path'])
        for x in hand_slots(9)[:-1]:
            try:score,_=stone_position(image,x)
            except ObservationError:continue
            self.assertLess(score,.85)

    def test_known_stone_still_verifies_selected_lift(self):
        from tests.test_stone_selection_20261009 import CARDS, frame
        for name,selected in (('before',set()),('first',{0,1,7}),('toggle_off',{0,1}),('final',{0,1,7})):
            found=read_hand(frame(name),8,identities=CARDS)
            self.assertEqual(selection_flags(found),selected)

    def test_subthreshold_texture_does_not_fill_an_unreadable_slot(self):
        plain=Card(7,'S');stone=Card(0,'?',enhancement='stone',rankless=True)
        def match(image,x,y,**kw):
            templates=kw.get('templates',{})
            if templates and all(c.face_down for c in templates):return [(.1,Card(0,'?',face_down=True),1,(0,0))]
            if any(c.is_stone for c in templates):return [(.48,stone,1,(0,0))]
            return [(.36,plain,1,(0,0))]
        with patch('perception.hand_counter',return_value=(1,1)), \
             patch('vision.match_at',side_effect=match), \
             patch('vision.stone_position',return_value=(.84,(1034,725))):
            with self.assertRaises(ObservationError):read_hand(np.zeros((1191,2064,3),np.uint8),1,allow_uncertain=True)

    def test_discovered_stone_still_needs_real_tooltip_identity(self):
        stone=Card(0,'?',enhancement='stone',rankless=True)
        with self.assertRaises(ObservationError):parse_card_tooltip(stone,[],require_identity=True)
        result=parse_card_tooltip(stone,['Stone Card','+50 Chips'],require_identity=True)
        self.assertTrue(result.is_stone)
        self.assertEqual(result.bonus_chips,0)

    def test_provisional_rank_errors_are_corrected_by_required_tooltip(self):
        # The same failed layout's K/D and 8/D can initially match 6/D and
        # 5/D. Never use these preliminary corners as verified card metadata.
        king=parse_card_tooltip(Card(6,'D'),['King of Diamonds','+10 Chips'],require_identity=True)
        eight=parse_card_tooltip(Card(5,'D'),['8 of Diamonds','+8 Chips'],require_identity=True)
        self.assertEqual(king.identity,(13,'D'))
        self.assertEqual(eight.identity,(8,'D'))


class CapitalTests(unittest.TestCase):
    def decide(self,name):
        c=next(c for c in FIXTURE['cases'] if c['name']==name);e=c['event']
        return decide_shop(items_from_event(e),state_from_dict(e['state']),reroll_cost=e['reroll_cost'])

    def test_real_early_gold_tarot_keeps_the_last_five_dollars(self):
        self.assertEqual(self.decide('early_cash_reserve').action,'leave')

    def test_real_credit_card_without_supported_debt_use_is_skipped(self):
        self.assertEqual(self.decide('inactive_credit_card').action,'leave')

    def test_real_fish_shop_keeps_immediate_survival_budget(self):
        # The old forecast peeked at face-down Fish draws and incorrectly
        # classified this input as a well-funded, safe investment window.
        d=self.decide('funded_reroll_voucher')
        self.assertEqual(d.action,'reroll')

    def test_the_same_cash_can_fund_a_discount_when_draws_are_visible(self):
        c=next(c for c in FIXTURE['cases'] if c['name']=='funded_reroll_voucher');e=c['event']
        state=replace(state_from_dict(e['state']),upcoming_boss='',upcoming_boss_target=0,upcoming_boss_ante=0)
        d=decide_shop(items_from_event(e),state,reroll_cost=e['reroll_cost'])
        self.assertEqual((d.action,d.item_index),('buy',2))

    def test_known_starter_and_scoring_replacement_are_preserved(self):
        self.assertEqual((self.decide('starter_pack').action,self.decide('starter_pack').item_index),('buy',3))
        d=self.decide('known_chip_upgrade')
        self.assertEqual((d.action,d.item_index,d.sell_index),('sell_then_buy',0,1))

    def test_early_gold_feeding_a_known_vampire_remains_eligible(self):
        state=RunState(ante=1,money=5,jokers=(Joker('Vampire',1.5),),objective='survival')
        item=ShopItem('tarot','The Devil',3)
        with patch('planning.shop_forecast',return_value=forecast(safe=4000,target=600)), \
             patch('strategy._item_value',return_value=100):
            self.assertEqual(decide_shop([item],state).action,'buy')

    def test_credit_cards_actual_holographic_scoring_can_still_be_bought(self):
        item=ShopItem('joker','Credit Card',4,Joker('Credit Card',edition='holographic'))
        state=RunState(ante=2,money=30,objective='survival')
        with patch('planning.shop_forecast',side_effect=lambda s:forecast(safe=4000 if s.has('credit_card') else 1000,target=600)):
            self.assertEqual(decide_shop([item],state).action,'buy')

    def test_abstract_slot_synergy_is_a_real_credit_card_score_source(self):
        item=ShopItem('joker','Credit Card',1,Joker('Credit Card'))
        state=RunState(ante=2,money=15,jokers=(Joker('Abstract Joker'),),objective='survival')
        with patch('planning.shop_forecast',side_effect=lambda s:forecast(safe=4000 if s.has('credit_card') else 1000,target=600)):
            self.assertEqual(decide_shop([item],state).action,'buy')


class DiscountInvestmentTests(unittest.TestCase):
    def setUp(self):
        self.state=RunState(ante=4,money=40,objective='survival')
        f=forecast(safe=20000,target=8000)
        self.forecast=replace(f,profile=replace(f.profile,weaknesses=('no_scaling','no_multiplier_or_retrigger')))
        self.item=ShopItem('voucher','Reroll Surplus',10)

    def test_discount_is_a_bounded_opportunity_and_does_not_modify_cash(self):
        value=reroll_investment_value(self.item,self.state,self.forecast)
        self.assertAlmostEqual(value,19.2)
        self.assertEqual(self.state.money,40)
        self.assertEqual(self.state.owned_vouchers,())

    def test_it_does_not_spend_rescue_or_low_cash_budget(self):
        for state,f in ((replace(self.state,money=29),self.forecast),
                        (replace(self.state,ante=6),self.forecast),
                        (self.state,replace(self.forecast,next_round=replace(self.forecast.next_round,safe_score=5000)))):
            self.assertEqual(reroll_investment_value(self.item,state,f),0)

    def test_cash_jokers_and_a_complete_engine_are_protected(self):
        for name in ('Bull','Bootstraps'):
            self.assertEqual(reroll_investment_value(self.item,replace(self.state,jokers=(Joker(name),)),self.forecast),0)
        full=replace(self.forecast,profile=replace(self.forecast.profile,weaknesses=()))
        self.assertEqual(reroll_investment_value(self.item,self.state,full),0)

    def test_unrelated_vouchers_do_not_get_the_search_savings(self):
        self.assertEqual(reroll_investment_value(ShopItem('voucher','Blank',10),self.state,self.forecast),0)


class SlowGrowthTests(unittest.TestCase):
    def test_real_runner_rate_does_not_hide_slow_engine_development(self):
        from planning import shop_forecast
        c=next(c for c in FIXTURE['cases'] if c['name']=='idle_runner')
        result=shop_forecast(state_from_dict(c['event']['state']))
        self.assertIn('slow_scaling',result.profile.weaknesses)
        self.assertGreater(dict(result.next_round.growth_per_round)['runner'],0)
        self.assertLess(result.development_score,result.sustainable_score*1.12)


class ScoreSeparatorTests(unittest.TestCase):
    def test_duplicate_ocr_separator_only_changes_exact_thousands_groups(self):
        from perception import numeric,score_numeric
        for text in ('38.,852','38,.852','38..852','38,,852','#38.,852','1.,234.,567'):
            self.assertEqual(score_numeric(text),1234567 if text.startswith('1.') else 38852)
        self.assertEqual(score_numeric('1.6'),1.6)
        self.assertEqual(score_numeric('3.88e4'),38800)
        self.assertEqual(numeric('1.200'),1.2)
        self.assertIsNone(score_numeric('38..85'))

    def test_real_cashout_score_does_not_become_a_fraction(self):
        from perception import read_numbers
        case=json.loads((ROOT/'tests/fixtures/score_separator_20261016.json').read_text('utf8'))
        self.assertEqual(read_numbers(load_frame(case['proof']['path']))['round_score'],38852)

    def test_repaired_pixels_can_settle_the_missing_blind_without_predictions(self):
        from run_history import summarize_events
        # Preserve original events; validate the independently read correction
        # only in memory and never substitute policy estimates into statistics.
        fixture=json.loads((ROOT/'tests/fixtures/score_statistics_20261016.json').read_text('utf8'))
        rows=fixture['events'];index=fixture['repair_index']
        fixed=[dict(r) for r in rows]
        original=summarize_events(rows,batch_id='autonomous-20261015-232222')
        self.assertIsNone(original['games'][0]['total_score_exact'])
        fixed[index]=dict(fixed[index],counters=dict(fixed[index]['counters'],round_score=38852.))
        corrected=summarize_events(fixed,batch_id='autonomous-20261015-232222')
        self.assertTrue(corrected['games'][0]['score_complete'])
        self.assertEqual(corrected['games'][0]['total_score_exact'],fixture['expected_corrected_total'])


if __name__=='__main__':unittest.main()
