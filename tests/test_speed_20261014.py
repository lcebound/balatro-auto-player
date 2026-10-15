"""Speedups must retain state semantics, pixels, and action verification."""
import pickle
import tempfile
import unittest
from dataclasses import asdict,replace
from pathlib import Path
from unittest.mock import Mock,patch

import cv2
import numpy as np

from bot import write_observation_png,Session
from controller import collect_cards,select_verified
from strategy import (Card,Joker,RunState,classify,_classify_uncached,_classify_cached,
                      _before_scoring,joker_name,score_hand,_boss)
from tests.test_runtime_recovery import FakeSession


class DerivedStateTests(unittest.TestCase):
    def test_normalized_names_are_bounded_cached_and_still_unicode_aware(self):
        joker_name.cache_clear()
        self.assertEqual(joker_name('Cavéndish'),'cavendish')
        for _ in range(20):self.assertEqual(joker_name('Ride the Bus'),'ride_the_bus')
        self.assertGreaterEqual(joker_name.cache_info().hits,19)
        self.assertLessEqual(joker_name.cache_info().maxsize,2048)

    def test_duplicate_data_keys_keep_the_original_first_value(self):
        j=Joker('Joker',data=(('foo',0),('foo',9)))
        self.assertEqual(j.get('foo',5),0)
        self.assertEqual(j.get('missing',5),5)
        newer=replace(j,data=(('foo',7),))
        self.assertEqual(newer.get('foo'),7)
        self.assertEqual(j.get('foo'),0)

    def test_copiers_physical_counts_and_debuffs_remain_distinct(self):
        state=RunState(jokers=(Joker('Blueprint'),Joker('Green Joker',3),Joker('Green Joker',8,debuffed=True)))
        self.assertEqual(state.joker_count('green_joker'),1)
        self.assertEqual(state.effect_count('green_joker'),2)
        self.assertEqual(state.joker_count('blueprint'),1)
        self.assertEqual(state.effect_count('blueprint'),0)
        self.assertFalse(state.has('mime'))
        reordered=replace(state,jokers=tuple(reversed(state.jokers)))
        self.assertEqual(reordered.effect_count('green_joker'),1)

    def test_hidden_order_does_not_invent_copier_targets(self):
        state=RunState(jokers=(Joker('Blueprint'),Joker('Card Sharp')),joker_order_unknown=True)
        self.assertEqual(state.effect_count('card_sharp'),1)
        visible=replace(state,joker_order_unknown=False)
        self.assertEqual(visible.effect_count('card_sharp'),2)

    def test_derived_caches_do_not_leak_into_json_or_replace(self):
        state=RunState(jokers=(Joker('Green Joker',3),),levels={'pair':1})
        original=asdict(state)
        state.has('green_joker');state.effect_count('green_joker');state.jokers[0].get('foo')
        self.assertEqual(asdict(state),original)
        new=replace(state,jokers=(Joker('Joker'),))
        self.assertFalse(new.has('green_joker'))
        self.assertEqual(new.effect_count('green_joker'),0)
        state.levels['pair']=3
        self.assertEqual(state.level('pair'),3)
        restored=pickle.loads(pickle.dumps(state))
        self.assertTrue(restored.has('green_joker'))
        self.assertEqual(restored.level('pair'),3)

    def test_nonconverting_roster_keeps_original_objects(self):
        state=RunState(jokers=(Joker('Green Joker',4),Joker('Card Sharp')))
        pattern=classify([Card(2,'S')],state)
        after,s=_before_scoring(pattern,state)
        self.assertIs(s,state)
        self.assertIs(after,pattern)

    def test_vampire_midas_and_wee_still_project_real_changes(self):
        state=RunState(jokers=(Joker('Midas Mask'),Joker('Vampire',1)))
        pattern=classify([Card(13,'S')],state)
        after,s=_before_scoring(pattern,state)
        self.assertEqual(after.played[0].enhancement,'')
        self.assertAlmostEqual(s.jokers[1].value,1.1)
        wee=RunState(jokers=(Joker('Wee Joker',2),))
        _,grown=_before_scoring(classify([Card(2,'S')],wee),wee)
        self.assertEqual(grown.jokers[0].value,10)

    def test_boss_changes_disable_and_chicot_are_not_stale(self):
        state=RunState(boss='The Needle')
        self.assertEqual(_boss(state),'needle')
        self.assertEqual(_boss(replace(state,boss='The Hook')),'hook')
        self.assertEqual(_boss(replace(state,boss_disabled=True)),'')
        self.assertEqual(_boss(replace(state,jokers=(Joker('Chicot'),))),'')

    def test_classification_cache_matches_uncached_across_all_rule_combinations(self):
        hands=([Card(r,s) for r,s in ((2,'S'),(4,'H'),(6,'C'),(8,'D'),(10,'S'))],
               [Card(2,'S'),Card(3,'S'),Card(4,'S'),Card(5,'S')],
               [Card(14,'H'),Card(14,'D'),Card(0,'?',enhancement='stone')],
               [Card(2,'H',enhancement='wild'),Card(3,'S'),Card(4,'C'),Card(5,'S'),Card(6,'C')])
        keys=('four_fingers','shortcut','smeared_joker','splash')
        for flags in range(16):
            state=RunState(jokers=tuple(Joker(k) for i,k in enumerate(keys) if flags&(1<<i)))
            for hand in hands:self.assertEqual(classify(hand,state),_classify_uncached(hand,state))

    def test_score_state_changes_reuse_only_classification_and_still_change_scores(self):
        _classify_cached.cache_clear()
        hand=[Card(14,'S'),Card(14,'H')]
        state=RunState(jokers=(Joker('Bull'),),money=5)
        a=score_hand(hand,state).score
        b=score_hand(hand,replace(state,money=25,levels={'pair':2})).score
        self.assertGreater(b,a)
        self.assertGreater(_classify_cached.cache_info().hits,0)
        self.assertEqual(_classify_cached.cache_info().maxsize,8192)


class ObservationTests(unittest.TestCase):
    def test_fast_png_round_trip_preserves_all_pixels_and_unicode_path(self):
        raw=np.random.default_rng(7).integers(0,256,(81,113,4),np.uint8)[:,:,:3]
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'牌面.png'
            write_observation_png(p,raw)
            decoded=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),1)
        np.testing.assert_array_equal(raw,decoded)

    def test_png_encoding_failure_is_not_an_observation(self):
        with patch('bot.cv2.imencode',return_value=(False,None)):
            with self.assertRaises(ValueError):write_observation_png(Path('unused.png'),np.zeros((2,2,3),np.uint8))

    def collect(self,first,parsed,second=None):
        with tempfile.TemporaryDirectory() as directory:
            session=FakeSession(directory);session.frame=np.zeros((1191,2064,3),np.uint8)
            found=[(first[0],first[1],(700,725))]
            reads=[found] if second is None else [found,[(second,.99,(701,725))]]
            with patch('plain_hand.read_plain_hand',return_value=[None]), \
                 patch('controller.read_hand',side_effect=reads) as reader, \
                 patch('controller.read_text',return_value=[]), \
                 patch('controller.parse_card_tooltip',return_value=parsed):
                result=collect_cards(session,session.frame,1,state=RunState())
            return result,reader.call_count,session.events[-1]

    def test_verified_same_corner_does_not_get_scanned_again_for_modifiers(self):
        c=Card(8,'S');changed=replace(c,bonus_chips=10,enhancement='bonus')
        result,calls,event=self.collect((c,.9),changed)
        self.assertEqual(calls,1)
        self.assertEqual(result[0][0],changed)
        self.assertEqual(result[0][2],(700,725))
        self.assertTrue(event['positions_reused'])

    def test_changed_identity_weak_corner_and_stone_require_relocation(self):
        for first,confidence,parsed in ((Card(8,'S'),.9,Card(9,'S')),
                (Card(8,'S'),.45,Card(8,'S')),
                (Card(8,'S'),.9,Card(0,'?',enhancement='stone'))):
            _,calls,event=self.collect((first,confidence),parsed,parsed)
            self.assertEqual(calls,2)
            self.assertFalse(event['positions_reused'])


class SelectionTests(unittest.TestCase):
    def test_bulk_selection_still_verifies_every_selected_card(self):
        found=[(Card(r,'S'),1.,(700+i*100,725)) for i,r in enumerate((2,3,4))]
        with tempfile.TemporaryDirectory() as directory:
            session=FakeSession(directory);session.hover=Mock()
            with patch('controller.selection_flags',side_effect=[set(),{0,1,2}]), \
                 patch('controller.read_hand',return_value=found):
                select_verified(session,found,[0,1,2],'play')
            self.assertEqual(len(session.taps),3)
            self.assertEqual(session.hover.call_count,2)
            self.assertEqual(session.events[-1]['event'],'selection_verified')

    def test_failed_bulk_attempt_falls_back_to_per_card_tooltip_clear(self):
        found=[(Card(r,'S'),1.,(700+i*100,725)) for i,r in enumerate((2,3))]
        with tempfile.TemporaryDirectory() as directory:
            session=FakeSession(directory);session.hover=Mock()
            with patch('controller.selection_flags',side_effect=[set(),set(),{0,1}]), \
                 patch('controller.read_hand',return_value=found):
                select_verified(session,found,[0,1],'play')
            self.assertEqual(len(session.taps),4)
            self.assertEqual(session.hover.call_count,5)
            self.assertEqual(session.events[-1]['attempts'],2)


if __name__=='__main__':unittest.main()
