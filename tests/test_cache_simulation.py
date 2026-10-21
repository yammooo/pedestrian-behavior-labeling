"""Hand-calculated byte eviction, cross-epoch reuse and no-cache controls."""

import unittest

import numpy as np
import torch

from pedestrian_behavior.inspection.cache_simulation import ByteLRU
from pedestrian_behavior.experiments.e002a_cache_probe import point_slots, replay_orders
from pedestrian_behavior.data.loading import track_batches


class CacheSimulationTest(unittest.TestCase):
    def test_byte_accounting_and_lru_refresh(self):
        cache = ByteLRU(10)
        for key,size in (("a",6),("b",4),("a",6),("c",4)):
            cache.access(key,size,1)
        self.assertEqual(list(cache.entries),["a","c"])
        self.assertEqual(cache.used,10)
        self.assertEqual(cache.counts["hits"],1)
        self.assertEqual(cache.counts["write_bytes"],14)
        self.assertEqual(cache.counts["read_bytes"],6)
        self.assertEqual(cache.counts["evicted_bytes"],4)
        cache.reset_counts();cache.access("a",6,2);cache.access("a",6,2)
        self.assertEqual(cache.counts["prior_epoch_hits"],1)

    def test_cyclic_scan_thrashes_but_unbounded_only_misses_once(self):
        for capacity,expected_hits in ((4,0),(None,6)):
            cache=ByteLRU(capacity)
            for epoch in (1,2):
                for key in range(6):cache.access(key,1,epoch)
            self.assertEqual(cache.counts["hits"],expected_hits)
            self.assertEqual(cache.counts["compulsory_misses"],6)
            self.assertEqual(cache.counts["capacity_misses"],6 if capacity==4 else 0)

    def test_oversize_bypass_keeps_existing_entries(self):
        cache=ByteLRU(5);cache.access("small",4,1);cache.access("large",6,1)
        self.assertEqual(list(cache.entries),["small"])
        self.assertEqual(cache.counts["oversize_bypasses"],1)
        self.assertEqual(cache.counts["write_bytes"],4)
        zero=ByteLRU(0);zero.access("a",1,1);zero.access("a",1,1)
        self.assertEqual(zero.used,0);self.assertEqual(zero.counts["hits"],0)
        self.assertEqual(zero.counts["write_bytes"],0)
        with self.assertRaises(ValueError):ByteLRU(-1)
        with self.assertRaises(ValueError):cache.access("bad",0,1)

    def test_only_input_availability_selects_requests(self):
        m={"source_frames":[1,2,3,None],"anchor_valid":True,
           "availability":[{"native_3d":True},{"native_3d":False},{"native_3d":True},{}]}
        self.assertEqual(point_slots(m,np.ones(4,bool),[True,True,False,True]),["1"])
        m["anchor_valid"]=False
        self.assertEqual(point_slots(m,np.ones(4,bool),np.ones(4,bool)),[])

    def test_reference_shuffle_matches_existing_persistent_loader(self):
        samples=[{"inputs":torch.zeros(1,1),"targets":torch.zeros(1,dtype=torch.long),
                  "gt_valid":torch.ones(1,dtype=torch.bool),"locator":str(i),"archive":"unused"} for i in range(70)]
        loader=track_batches(samples,training=True,seed=0)
        actual=[[int(i) for batch in loader for i in batch["locators"]] for _ in range(2)]
        self.assertEqual(actual,replay_orders(70,2))


if __name__ == "__main__":
    unittest.main()
