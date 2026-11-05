import pickle
import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class HybridSchedulerTests(unittest.TestCase):
    def make_scheduler(self, slots=1, blocks=8):
        return Scheduler(SimpleNamespace(max_num_seqs=slots, max_num_batched_tokens=128,
            eos=-1, kvcache_block_size=256, num_kvcache_blocks=blocks,
            enable_prefix_cache=False, is_hybrid=True))

    def test_chunk_continuation_and_slot_recycling(self):
        scheduler = self.make_scheduler()
        first = Sequence(list(range(300)), SamplingParams(max_tokens=2))
        second = Sequence(list(range(300)), SamplingParams(max_tokens=1))
        scheduler.add(first)
        scheduler.add(second)
        slot = None
        for cached in (0, 128, 256):
            seqs, prefill = scheduler.schedule()
            self.assertTrue(prefill)
            self.assertEqual(seqs, [first])
            self.assertEqual(first.num_cached_tokens, cached)
            slot = first.state_slot if slot is None else slot
            self.assertEqual(first.state_slot, slot)
            scheduler.postprocess(seqs, [7], prefill)
        seqs, prefill = scheduler.schedule()
        self.assertFalse(prefill)
        scheduler.postprocess(seqs, [8], prefill)
        self.assertTrue(first.is_finished)
        self.assertEqual(first.state_slot, -1)
        seqs, prefill = scheduler.schedule()
        self.assertEqual(seqs, [second])
        self.assertEqual(second.num_cached_tokens, 0)  # identical prompt must be recomputed
        self.assertEqual(second.state_slot, slot)

    def test_preemption_resets_state_and_tp_serialization(self):
        scheduler = self.make_scheduler()
        seq = Sequence([1] * 300)
        scheduler.add(seq)
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [7], prefill)
        restored = pickle.loads(pickle.dumps(seq))
        self.assertEqual(restored.state_slot, seq.state_slot)
        self.assertEqual(restored.num_cached_tokens, 128)
        scheduler.waiting.remove(seq)
        scheduler.preempt(seq)
        self.assertEqual(seq.state_slot, -1)
        self.assertEqual(seq.num_cached_tokens, 0)
        scheduler.schedule()
        self.assertEqual(seq.state_slot, 0)
        self.assertEqual(seq.num_cached_tokens, 0)


if __name__ == "__main__":
    unittest.main()
