import importlib.util
from pathlib import Path
import unittest

import torch

from pedestrian_behavior.models.multimodal import SpatialReadout


_SCRIPT = Path(__file__).parents[1] / "scripts/benchmark-e002a-packed-spatial-readout.py"
_SPEC = importlib.util.spec_from_file_location("packed_spatial_benchmark", _SCRIPT)
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)


class PackedSpatialReadoutTest(unittest.TestCase):
    def test_unequal_frame_lengths_match_eval_output_and_gradients(self):
        torch.manual_seed(11)
        lengths = (1, 4, 2, 7, 3)
        values = torch.randn(sum(lengths), 68)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        padded = SpatialReadout().to(device).eval()
        packed = SpatialReadout().to(device).eval()
        packed.load_state_dict(padded.state_dict())
        padded_input = values.clone().requires_grad_()
        packed_input = values.clone().requires_grad_()

        padded_output = padded(padded_input, lengths, padded.seed)
        packed_output = _BENCHMARK.packed_forward(packed, packed_input, lengths, packed.seed)
        weights = torch.randn_like(padded_output)
        (padded_output * weights).sum().backward()
        (packed_output * weights).sum().backward()

        torch.testing.assert_close(packed_output, padded_output, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(packed_input.grad, padded_input.grad, rtol=1e-4, atol=1e-5)
        for (padded_name, padded_parameter), (packed_name, packed_parameter) in zip(
                padded.named_parameters(), packed.named_parameters()):
            self.assertEqual(padded_name, packed_name)
            torch.testing.assert_close(packed_parameter.grad, padded_parameter.grad,
                                       rtol=1e-4, atol=1e-5, msg=padded_name)


if __name__ == "__main__":
    unittest.main()
