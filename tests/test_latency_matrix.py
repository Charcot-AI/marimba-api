import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def bench():
    pytest.importorskip("torch")
    pytest.importorskip("qwen_asr")
    path = Path(__file__).parents[1] / "scripts" / "latency_matrix.py"
    spec = importlib.util.spec_from_file_location("latency_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_summary(bench):
    assert bench.summary([]) == {"n": 0}
    assert bench.summary([1, 2, 3])["p50_ms"] == 2
    assert bench.summary([1, 2, 3])["n"] == 3


def test_matrix_auc_ties_and_errors(bench):
    assert bench.auroc([0, 1], [0.1, 0.9]) == 1
    assert bench.auroc([0, 1], [0.9, 0.1]) == 0
    assert bench.auroc([0, 1], [0.5, 0.5]) == 0.5
    with pytest.raises(ValueError, match="both classes"):
        bench.auroc([1, 1], [0.1, 0.9])


def test_export_taps_and_hook_cleanup(bench):
    torch = pytest.importorskip("torch")

    class Layer(torch.nn.Module):
        def forward(self, hidden):
            return (hidden + 1,)

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(18)])

        def forward(self, mel, feature_lens):
            hidden = mel.T
            for layer in self.layers:
                hidden = layer(hidden)[0]
            return type("Output", (), {"last_hidden_state": hidden})()

    tower = Tower()
    wrapper = bench.AudioExport(tower)
    output, l9, l18 = wrapper(torch.zeros((128, 800)))
    assert output.shape == (800, 128)
    assert l9.mean().item() == 9
    assert l18.mean().item() == 18
    wrapper.close()
    assert not tower.layers[8]._forward_hooks
