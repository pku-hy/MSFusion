"""Check pre-fusion supervision, recursion and backward pass on CPU."""
from types import SimpleNamespace
import torch
from models.ms_fusion import Model

c = SimpleNamespace(task_name='long_term_forecast', seq_len=48, label_len=0,
    pred_len=24, down_sampling_window=2, down_sampling_layers=2,
    c_in=3, c_out=3, d_model=8, embed='timeF', freq='h',
    predict_dropout=0., dropout=0., predict_layer=1, use_norm=1,
    down_sampling_method='avg')
m = Model(c).eval()
x = torch.randn(2, 48, 3)
base, raw = m(x, None, 1)
fused, raw2 = m(x, None, 3)
assert [p.shape[1] for p in raw] == [24, 12, 6]
assert torch.allclose(base, raw[0])
assert all(torch.allclose(a, b) for a,b in zip(raw,raw2))
expected, _ = m.moe_fusion(list(raw))
assert torch.allclose(fused, expected[0])
(fused.square().mean() + sum(p.square().mean() for p in raw2)).backward()
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
assert m.moe_fusion.residual_net[0][0].weight.grad is not None
print('PASS: scale shapes, pre-fusion targets, recursion, backward')
