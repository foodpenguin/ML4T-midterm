import numpy as np
import torch

from crowding.deep import CrowdingNet, pearson_loss


def test_pearson_loss_ignores_padding_and_matches_correlation():
    rng = np.random.default_rng(0)
    p, t = rng.normal(size=(2, 10)), rng.normal(size=(2, 10))
    pad = np.zeros((2, 10), dtype=bool)
    pad[1, 7:] = True
    t_pad = t.copy()
    t_pad[1, 7:] = 99.0                                # junk in padded slots must not matter
    loss = pearson_loss(torch.tensor(p), torch.tensor(t_pad), torch.tensor(pad)).item()
    c0 = np.corrcoef(p[0], t[0])[0, 1]
    c1 = np.corrcoef(p[1, :7], t[1, :7])[0, 1]
    assert np.isclose(loss, 1 - (c0 + c1) / 2, atol=1e-6)


def test_net_scores_each_name_and_respects_padding():
    torch.manual_seed(0)
    net = CrowdingNet(in_ch=7, n_static=3, width=8, dropout=0.0).eval()
    seq, static = torch.randn(2, 5, 168, 7), torch.randn(2, 5, 3)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    pad[0, 4] = True
    out = net(seq, static, pad)
    assert out.shape == (2, 5)
    seq2 = seq.clone()
    seq2[0, 4] += 100.0                               # change only the padded name
    assert torch.allclose(net(seq2, static, pad)[0, :4], out[0, :4], atol=1e-5)
