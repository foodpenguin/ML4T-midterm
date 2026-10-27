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


def test_static_only_ablation_ignores_the_sequence():
    torch.manual_seed(0)
    net = CrowdingNet(in_ch=7, n_static=3, width=8, dropout=0.0, xs_layers=1, use_sequence=False).eval()
    seq, static = torch.randn(1, 6, 168, 7), torch.randn(1, 6, 3)
    pad = torch.zeros(1, 6, dtype=torch.bool)
    assert torch.allclose(net(seq, static, pad), net(seq * 0 + 5.0, static, pad))
    assert net.enc is None


def test_config_round_trips_through_csv_strings():
    import pandas as pd
    from crowding.deep import Config, config_from_row
    row = pd.Series({"width": "64", "learning_rate": "0.0005", "dropout": 0.2, "weight_decay": 1e-3,
                     "batch_dates": 16.0, "max_epochs": 30, "patience": 5, "xs_layers": 3, "use_sequence": "False"})
    c = config_from_row(row)
    assert c == Config(width=64, learning_rate=0.0005, dropout=0.2, weight_decay=1e-3, batch_dates=16,
                       max_epochs=30, patience=5, xs_layers=3, use_sequence=False)


def test_run_folds_trains_scales_and_validates_on_the_right_rows():
    import pandas as pd
    from crowding import deep
    from crowding.config import DAY, setup
    from crowding.splits import folds

    dates = pd.date_range(setup()["evaluation"]["dev_start"], folds()[-1].val_end)
    meta = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(3)]],
                                      names=["date", "symbol"]).to_frame(index=False)
    meta["target"] = 0.0
    meta["label_end"] = meta["date"] + 7 * DAY
    n = len(meta)
    data = deep.DeepData(np.zeros((n, 168, 6), np.float32), np.ones((n, 168), np.uint8),
                         np.zeros((n, 2), np.float32), meta)

    class Recorder(deep.Trainer):
        def __init__(self, d):
            self.d, self.calls, self.scaled_on = d, [], None

        def prepare(self, rows):
            self.scaled_on = np.asarray(rows)

        def fit(self, cfg, train_rows, seed, epochs=None, val_rows=None):
            self.calls.append((np.asarray(train_rows), self.scaled_on, val_rows))
            return object(), 3, 0.1

        def predict(self, model, rows):
            return np.zeros(len(rows))

    tr = Recorder(data)
    picked = {0, 15, 31}
    deep.run_folds(tr, deep.Config(), seeds=2, only=picked, log=lambda *_: None)
    by_fold = [f for f in folds() if f.fold in picked]
    assert len(tr.calls) == 3 * len(by_fold)                 # inner + 2 seeds per fold
    for i, f in enumerate(by_fold):
        inner, *refits = tr.calls[3 * i: 3 * i + 3]
        train, scaled, val = inner
        assert np.array_equal(train, scaled), "inner scaling must use only the inner training rows"
        assert meta["label_end"].iloc[train].max() <= meta["date"].iloc[val].min()
        assert meta["label_end"].iloc[val].max() <= f.val_start
        for train, scaled, val in refits:
            assert val is None and np.array_equal(train, scaled)
            assert meta["label_end"].iloc[train].max() <= f.val_start
            assert meta["date"].iloc[train].max() < f.val_start
