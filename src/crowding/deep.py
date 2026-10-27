"""Rung 4 (config/deep.yaml): hourly-sequence TCN encoder + cross-sectional attention.

Runs on CPU (smoke test) or CUDA (Colab A100). All scaling is fitted on each fold's training
rows only. Folds, purge and the holdout boundary come from crowding.splits, exactly as for W4.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn

from .splits import folds, train_mask, val_mask


# --- data ---------------------------------------------------------------------------------

@dataclass
class DeepData:
    X: np.ndarray            # (n, 168, 6) float32, NaN for missing
    mask: np.ndarray         # (n, 168) uint8
    static: np.ndarray       # (n, s) float32
    meta: pd.DataFrame       # date, symbol, target, label_end (row-aligned)

    def rows(self, sel: np.ndarray) -> np.ndarray:
        return np.nonzero(sel)[0]


def fit_scaling(d: DeepData, train_rows: np.ndarray) -> dict:
    x = d.X[train_rows]
    mu = np.nanmean(x, axis=(0, 1))
    sd = np.nanstd(x, axis=(0, 1))
    smu = np.nanmean(d.static[train_rows], axis=0)
    ssd = np.nanstd(d.static[train_rows], axis=0)
    return {"mu": mu, "sd": np.where(sd > 0, sd, 1.0), "smu": smu, "ssd": np.where(ssd > 0, ssd, 1.0)}




def date_groups(meta: pd.DataFrame, rows: np.ndarray) -> list[np.ndarray]:
    g = pd.Series(rows).groupby(meta["date"].to_numpy()[rows])
    return [v.to_numpy() for _, v in g]


def batch_index(groups: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(g) for g in groups)
    idx = np.full((len(groups), width), -1, dtype=np.int64)
    for i, g in enumerate(groups):
        idx[i, : len(g)] = g
    t = torch.tensor(idx, device=device)
    return t.clamp(min=0), t < 0          # gather index, padding mask (True = pad)


# --- model --------------------------------------------------------------------------------

class TCN(nn.Module):
    def __init__(self, in_ch: int, width: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, width, 1)
        self.blocks = nn.ModuleList()
        self.pads = [2 * d for d in (1, 4, 16)]
        for d in (1, 4, 16):
            self.blocks.append(nn.Sequential(nn.Conv1d(width, width, 3, dilation=d), nn.GELU(), nn.Dropout(dropout)))
        self.out = nn.Linear(2 * width, width)

    def forward(self, x):                                  # x: (n, 168, c)
        h = self.inp(x.transpose(1, 2))
        for pad, blk in zip(self.pads, self.blocks):
            h = h + blk(nn.functional.pad(h, (pad, 0)))    # causal: pad on the left only
        return self.out(torch.cat([h.mean(dim=2), h[:, :, -1]], dim=1))


class CrowdingNet(nn.Module):
    """``use_sequence=False`` is the registered ablation: no hourly branch, static inputs only."""

    def __init__(self, in_ch: int, n_static: int, width: int, dropout: float,
                 xs_layers: int = 2, use_sequence: bool = True):
        super().__init__()
        self.use_sequence = use_sequence
        self.enc = TCN(in_ch, width, dropout) if use_sequence else None
        self.token = nn.Sequential(nn.Linear((width if use_sequence else 0) + n_static, width),
                                   nn.GELU(), nn.Dropout(dropout))
        heads = 4 if width % 4 == 0 else 1
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout, batch_first=True)
        self.xs = nn.TransformerEncoder(layer, xs_layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, 1)

    def forward(self, seq, static, pad):                   # seq: (b, k, 168, c); static: (b, k, s)
        if self.use_sequence:
            b, k = seq.shape[:2]
            e = self.enc(seq.reshape(b * k, *seq.shape[2:])).reshape(b, k, -1)
            x = torch.cat([e, static], dim=2)
        else:
            x = static
        z = self.xs(self.token(x), src_key_padding_mask=pad)
        return self.head(z).squeeze(-1)


def pearson_loss(pred, target, pad):
    """1 - within-date Pearson correlation, averaged over the dates in the batch."""
    valid = (~pad).float()
    n = valid.sum(1).clamp(min=2)
    pm = (pred * valid).sum(1) / n
    tm = (target * valid).sum(1) / n
    pc, tc = (pred - pm[:, None]) * valid, (target - tm[:, None]) * valid
    corr = (pc * tc).sum(1) / (pc.pow(2).sum(1).sqrt() * tc.pow(2).sum(1).sqrt() + 1e-8)
    return 1 - corr.mean()


# --- training -----------------------------------------------------------------------------

@dataclass
class Config:
    width: int = 64
    learning_rate: float = 1e-3
    dropout: float = 0.1
    weight_decay: float = 1e-4
    batch_dates: int = 8
    max_epochs: int = 30
    patience: int = 5
    xs_layers: int = 2
    use_sequence: bool = True


def config_from_row(row) -> "Config":
    """Rebuild a Config from a tuning.csv row (bools arrive as strings or numpy bools)."""
    out = {}
    for k, default in vars(Config()).items():
        if k not in row or pd.isna(row[k]):
            continue
        v = row[k]
        if isinstance(default, bool):
            out[k] = str(v).strip().lower() in ("true", "1")
        else:
            out[k] = type(default)(v)
    return Config(**out)


class Trainer:
    """Raw inputs live on the device once; ``prepare`` only refits the per-channel scaling,
    which is applied to each gathered batch."""

    def __init__(self, d: DeepData, device):
        self.d, self.device = d, device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.X = torch.tensor(d.X, dtype=dtype, device=device)            # NaN kept for missing
        self.M = torch.tensor(d.mask, dtype=dtype, device=device)
        self.S = torch.tensor(d.static, dtype=dtype, device=device)
        self.target = torch.tensor(d.meta["target"].to_numpy(np.float32), device=device)
        self.in_ch = d.X.shape[-1] + 1                                     # channels + observed mask
        self.scaled_on: np.ndarray | None = None

    def prepare(self, train_rows):
        """Fit scaling on ``train_rows`` only (they must be the rows the next fit trains on)."""
        sc = fit_scaling(self.d, train_rows)
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        self.mu, self.sd, self.smu, self.ssd = t(sc["mu"]), t(sc["sd"]), t(sc["smu"]), t(sc["ssd"])
        self.scaled_on = np.asarray(train_rows)

    def _inputs(self, gi):
        x = torch.nan_to_num((self.X[gi].float() - self.mu) / self.sd, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.cat([x, self.M[gi].float().unsqueeze(-1)], dim=-1)
        s = torch.nan_to_num((self.S[gi].float() - self.smu) / self.ssd, nan=0.0)
        return x, s

    def _forward(self, model, groups):
        gi, pad = batch_index(groups, self.device)
        x, s = self._inputs(gi)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = model(x, s, pad)
        return out.float(), gi, pad

    def predict(self, model, rows: np.ndarray) -> np.ndarray:
        model.eval()
        out = np.full(len(self.d.meta), np.nan, dtype=np.float32)
        groups = date_groups(self.d.meta, rows)
        with torch.no_grad():
            for i in range(0, len(groups), 32):
                pred, gi, pad = self._forward(model, groups[i:i + 32])
                keep = ~pad
                out[gi[keep].cpu().numpy()] = pred[keep].cpu().numpy()
        return out[rows]

    def mean_ic(self, model, rows: np.ndarray) -> float:
        p = self.predict(model, rows)
        m = self.d.meta.iloc[rows]
        ic = pd.DataFrame({"date": m["date"].to_numpy(), "p": p, "y": m["target"].to_numpy()}).groupby("date").apply(
            lambda g: stats.spearmanr(g["p"], g["y"]).statistic if len(g) >= 20 else np.nan, include_groups=False)
        return float(ic.mean())

    def fit(self, cfg: Config, train_rows, seed: int, epochs: int | None = None, val_rows=None):
        """Train; with ``val_rows`` early-stop on mean daily IC and return the best epoch count."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = CrowdingNet(self.in_ch, self.S.shape[-1], cfg.width, cfg.dropout,
                            cfg.xs_layers, cfg.use_sequence).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        groups = date_groups(self.d.meta, train_rows)
        rng = np.random.default_rng(seed)
        best, best_ic, best_state, bad = 0, -np.inf, None, 0
        for epoch in range(1, (epochs or cfg.max_epochs) + 1):
            model.train()
            order = rng.permutation(len(groups))
            for i in range(0, len(order), cfg.batch_dates):
                batch = [groups[j] for j in order[i:i + cfg.batch_dates]]
                pred, gi, pad = self._forward(model, batch)
                loss = pearson_loss(pred, self.target[gi], pad)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if val_rows is not None:
                ic = self.mean_ic(model, val_rows)
                if ic > best_ic:
                    best, best_ic, bad = epoch, ic, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        break
        if val_rows is not None and best_state is not None:
            model.load_state_dict(best_state)
        return model, best, best_ic


def conformal_q(residuals: np.ndarray, coverage: float) -> float:
    r = np.sort(np.abs(residuals[np.isfinite(residuals)]))
    k = min(len(r) - 1, int(np.ceil((len(r) + 1) * coverage)) - 1)
    return float(r[k])


def tuning_rows(d: DeepData) -> tuple[np.ndarray, np.ndarray]:
    m = d.meta
    cut, end = pd.Timestamp("2022-12-01"), pd.Timestamp("2023-03-01")
    tr = np.nonzero((m["label_end"] <= cut).to_numpy())[0]
    va = np.nonzero(((m["date"] >= cut) & (m["label_end"] <= end)).to_numpy())[0]
    assert m["label_end"].iloc[va].max() <= folds()[0].val_start
    return tr, va


def tune(tr: Trainer, grid: list[Config], log=print) -> tuple[Config, pd.DataFrame]:
    rows_tr, rows_va = tuning_rows(tr.d)
    tr.prepare(rows_tr)
    out = []
    for cfg in grid:
        t0 = time.time()
        _, ep, ic = tr.fit(cfg, rows_tr, seed=0, val_rows=rows_va)
        out.append({**vars(cfg), "best_epoch": ep, "inner_ic": ic, "seconds": time.time() - t0})
        log(f"tune {vars(cfg)} -> IC {ic:.4f} at epoch {ep} ({time.time() - t0:.0f}s)")
    t = pd.DataFrame(out)
    return config_from_row(t.loc[t["inner_ic"].idxmax()]), t


def tune_optuna(tr: "Trainer", n_trials: int, seed: int = 42, log=print,
                max_epochs: int = 30) -> tuple[Config, pd.DataFrame]:
    """Registered Optuna TPE search (config/deep_v2.yaml) on the pre-validation tuning window."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows_tr, rows_va = tuning_rows(tr.d)
    tr.prepare(rows_tr)
    records = []

    def objective(trial):
        cfg = Config(width=trial.suggest_categorical("width", [16, 32, 64, 96]),
                     learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
                     dropout=trial.suggest_float("dropout", 0.0, 0.5),
                     weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
                     xs_layers=trial.suggest_int("xs_layers", 1, 3),
                     batch_dates=trial.suggest_categorical("batch_dates", [4, 8, 16]),
                     max_epochs=max_epochs, patience=min(5, max_epochs))
        t0 = time.time()
        _, ep, ic = tr.fit(cfg, rows_tr, seed=0, val_rows=rows_va)
        records.append({**vars(cfg), "best_epoch": ep, "inner_ic": ic, "seconds": time.time() - t0})
        log(f"optuna trial {trial.number}: {vars(cfg)} -> IC {ic:.4f} at epoch {ep}")
        return ic if np.isfinite(ic) else -1.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    t = pd.DataFrame(records)
    return config_from_row(t.loc[t["inner_ic"].idxmax()]), t


def run_folds(tr: Trainer, cfg: Config, seeds: int = 5, only=None, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = tr.d.meta
    outs, info = [], []
    for f in folds():
        if only is not None and f.fold not in only:
            continue
        t0 = time.time()
        d, e = m["date"], m["label_end"]
        r_tr = np.nonzero(train_mask(d, e, f).to_numpy())[0]
        r_itr = np.nonzero(train_mask(d, e, f, inner=True).to_numpy())[0]
        r_iva = np.nonzero(val_mask(d, e, f, inner=True).to_numpy())[0]
        r_va = np.nonzero(val_mask(d, e, f).to_numpy())[0]
        # Inner model: scaling fitted on the inner training rows only, so its early stopping and
        # conformal calibration on the inner window never see that window's statistics.
        tr.prepare(r_itr)
        inner, n_ep, inner_ic = tr.fit(cfg, r_itr, seed=0, val_rows=r_iva)
        n_ep = max(n_ep, 1)
        res = m["target"].to_numpy()[r_iva] - tr.predict(inner, r_iva)
        q80, q90 = conformal_q(res, 0.8), conformal_q(res, 0.9)
        pred_inner = tr.predict(inner, r_va)
        # Refits: scaling refitted on the full training window they train on.
        # Registered design: the epoch count comes from the single seed-0 inner run (see report).
        tr.prepare(r_tr)
        preds = [tr.predict(tr.fit(cfg, r_tr, seed=s, epochs=n_ep)[0], r_va) for s in range(seeds)]
        o = m.iloc[r_va][["date", "symbol"]].copy()
        o["fold"], o["pred"], o["pred_inner"], o["q80"], o["q90"] = f.fold, np.mean(preds, axis=0), pred_inner, q80, q90
        outs.append(o)
        info.append({"fold": f.fold, "epochs": n_ep, "inner_ic": inner_ic, "seconds": time.time() - t0,
                     "seed_pred_corr": float(np.corrcoef(preds)[np.triu_indices(seeds, 1)].mean()) if seeds > 1 else np.nan})
        log(f"fold {f.fold}: epochs {n_ep}, inner IC {inner_ic:.4f}, {time.time() - t0:.0f}s")
    return pd.concat(outs, ignore_index=True), pd.DataFrame(info)
