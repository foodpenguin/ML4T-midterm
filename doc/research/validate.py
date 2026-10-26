# -*- coding: utf-8 -*-
"""
深度驗證：對候選訊號做壓力測試（試圖推翻它）

檢驗項目：
  1. 極端值依賴（截尾後是否仍顯著）
  2. 子期穩健性（分 3 段是否一致）
  3. 參數鄰域（鄰近參數是否一致好 → 真 edge；隨機散布 → 過擬合）
  4. 市場 beta 汙染（多空是否真的中性）
  5. 隨機訊號對照（同框架下隨機訊號應為負）
  6. 成本敏感度（多高的成本會殺死它）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from engine import Research, DataLayer, newey_west_t, bootstrap_ci

sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

d = DataLayer()
R = Research(capital_usd=20000, n_positions=20, data=d)


def volmom(dd, lb):
    r = dd.ret(lb)
    vol = dd.close.pct_change(fill_method=None).rolling(lb).std()
    return Research.xs_z(r / (vol + 1e-9))


CAND = [("volmom_21d", "大(1-50)", 10, lambda dd: volmom(dd, 21)),
        ("ret_21d", "極小(301-500)", 5, lambda dd: Research.xs_z(dd.ret(21)))]

for name, tier, H, fn in CAND:
    sig = fn(d)
    g, adv = R.ls_returns(sig, tier, H)
    cost = R.cost.round_trip(adv)
    direction = 1 if g.mean() > 0 else -1
    net = direction * g - cost
    t0, n0 = newey_west_t(net.values)
    print(f"\n{'='*78}")
    print(f"# 候選: {name} / {tier} / H={H}d / {'動量' if direction>0 else '反轉'}")
    print(f"{'='*78}")
    print(f"  基準: 淨 {net.mean()*100:+.3f}%/期 | 年化 {net.mean()*365/H*100:+.1f}% | "
          f"t={t0:+.2f} | n={n0} | 成本 {cost*100:.3f}%")

    # 1. 極端值
    tr = net[(net > net.quantile(0.05)) & (net < net.quantile(0.95))]
    t_tr, _ = newey_west_t(tr.values)
    print(f"\n  [1] 截尾5%: 淨 {tr.mean()*100:+.3f}% | t={t_tr:+.2f} | "
          f"勝率 {(net>0).mean():.1%} → {'✅ 非極端值驅動' if abs(t_tr)>2 else '⚠️ 顯著性下降'}")

    # 2. 子期
    print(f"  [2] 子期穩健:", end=" ")
    parts = np.array_split(net.values, 3)
    for i, p in enumerate(parts):
        tt, _ = newey_west_t(p)
        print(f"P{i+1}={p.mean()*100:+.2f}%(t={tt:+.1f})", end="  ")
    consist = all(p.mean() > 0 for p in parts)
    print(f"→ {'✅ 三段皆正' if consist else '⚠️ 不一致'}")

    # 3. 參數鄰域
    print(f"  [3] 參數鄰域:", end=" ")
    nb = []
    for lb2 in [14, 21, 28, 42]:
        s2 = volmom(d, lb2) if "volmom" in name else Research.xs_z(d.ret(lb2))
        g2, adv2 = R.ls_returns(s2, tier, H)
        if len(g2) > 15:
            n2 = direction * g2 - R.cost.round_trip(adv2)
            tt, _ = newey_west_t(n2.values)
            nb.append(tt)
            print(f"lb{lb2}:t={tt:+.1f}", end="  ")
    print(f"→ {'✅ 鄰域一致' if sum(x>1 for x in nb)>=3 else '⚠️ 散布'}")

    # 4. beta 汙染
    btc = d.close["BTCUSDT"].pct_change(fill_method=None)
    btc_h = (1 + btc).rolling(H).apply(np.prod, raw=True) - 1
    bench = btc_h.reindex(net.index).dropna()
    common = net.index.intersection(bench.index)
    if len(common) > 20:
        beta = np.polyfit(bench[common], net[common], 1)[0]
        alpha_res = net[common] - beta * bench[common]
        t_a, _ = newey_west_t(alpha_res.values)
        print(f"  [4] 市場beta={beta:+.3f} | 剔除beta後 t={t_a:+.2f} → "
              f"{'✅ 非beta' if abs(t_a)>2 else '⚠️ 部分來自beta'}")

    # 5. 隨機對照
    rng = np.random.default_rng(42)
    rt = []
    for s in range(10):
        rs = pd.DataFrame(rng.standard_normal(d.close.shape), index=d.close.index,
                          columns=d.close.columns)
        gr, ar = R.ls_returns(rs, tier, H)
        if len(gr) > 15:
            nr = (gr if gr.mean() > 0 else -gr) - R.cost.round_trip(ar)
            tt, _ = newey_west_t(nr.values)
            rt.append(tt)
    print(f"  [5] 隨機對照(10次): t 平均 {np.mean(rt):+.2f} | 最大 {np.max(rt):+.2f} → "
          f"{'✅ 候選遠優於隨機' if t0 > np.max(rt) else '❌ 未勝過隨機'}")

    # 6. 成本敏感度
    print(f"  [6] 成本敏感度:", end=" ")
    for c in [0.002, 0.004, 0.006, 0.01]:
        nn = direction * g - c
        tt, _ = newey_west_t(nn.values)
        print(f"{c*100:.1f}%:{nn.mean()*365/H*100:+.0f}%(t={tt:+.1f})", end="  ")
    be = abs(g.mean())
    print(f"\n      → 損益兩平成本 {be*100:.3f}%（目前 {cost*100:.3f}%，"
          f"餘裕 {be/cost:.1f}倍）")
