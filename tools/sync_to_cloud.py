# -*- coding: utf-8 -*-
"""
每日雲端同步 —— 把研究資料增量備份到 Google Drive

為何自建而不用 rclone：
  Google Drive 虛擬磁碟機(G:)寫入後檔案大小會變動(實測 2,831,279 → 2,831,360)，
  rclone 的完整性檢查會判定 "corrupted on transfer" 而拒絕寫入。
  原生檔案複製正常，故自行以 (size, mtime) 比對做增量同步，少一個依賴也更可控。

同步範圍：只收「有價值且不可輕易重建」者
  * Data/ALPHA     ← 最重要：API 僅 21 天滾動視窗，無法回溯，遺失即永久遺失
  * Data/UNIVERSE  ← 可重抓但費時（665 幣 × 7 年）
  * Data/LP        ← DeFiLlama LP 歷史
  * Data/METRICS/hourly ← 倉位歷史回補（多數幣 2021-12 起，按月分割 Parquet；只有當月檔會變動）
  * Data/DATA_QUALITY.md ← 資料品質報告
  * research/ tools/ 報告  ← 程式與結論

用法：
  python sync_to_cloud.py              # 增量同步
  python sync_to_cloud.py --dry-run    # 只顯示會做什麼
  python sync_to_cloud.py --dest "D:/backup"   # 指定目標
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SUBDIR = "BNB_quant_backup"

INCLUDE = [
    "Data/ALPHA", "Data/UNIVERSE", "Data/LP",
    # 倉位歷史回補：只同步按月分割的成品與稽核資料；
    # staging/ 約 1.4 萬個中間檔可由 backfill_metrics.py 重建，不同步
    "Data/METRICS/hourly", "Data/METRICS/manifest.parquet", "Data/METRICS/backfill_plan.csv",
    "Data/DATA_QUALITY.md",
    "research", "tools",
    "RESEARCH_FINDINGS.md", "ecosystem.config.cjs",
]
EXCLUDE = ["__pycache__", ".pyc", "hourly_px.csv", "rclone", ".venv", ".partial"]
MTIME_TOL = 2.0          # 秒；雲端檔案系統時間戳可能有微小差異


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def excluded(p: Path) -> bool:
    s = str(p)
    return any(x in s for x in EXCLUDE)


def iter_pairs(dest_root: Path):
    """產生 (來源檔, 目標檔) 配對。"""
    for item in INCLUDE:
        src = ROOT / item
        if src.is_file():
            if not excluded(src):
                yield src, dest_root / item
        elif src.is_dir():
            for f in src.rglob("*"):
                if f.is_file() and not excluded(f):
                    yield f, dest_root / item / f.relative_to(src)


def needs_copy(src: Path, dst: Path) -> bool:
    """以 (size, mtime) 判斷是否需要複製。"""
    if not dst.exists():
        return True
    try:
        ss, ds = src.stat(), dst.stat()
    except OSError:
        return True
    # 雲端碟會動到 size，故 size 不同時再比 mtime，避免每次都全量重傳
    if abs(ss.st_mtime - ds.st_mtime) > MTIME_TOL:
        return True
    return False


def find_dest():
    """找本機雲端同步資料夾。Google Drive 優先（含中文路徑）。"""
    home = Path(os.path.expanduser("~"))
    for name, p in [
        ("Google Drive", Path("G:/我的雲端硬碟")),
        ("Google Drive", Path("G:/My Drive")),
        ("Google Drive", home / "My Drive"),
        ("OneDrive", Path(os.environ.get("OneDrive", str(home / "OneDrive")))),
    ]:
        if p.exists():
            return name, p
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=None, help="目標根目錄（預設自動偵測雲端資料夾）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.dest:
        name, base = "指定路徑", Path(a.dest)
    else:
        name, base = find_dest()
    if not base:
        print("❌ 找不到雲端同步資料夾（Google Drive / OneDrive）")
        sys.exit(1)
    dest_root = base / SUBDIR
    print(f"目標：{name} → {dest_root}")

    pairs = list(iter_pairs(dest_root))
    todo = [(s, d) for s, d in pairs if needs_copy(s, d)]
    total_all = sum(s.stat().st_size for s, _ in pairs)
    total_new = sum(s.stat().st_size for s, _ in todo)
    print(f"來源共 {len(pairs)} 檔 / {human(total_all)}")
    print(f"需同步 {len(todo)} 檔 / {human(total_new)}"
          f"（{'全量首次' if len(todo) == len(pairs) else '增量'}）")

    if a.dry_run:
        for s, d in todo[:15]:
            print(f"  [dry] {s.relative_to(ROOT)} → {human(s.stat().st_size)}")
        if len(todo) > 15:
            print(f"  ... 共 {len(todo)} 檔")
        return

    ok = fail = 0
    done_bytes = 0
    t0 = time.time()
    for i, (s, d) in enumerate(todo, 1):
        try:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)          # copy2 保留 mtime → 下次可正確判斷增量
            ok += 1
            done_bytes += s.stat().st_size
        except Exception as e:
            fail += 1
            print(f"  ⚠️ {s.name}: {str(e)[:100]}")
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)} | {human(done_bytes)} | "
                  f"{human(done_bytes/max(el,0.1))}/s", flush=True)

    print(f"\n完成：成功 {ok} 檔、失敗 {fail} 檔、{human(done_bytes)}、"
          f"耗時 {time.time()-t0:.0f}s")
    print("Google Drive 客戶端會自動把這些檔案上傳到雲端。")


if __name__ == "__main__":
    main()
