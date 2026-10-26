# -*- coding: utf-8 -*-
"""
每日排程入口（Windows 工作排程器 BNB_Quant_DailySync，每天 03:30 台北 = 19:30 UTC）

依序：
  1. update_daily.py            日線（兩版）延伸到昨天(UTC) + symbols_meta + 重建 PIT 宇宙
  2. update_funding.py --daily  funding 延伸到昨天 + 新上市幣完整歷史
  3. backfill_metrics.py daily  重建回補計畫（新晉前 150 名自動加入）→ 倉位歷史延伸到昨天，
                                只重寫有變動的月檔
  4. sync_to_cloud.py           增量同步到 Google Drive

前面步驟失敗不阻擋後面（尤其同步——ALPHA 資料的備份優先）；任一步失敗則以非 0 結束，
工作排程器的「上次執行結果」會顯示出來。輸出附加到 Data/daily_update.log。
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
TOOLS = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("日線與 PIT 宇宙", [PY, "-I", "-u", str(TOOLS / "update_daily.py")]),
    ("funding", [PY, "-I", "-u", str(TOOLS / "update_funding.py"), "--daily"]),
    ("倉位歷史延伸", [PY, "-I", "-u", str(TOOLS / "backfill_metrics.py"), "daily"]),
    ("雲端同步", [PY, "-u", str(TOOLS / "sync_to_cloud.py")]),
]


def main():
    print(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} 每日更新 =====", flush=True)
    failed = []
    for name, cmd in STEPS:
        print(f"--- {name} ---", flush=True)
        rc = subprocess.run(cmd, cwd=TOOLS).returncode     # 子程序輸出直接寫入同一個 log
        if rc != 0:
            failed.append(name)
            print(f"⚠️ {name} 失敗（exit {rc}）", flush=True)
    print(f"===== 結束：{'全部成功' if not failed else '失敗：' + '、'.join(failed)} =====", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
