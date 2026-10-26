import ccxt
import pandas as pd
import time
import os
import sys
import logging
from datetime import datetime


sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ==========================================
# 🔧 修改 1: 強制鎖定工作目錄 (解決 PM2/D槽 路徑問題)
# ==========================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- 設定區 ---
SYMBOL_PERP = 'BNB/USDT:USDT'
SYMBOL_SPOT = 'BNB/USDT'
SYMBOL_BTC  = 'BTC/USDT:USDT'
LIMIT_OB = 10
DATA_DIR = 'D:\\code\\python\\BNB_predict\\Data_Collcet\\Data'
LOG_DIR = 'D:\\code\\python\\BNB_predict\\Data_Collcet\\Logs'

# --- 初始化區 ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, 'system.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

exchange_perp = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

exchange_spot = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

def get_raw_data():
    try:
        ts = exchange_perp.milliseconds()
        ob = exchange_perp.fetch_order_book(SYMBOL_PERP, limit=LIMIT_OB)
        perp_ohlcv = exchange_perp.fetch_ohlcv(SYMBOL_PERP, timeframe='1m', limit=1)
        perp_candle = perp_ohlcv[0]
        funding = exchange_perp.fetch_funding_rate(SYMBOL_PERP)
        spot_ticker = exchange_spot.fetch_ticker(SYMBOL_SPOT)
        btc_ticker = exchange_perp.fetch_ticker(SYMBOL_BTC)

        data = {
            'timestamp': ts,
            'readable_time': datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M:%S'),
            'open': perp_candle[1],
            'high': perp_candle[2],
            'low': perp_candle[3],
            'close': perp_candle[4],
            'volume': perp_candle[5],
            'funding_rate': funding['fundingRate'],
            'bnb_spot_close': spot_ticker['last'],
            'btc_perp_close': btc_ticker['last'],
        }

        bids = ob['bids']
        asks = ob['asks']
        
        for i in range(LIMIT_OB):
            if i < len(bids):
                data[f'bid_p_{i+1}'] = bids[i][0]
                data[f'bid_q_{i+1}'] = bids[i][1]
            else:
                data[f'bid_p_{i+1}'] = 0
                data[f'bid_q_{i+1}'] = 0
            
            if i < len(asks):
                data[f'ask_p_{i+1}'] = asks[i][0]
                data[f'ask_q_{i+1}'] = asks[i][1]
            else:
                data[f'ask_p_{i+1}'] = 0
                data[f'ask_q_{i+1}'] = 0
                
        return data

    except Exception as e:
        logging.error(f"抓取數據失敗: {e}")
        return None

def save_to_daily_csv(data_dict):
    if data_dict is None:
        return

    today_str = datetime.now().strftime('%Y-%m-%d')
    filename = f'bnb_raw_{today_str}.csv'
    file_path = os.path.join(DATA_DIR, filename)
    
    df = pd.DataFrame([data_dict])
    
    try:
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False, mode='w', encoding='utf-8')
            logging.info(f"建立新檔案: {filename}")
        else:
            df.to_csv(file_path, index=False, mode='a', header=False, encoding='utf-8')
            
    except Exception as e:
        logging.error(f"寫入 CSV 失敗: {e}")

def main():
    logging.info(f"目標: {SYMBOL_PERP} (每1分鐘)")
    logging.info(f"資料路徑: {os.path.abspath(DATA_DIR)}") # 顯示絕對路徑確認
    
    while True:
        try:
            now = datetime.now()
            second = now.second
            
            if second >= 58:
                logging.info(f"[{now.strftime('%H:%M:%S')}] 觸發抓取...")
                raw_data = get_raw_data()
                
                if raw_data:
                    save_to_daily_csv(raw_data)
                    logging.info(f"存檔成功! Price: {raw_data['close']} | Spot: {raw_data['bnb_spot_close']}")
                
                time.sleep(3) 
            else:
                time.sleep(0.5)

        except KeyboardInterrupt:
            logging.info("程式由使用者手動停止")
            break
        except Exception as e:
            logging.error(f"主迴圈發生未知錯誤: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()