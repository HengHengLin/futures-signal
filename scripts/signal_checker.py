"""
期货季节性信号检测器
每日收盘后自动运行，输出 index.html 到根目录
"""

import json
import os
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional

import akshare as ak
import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

# 北京时间
CST = timezone(timedelta(hours=8))
TODAY = datetime.now(CST)
TODAY_STR = TODAY.strftime("%Y-%m-%d")
TODAY_MONTH = TODAY.month
TODAY_DAY = TODAY.day

# 飞书 webhook（从 GitHub Secret 读取，没有则跳过）
FEISHU_WEBHOOK = os.environ.get("FEISHU_WEBHOOK", "")

# ─────────────────────────────────────────────
# 品种配置（来自第二张表）
# ─────────────────────────────────────────────

STRATEGIES = [
    {
        "id": "RB",
        "name": "螺纹钢",
        "exchange": "SHFE",
        "ak_symbol": "RB0",
        "direction": "long",
        "entry_month": 2,
        "entry_day_start": 5,
        "entry_day_end": 20,
        "entry_desc": "2月春节后",
        "exit_month": 4,
        "exit_day_start": 10,
        "exit_day_end": 20,
        "exit_desc": "4月中旬",
        "cross_year": False,
        "contract": "05合约",
        "stated_winrate": 80,
        "logic": "全国建筑工地元宵节后全面复工，需求确定性回归",
    },
    {
        "id": "BU",
        "name": "沥青",
        "exchange": "SHFE",
        "ak_symbol": "BU0",
        "direction": "short",
        "entry_month": 11,
        "entry_day_start": 1,
        "entry_day_end": 15,
        "entry_desc": "11月上旬",
        "exit_month": 12,
        "exit_day_start": 20,
        "exit_day_end": 31,
        "exit_desc": "12月下旬",
        "cross_year": False,
        "contract": "01或06合约",
        "stated_winrate": 80,
        "logic": "北方入冬道路停工，需求进入物理冰封期",
    },
    {
        "id": "I",
        "name": "铁矿石",
        "exchange": "DCE",
        "ak_symbol": "I0",
        "direction": "long",
        "entry_month": 11,
        "entry_day_start": 10,
        "entry_day_end": 20,
        "entry_desc": "11月中旬",
        "exit_month": 1,
        "exit_day_start": 10,
        "exit_day_end": 20,
        "exit_desc": "次年1月中旬",
        "cross_year": True,
        "contract": "05合约",
        "stated_winrate": 82,
        "logic": "钢厂冬储补库，澳洲/巴西雨季供应收缩",
    },
    {
        "id": "JD_short",
        "name": "鸡蛋(秋季空)",
        "exchange": "DCE",
        "ak_symbol": "JD0",
        "direction": "short",
        "entry_month": 9,
        "entry_day_start": 15,
        "entry_day_end": 25,
        "entry_desc": "9月中旬(中秋后)",
        "exit_month": 10,
        "exit_day_start": 20,
        "exit_day_end": 31,
        "exit_desc": "10月下旬",
        "cross_year": False,
        "contract": "01合约",
        "stated_winrate": 85,
        "logic": "中秋国庆备货结束，秋季产蛋率全年高位，供需错配",
    },
    {
        "id": "JD_long",
        "name": "鸡蛋(夏季多)",
        "exchange": "DCE",
        "ak_symbol": "JD0",
        "direction": "long",
        "entry_month": 6,
        "entry_day_start": 20,
        "entry_day_end": 30,
        "entry_desc": "6月下旬",
        "exit_month": 8,
        "exit_day_start": 10,
        "exit_day_end": 20,
        "exit_desc": "8月中旬",
        "cross_year": False,
        "contract": "09合约",
        "stated_winrate": 80,
        "logic": "盛夏高温产蛋率下降，中秋月饼厂提前2个月大规模采购",
    },
    {
        "id": "JD_spring",
        "name": "鸡蛋(春季空)",
        "exchange": "DCE",
        "ak_symbol": "JD0",
        "direction": "short",
        "entry_month": 2,
        "entry_day_start": 20,
        "entry_day_end": 28,
        "entry_desc": "2月下旬",
        "exit_month": 4,
        "exit_day_start": 20,
        "exit_day_end": 30,
        "exit_desc": "4月下旬",
        "cross_year": False,
        "contract": "05合约",
        "stated_winrate": 79,
        "logic": "春节后消费淡季，梅雨季贸易商被迫低价去库存",
    },
    {
        "id": "M",
        "name": "豆粕",
        "exchange": "DCE",
        "ak_symbol": "M0",
        "direction": "long",
        "entry_month": 6,
        "entry_day_start": 20,
        "entry_day_end": 30,
        "entry_desc": "6月下旬",
        "exit_month": 8,
        "exit_day_start": 10,
        "exit_day_end": 20,
        "exit_desc": "8月中旬",
        "cross_year": False,
        "contract": "09合约",
        "stated_winrate": 80,
        "logic": "美豆生长季天气升水炒作，国内饲料需求旺季",
    },
]

# ─────────────────────────────────────────────
# 技术指标计算
# ─────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    up   = high - high.shift()
    down = low.shift() - low
    dm_plus  = up.where((up > down) & (up > 0), 0)
    dm_minus = down.where((down > up) & (down > 0), 0)

    di_plus  = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss
    return 100 - 100 / (1 + rs)


def calc_bollinger(series: pd.Series, period: int = 20):
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    bw    = (upper - lower) / mid * 100  # bandwidth %
    return upper, lower, bw


def calc_bias(series: pd.Series, period: int = 20) -> pd.Series:
    ma = series.rolling(period).mean()
    return (series - ma) / ma * 100


def get_indicators(df: pd.DataFrame) -> dict:
    """计算所有需要的技术指标，返回最新值"""
    close = df["close"]

    adx   = calc_adx(df).iloc[-1]
    rsi   = calc_rsi(close).iloc[-1]
    ema20 = calc_ema(close, 20).iloc[-1]
    ema50 = calc_ema(close, 50).iloc[-1]
    _, _, bw = calc_bollinger(close)
    bw_now = bw.iloc[-1]
    bw_min = bw.iloc[-60:].min()          # 近60日最低带宽
    bias  = calc_bias(close).iloc[-1]
    price = close.iloc[-1]
    _, bb_upper, _ = calc_bollinger(close)
    _, bb_lower, _ = calc_bollinger(close)
    # 重新算一次拿上下轨
    ma20  = close.rolling(20).mean().iloc[-1]
    std20 = close.rolling(20).std().iloc[-1]
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20

    return {
        "price":    round(float(price), 2),
        "adx":      round(float(adx), 2),
        "rsi":      round(float(rsi), 2),
        "ema20":    round(float(ema20), 2),
        "ema50":    round(float(ema50), 2),
        "ema_cross": "golden" if ema20 > ema50 else "dead",
        "bw_now":   round(float(bw_now), 2),
        "bw_min":   round(float(bw_min), 2),
        "bw_expanding": bool(bw_now > bw_min * 1.5),  # 带宽从极低位开口
        "bias":     round(float(bias), 2),
        "bb_upper": round(float(upper), 2),
        "bb_lower": round(float(lower), 2),
        "price_break_upper": bool(price > upper),
        "price_break_lower": bool(price < lower),
    }


# ─────────────────────────────────────────────
# 市场状态判断（第一张表）
# ─────────────────────────────────────────────

def judge_market_state(ind: dict) -> dict:
    adx  = ind["adx"]
    rsi  = ind["rsi"]
    bias = ind["bias"]

    # 状态4：极端行情（优先判断）
    if abs(bias) > 5 or rsi > 80 or rsi < 20:
        return {
            "state": 4,
            "name": "极端行情",
            "desc": "乖离率过高或RSI极端，逢高减仓",
            "position": "轻仓 / 减仓",
            "action": "分批减仓，保留利润",
            "color": "red",
        }

    # 状态3：缓慢趋势（鱼身行情）
    if adx > 25 and ind["ema_cross"] in ("golden", "dead"):
        direction = "做多方向" if ind["ema_cross"] == "golden" else "做空方向"
        return {
            "state": 3,
            "name": "缓慢趋势（鱼身）",
            "desc": f"ADX={adx:.1f}>25，EMA {'金叉' if ind['ema_cross']=='golden' else '死叉'}",
            "position": "重仓 80-100%",
            "action": f"顺趋势 {direction}，盈亏比最高",
            "color": "green",
        }

    # 状态2：趋势启动（变盘）
    if ind["bw_expanding"] and (ind["price_break_upper"] or ind["price_break_lower"]):
        direction = "向上突破" if ind["price_break_upper"] else "向下突破"
        return {
            "state": 2,
            "name": "趋势启动",
            "desc": f"布林带开口+价格{direction}，变盘信号",
            "position": "中仓 50%",
            "action": f"顺势切入，{direction}",
            "color": "blue",
        }

    # 状态1：窄幅震荡
    return {
        "state": 1,
        "name": "窄幅震荡",
        "desc": f"ADX={adx:.1f}<20，带宽收窄，震荡格局",
        "position": "轻仓 20-30%",
        "action": "高抛低吸，双向轻仓",
        "color": "yellow",
    }


# ─────────────────────────────────────────────
# 季节性窗口判断（第二张表）
# ─────────────────────────────────────────────

def get_window_status(strat: dict) -> str:
    m, d = TODAY_MONTH, TODAY_DAY

    if m == strat["entry_month"] and strat["entry_day_start"] <= d <= strat["entry_day_end"]:
        return "active"      # 窗口开启中

    # 2周内即将开启
    target = datetime(TODAY.year, strat["entry_month"], strat["entry_day_start"], tzinfo=CST)
    if target < TODAY:
        target = datetime(TODAY.year + 1, strat["entry_month"], strat["entry_day_start"], tzinfo=CST)
    days_to = (target - TODAY).days
    if 0 < days_to <= 14:
        return "soon"

    return "inactive"


def check_signal(strat: dict, market_state: dict, ind: dict) -> dict:
    """综合判断：季节性窗口 × 市场状态"""
    ws = get_window_status(strat)

    # 窗口内才考虑入场
    if ws != "active":
        return {"signal": False, "window": ws, "reason": "不在季节性窗口"}

    state = market_state["state"]
    direction = strat["direction"]

    # 状态4极端行情不开新仓
    if state == 4:
        return {"signal": False, "window": ws, "reason": "极端行情，不开新仓"}

    # 状态1震荡期，季节性信号效果差
    if state == 1:
        return {"signal": False, "window": ws, "reason": "震荡行情，等待趋势确认"}

    # 状态2/3：检查方向是否匹配EMA
    if state == 3:
        ema_ok = (direction == "long" and ind["ema_cross"] == "golden") or \
                 (direction == "short" and ind["ema_cross"] == "dead")
        if not ema_ok:
            return {"signal": False, "window": ws, "reason": "季节性方向与EMA趋势相反，观望"}

    return {
        "signal": True,
        "window": ws,
        "reason": f"季节性窗口开启 + 市场状态{state}匹配",
    }


# ─────────────────────────────────────────────
# 数据获取
# ─────────────────────────────────────────────

def fetch_futures_data(ak_symbol: str, days: int = 120) -> Optional[pd.DataFrame]:
    """拉取主力合约日线，返回含 open/high/low/close/volume 的 DataFrame"""
    end   = TODAY.strftime("%Y%m%d")
    start = (TODAY - timedelta(days=days)).strftime("%Y%m%d")
    try:
        df = ak.futures_main_sina(symbol=ak_symbol, start_date=start, end_date=end)
        df.columns = [c.lower() for c in df.columns]
        # 统一列名
        col_map = {}
        for c in df.columns:
            if any(k in c for k in ["date", "日期", "time"]): col_map[c] = "date"
            elif any(k in c for k in ["open", "开盘"]):  col_map[c] = "open"
            elif any(k in c for k in ["high", "最高"]):  col_map[c] = "high"
            elif any(k in c for k in ["low",  "最低"]):  col_map[c] = "low"
            elif any(k in c for k in ["close","收盘"]):  col_map[c] = "close"
            elif any(k in c for k in ["vol",  "成交量"]): col_map[c] = "volume"
        df = df.rename(columns=col_map)
        df["date"]  = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["high"]  = pd.to_numeric(df.get("high", df["close"]), errors="coerce")
        df["low"]   = pd.to_numeric(df.get("low",  df["close"]), errors="coerce")
        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        return df if len(df) >= 30 else None
    except Exception:
        return None


# ─────────────────────────────────────────────
# HTML 生成
# ─────────────────────────────────────────────

STATE_COLORS = {
    "green":  ("#00e5a0", "rgba(0,229,160,0.1)"),
    "blue":   ("#4da6ff", "rgba(77,166,255,0.1)"),
    "yellow": ("#f5c842", "rgba(245,200,66,0.1)"),
    "red":    ("#ff4d6a", "rgba(255,77,106,0.1)"),
}

def render_signal_card(strat: dict, ind: Optional[dict],
                       market: Optional[dict], sig: Optional[dict]) -> str:
    name      = strat["name"]
    direction = "做多 ↑" if strat["direction"] == "long" else "做空 ↓"
    dir_color = "#00e5a0" if strat["direction"] == "long" else "#ff4d6a"
    winrate   = strat["stated_winrate"]
    wr_color  = "#00e5a0" if winrate >= 82 else "#f5c842"

    # 数据获取失败
    if ind is None or market is None or sig is None:
        return f"""
        <div class="card card-dim">
          <div class="card-left">
            <div class="card-top">
              <span class="cname">{name}</span>
              <span class="badge" style="color:{dir_color};border-color:{dir_color}">{direction}</span>
              <span class="badge-grey">数据获取失败</span>
            </div>
            <div class="card-meta">{strat['entry_desc']} → {strat['exit_desc']} · {strat['contract']}</div>
            <div class="card-logic">{strat['logic']}</div>
          </div>
          <div class="card-right">
            <div class="winrate" style="color:{wr_color}">{winrate}%</div>
            <div class="winrate-label">10Y胜率</div>
          </div>
        </div>"""

    ws        = sig["window"]
    triggered = sig["signal"]
    reason    = sig["reason"]

    # 窗口状态
    if ws == "active":
        dot_color = "#00e5a0"; ws_text = "窗口开启中"
    elif ws == "soon":
        dot_color = "#f5c842"; ws_text = "2周内开启"
    else:
        dot_color = "#4a5a6a"; ws_text = "窗口未到"

    # 信号状态
    if triggered:
        border_color = dir_color
        sig_badge = f'<span class="sig-badge" style="background:{dir_color}20;color:{dir_color};border:1px solid {dir_color}">🔔 信号触发</span>'
    elif ws == "active":
        border_color = "#f5c842"
        sig_badge = f'<span class="sig-badge" style="background:#f5c84220;color:#f5c842;border:1px solid #f5c842">⚠ 窗口开启·条件未满足</span>'
    elif ws == "soon":
        border_color = "#4a5a6a"
        sig_badge = f'<span class="sig-badge" style="background:#4a5a6a20;color:#4a5a6a;border:1px solid #4a5a6a">⏳ 即将开启</span>'
    else:
        border_color = "#1e2530"
        sig_badge = ""

    # 指标行
    adx_color = "#00e5a0" if ind["adx"] > 25 else ("#f5c842" if ind["adx"] > 20 else "#4a5a6a")
    rsi_color = "#ff4d6a" if ind["rsi"] > 70 else ("#00e5a0" if ind["rsi"] < 30 else "#c8d4e0")
    ema_text  = "金叉↑" if ind["ema_cross"] == "golden" else "死叉↓"
    ema_color = "#00e5a0" if ind["ema_cross"] == "golden" else "#ff4d6a"
    bias_color = "#ff4d6a" if abs(ind["bias"]) > 5 else "#c8d4e0"

    m_color = STATE_COLORS.get(market["color"], STATE_COLORS["yellow"])[0]

    return f"""
    <div class="card" style="border-left:3px solid {border_color}">
      <div class="card-left">
        <div class="card-top">
          <span class="cname">{name}</span>
          <span class="badge" style="color:{dir_color};border-color:{dir_color}">{direction}</span>
          {sig_badge}
        </div>
        <div class="card-meta">{strat['entry_desc']} → {strat['exit_desc']} · {strat['contract']}</div>

        <div class="ind-row">
          <span class="ind-item">ADX <b style="color:{adx_color}">{ind['adx']}</b></span>
          <span class="ind-item">RSI <b style="color:{rsi_color}">{ind['rsi']}</b></span>
          <span class="ind-item">EMA <b style="color:{ema_color}">{ema_text}</b></span>
          <span class="ind-item">BIAS <b style="color:{bias_color}">{ind['bias']:+.1f}%</b></span>
          <span class="ind-item">价格 <b style="color:#e8f0f8">{ind['price']}</b></span>
        </div>

        <div class="state-row">
          <span style="color:{m_color}">▸ STATE {market['state']} {market['name']}</span>
          <span style="color:#4a5a6a;margin-left:12px">{market['position']}</span>
        </div>

        <div class="window-row">
          <span class="dot" style="background:{dot_color}"></span>
          <span style="color:{dot_color}">{ws_text}</span>
          <span style="color:#4a5a6a;margin-left:10px">{reason}</span>
        </div>

        <div class="card-logic">{strat['logic']}</div>
      </div>
      <div class="card-right">
        <div class="winrate" style="color:{wr_color}">{winrate}%</div>
        <div class="winrate-label">10Y胜率</div>
        <div class="price-tag">{ind['price']}</div>
      </div>
    </div>"""


def build_html(results: list) -> str:
    triggered = [r for r in results if r.get("sig") and r["sig"]["signal"]]
    soon      = [r for r in results if r.get("sig") and r["sig"]["window"] == "soon"]

    cards_html = ""
    # 排序：触发 > 窗口开启未触发 > 即将 > 其他
    def sort_key(r):
        if not r.get("sig"): return 9
        if r["sig"]["signal"]: return 0
        if r["sig"]["window"] == "active": return 1
        if r["sig"]["window"] == "soon": return 2
        return 3
    for r in sorted(results, key=sort_key):
        cards_html += render_signal_card(
            r["strat"], r.get("ind"), r.get("market"), r.get("sig")
        )

    triggered_count = len(triggered)
    soon_count      = len(soon)
    trigger_color   = "#00e5a0" if triggered_count > 0 else "#4a5a6a"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>期货信号 · {TODAY_STR}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
:root{{
  --bg:#0a0c0f;--surface:#111418;--border:#1e2530;
  --text:#c8d4e0;--bright:#e8f0f8;--dim:#4a5a6a;
  --green:#00e5a0;--red:#ff4d6a;--yellow:#f5c842;--blue:#4da6ff;
  --mono:'JetBrains Mono',monospace;--sans:'Noto Sans SC',sans-serif;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;padding:28px 32px;max-width:960px;margin:0 auto}}
body::before{{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(30,37,48,.3) 1px,transparent 1px),linear-gradient(90deg,rgba(30,37,48,.3) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}}
.wrap{{position:relative;z-index:1}}
.header{{display:flex;justify-content:space-between;align-items:baseline;padding-bottom:16px;border-bottom:1px solid var(--border);margin-bottom:20px}}
.title{{font-family:var(--mono);font-size:15px;font-weight:700;color:var(--bright);letter-spacing:.1em;text-transform:uppercase}}
.date{{font-family:var(--mono);font-size:11px;color:var(--dim)}}
.statusbar{{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap}}
.pill{{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:7px 14px;font-family:var(--mono);font-size:11px}}
.pill .lbl{{color:var(--dim);margin-right:6px}}
.pill .val{{color:var(--bright);font-weight:600}}
.cards{{display:flex;flex-direction:column;gap:10px}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:14px 16px;display:flex;gap:16px;transition:border-color .15s}}
.card:hover{{border-color:#2a3545}}
.card-dim{{opacity:.45}}
.card-left{{flex:1}}
.card-right{{text-align:right;min-width:64px}}
.card-top{{display:flex;align-items:center;gap:8px;margin-bottom:7px;flex-wrap:wrap}}
.cname{{font-family:var(--mono);font-size:14px;font-weight:700;color:var(--bright)}}
.badge{{font-family:var(--mono);font-size:10px;font-weight:700;padding:2px 8px;border-radius:3px;border:1px solid;letter-spacing:.05em}}
.badge-grey{{font-family:var(--mono);font-size:10px;color:var(--dim);border:1px solid var(--border);padding:2px 8px;border-radius:3px}}
.sig-badge{{font-family:var(--mono);font-size:10px;font-weight:600;padding:2px 8px;border-radius:3px}}
.card-meta{{font-size:11px;color:var(--dim);margin-bottom:8px}}
.ind-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:6px}}
.ind-item{{font-family:var(--mono);font-size:11px;color:var(--dim)}}
.ind-item b{{font-weight:600}}
.state-row{{font-family:var(--mono);font-size:11px;margin-bottom:5px}}
.window-row{{display:flex;align-items:center;gap:6px;margin-bottom:6px;font-family:var(--mono);font-size:10px}}
.dot{{width:6px;height:6px;border-radius:50%;display:inline-block;flex-shrink:0}}
.card-logic{{font-size:11px;color:var(--dim);line-height:1.6}}
.winrate{{font-family:var(--mono);font-size:22px;font-weight:700;line-height:1}}
.winrate-label{{font-family:var(--mono);font-size:9px;color:var(--dim);margin-top:3px}}
.price-tag{{font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:8px}}
.footer{{margin-top:24px;padding-top:14px;border-top:1px solid var(--border);display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;color:var(--dim)}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div>
      <div class="title">FUTURES SIGNAL</div>
      <div style="font-family:var(--mono);font-size:10px;color:var(--dim);margin-top:3px">季节性窗口 × 技术状态</div>
    </div>
    <div class="date">{TODAY_STR} 收盘</div>
  </div>

  <div class="statusbar">
    <div class="pill" style="border-color:{trigger_color}">
      <span class="lbl">触发信号</span>
      <span class="val" style="color:{trigger_color}">{triggered_count}</span>
    </div>
    <div class="pill">
      <span class="lbl">即将开启</span>
      <span class="val">{soon_count}</span>
    </div>
    <div class="pill">
      <span class="lbl">品种数</span>
      <span class="val">{len(results)}</span>
    </div>
    <div class="pill">
      <span class="lbl">更新时间</span>
      <span class="val">{TODAY.strftime('%H:%M')} CST</span>
    </div>
  </div>

  <div class="cards">
    {cards_html}
  </div>

  <div class="footer">
    <span>数据来源：AKShare · 胜率基于历史10年统计 · 仅供参考</span>
    <span>自动更新 via GitHub Actions</span>
  </div>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────
# 飞书推送
# ─────────────────────────────────────────────

def push_feishu(results: list, positions: list, closed: list, closed_trades: list):
    if not FEISHU_WEBHOOK:
        return

    triggered = [r for r in results if r.get("sig") and r["sig"]["signal"]]
    soon      = [r for r in results if r.get("sig") and r["sig"]["window"] == "soon" and not r["sig"]["signal"]]
    open_pos  = [p for p in positions if p["status"] == "open"]

    lines = []

    # ── 标题 ──
    if triggered:
        lines.append(f"🔔 期货信号日报  {TODAY_STR}  【{len(triggered)}个信号触发】")
    else:
        lines.append(f"📊 期货信号日报  {TODAY_STR}")
    lines.append("─" * 36)

    # ── 今日平仓 ──
    if closed:
        lines.append("【今日平仓】")
        for p in closed:
            icon = "✅" if p["pnl_pct"] > 0 else "❌"
            lines.append(
                f"  {icon} {p['name']}  "
                f"入:{p['entry_price']} → 出:{p['exit_price']}  "
                f"盈亏:{p['pnl_pct']:+.2f}%  持仓{p['hold_days']}天"
            )
        lines.append("─" * 36)

    # ── 触发新信号 ──
    if triggered:
        lines.append("【新信号触发 → 模拟开仓】")
        for r in triggered:
            s, ind, mkt = r["strat"], r["ind"], r["market"]
            d = "做多↑" if s["direction"] == "long" else "做空↓"
            lines.append(f"  🟢 {s['name']} {d}  入场价:{ind['price']}  胜率:{s['stated_winrate']}%")
            lines.append(f"     STATE{mkt['state']} {mkt['name']}  目标平仓:{s['exit_desc']}")
        lines.append("─" * 36)

    # ── 当前模拟持仓 ──
    if open_pos:
        lines.append(f"【模拟持仓 ({len(open_pos)}个)】")
        for p in open_pos:
            pnl = p.get("pnl_pct", 0)
            icon = "📈" if pnl > 0 else ("📉" if pnl < 0 else "➡️")
            d = "多" if p["direction"] == "long" else "空"
            lines.append(
                f"  {icon} {p['name']}({d})  "
                f"入:{p['entry_price']} 现:{p.get('current_price','—')}  "
                f"浮盈:{pnl:+.2f}%  {p['hold_days']}天"
            )
        lines.append("─" * 36)
    else:
        lines.append("【模拟持仓】暂无持仓")
        lines.append("─" * 36)

    # ── 即将开启 ──
    if soon:
        lines.append("【即将开启（2周内）】")
        for r in soon:
            s, ind = r["strat"], r["ind"]
            d = "做多↑" if s["direction"] == "long" else "做空↓"
            price_str = f"  现价:{ind['price']}" if ind else ""
            lines.append(f"  ⏳ {s['name']} {d}  建仓:{s['entry_desc']}{price_str}")
        lines.append("─" * 36)

    # ── 全品种数据快照 ──
    lines.append("【全品种指标快照】")
    for r in results:
        s   = r["strat"]
        ind = r.get("ind")
        mkt = r.get("market")
        sig = r.get("sig")
        d   = "多" if s["direction"] == "long" else "空"

        if ind is None:
            lines.append(f"  ❌ {s['name']}({d})  数据获取失败")
            continue

        ws       = sig["window"] if sig else "—"
        ws_icon  = "🟢" if ws == "active" else ("🟡" if ws == "soon" else "⚪")
        state_n  = mkt["state"] if mkt else "—"
        ema      = "金叉" if ind["ema_cross"] == "golden" else "死叉"
        lines.append(
            f"  {ws_icon} {s['name']}({d})  "
            f"价:{ind['price']}  ADX:{ind['adx']}  RSI:{ind['rsi']}  "
            f"EMA:{ema}  BIAS:{ind['bias']:+.1f}%  S{state_n}"
        )

    # ── 历史战绩 ──
    if closed_trades:
        wins  = sum(1 for t in closed_trades if t.get("pnl_pct", 0) > 0)
        total = len(closed_trades)
        avg   = sum(t.get("pnl_pct", 0) for t in closed_trades) / total
        lines.append("─" * 36)
        lines.append(
            f"【观察期战绩】共{total}笔  "
            f"胜率:{wins/total*100:.0f}%  "
            f"平均盈亏:{avg:+.2f}%"
        )

    # ── 页脚 ──
    lines.append("─" * 36)
    lines.append(f"🔗 https://henghenglin.github.io/futures-signal")

    text = "\n".join(lines)
    try:
        requests.post(
            FEISHU_WEBHOOK,
            json={"msg_type": "text", "content": {"text": text}},
            timeout=10
        )
    except Exception:
        pass


# ─────────────────────────────────────────────
# 持仓追踪模块
# ─────────────────────────────────────────────

POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "..", "positions.json")


def load_positions() -> list:
    """读取当前模拟持仓"""
    if not os.path.exists(POSITIONS_FILE):
        return []
    with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_positions(positions: list):
    with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)


def update_positions(results: list) -> tuple[list, list]:
    """
    1. 检查现有持仓是否到达平仓窗口 → 自动平仓，记录结果
    2. 检查今日信号触发 → 自动开仓
    返回 (当前持仓列表, 今日平仓记录列表)
    """
    positions  = load_positions()
    closed     = []   # 今日平仓记录
    new_positions = []

    # ── Step1: 检查现有持仓是否应该平仓 ──
    for pos in positions:
        strat = next((s for s in STRATEGIES if s["id"] == pos["strategy_id"]), None)
        if strat is None:
            new_positions.append(pos)
            continue

        # 找对应品种当前价格
        cur_price = None
        for r in results:
            if r["strat"]["id"] == pos["strategy_id"] and r.get("ind"):
                cur_price = r["ind"]["price"]
                break

        if cur_price:
            pos["current_price"] = cur_price
            # 计算当前浮动盈亏
            if pos["direction"] == "long":
                pos["pnl_pct"] = round((cur_price - pos["entry_price"]) / pos["entry_price"] * 100, 2)
            else:
                pos["pnl_pct"] = round((pos["entry_price"] - cur_price) / pos["entry_price"] * 100, 2)
            pos["hold_days"] = (TODAY - datetime.fromisoformat(pos["entry_date"]).replace(tzinfo=CST)).days

        # 判断是否到平仓窗口
        exit_year  = TODAY.year
        exit_month = strat["exit_month"]
        in_exit_window = (
            TODAY_MONTH == exit_month and
            strat["exit_day_start"] <= TODAY_DAY <= strat["exit_day_end"]
        )
        # 跨年品种特殊处理
        if strat.get("cross_year") and TODAY_MONTH == exit_month:
            in_exit_window = True

        if in_exit_window and cur_price:
            # 平仓
            pos["exit_date"]  = TODAY_STR
            pos["exit_price"] = cur_price
            pos["status"]     = "closed"
            closed.append(pos)
            print(f"  📤 平仓: {pos['name']}  盈亏:{pos['pnl_pct']:+.2f}%  持仓{pos['hold_days']}天")
        else:
            pos["status"] = "open"
            new_positions.append(pos)

    # ── Step2: 检查今日新信号，开仓 ──
    existing_ids = {p["strategy_id"] for p in new_positions}
    for r in results:
        sig  = r.get("sig")
        ind  = r.get("ind")
        strat = r["strat"]

        if not sig or not sig["signal"] or ind is None:
            continue
        if strat["id"] in existing_ids:
            # 已有持仓，不重复开
            continue

        new_pos = {
            "strategy_id":  strat["id"],
            "name":         strat["name"],
            "direction":    strat["direction"],
            "entry_date":   TODAY_STR,
            "entry_price":  ind["price"],
            "current_price": ind["price"],
            "exit_target":  f"{strat['exit_month']}月{strat['exit_day_start']}-{strat['exit_day_end']}日",
            "contract":     strat["contract"],
            "stated_winrate": strat["stated_winrate"],
            "pnl_pct":      0.0,
            "hold_days":    0,
            "status":       "open",
        }
        new_positions.append(new_pos)
        existing_ids.add(strat["id"])
        print(f"  📥 开仓: {strat['name']}  入场价:{ind['price']}")

    save_positions(new_positions)

    # ── Step3: 保存平仓记录 ──
    if closed:
        closed_file = os.path.join(os.path.dirname(__file__), "..", "closed_trades.json")
        existing_closed = []
        if os.path.exists(closed_file):
            with open(closed_file, "r", encoding="utf-8") as f:
                existing_closed = json.load(f)
        existing_closed.extend(closed)
        with open(closed_file, "w", encoding="utf-8") as f:
            json.dump(existing_closed, f, ensure_ascii=False, indent=2)

    return new_positions, closed


def load_closed_trades() -> list:
    closed_file = os.path.join(os.path.dirname(__file__), "..", "closed_trades.json")
    if not os.path.exists(closed_file):
        return []
    with open(closed_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*50}")
    print(f"  期货信号检测  {TODAY_STR}")
    print(f"{'='*50}\n")

    results = []

    for strat in STRATEGIES:
        print(f"[{strat['name']}] 拉取数据...")
        df = fetch_futures_data(strat["ak_symbol"])

        if df is None or df.empty:
            print(f"  ✗ 数据获取失败")
            results.append({"strat": strat, "ind": None, "market": None, "sig": None})
            continue

        try:
            ind    = get_indicators(df)
            market = judge_market_state(ind)
            sig    = check_signal(strat, market, ind)

            status = "🔔 信号触发" if sig["signal"] else f"  {sig['reason']}"
            print(f"  价格:{ind['price']}  ADX:{ind['adx']}  RSI:{ind['rsi']}  "
                  f"EMA:{ind['ema_cross']}  BIAS:{ind['bias']:+.1f}%")
            print(f"  市场状态: STATE{market['state']} {market['name']}")
            print(f"  信号: {status}")

            results.append({"strat": strat, "ind": ind, "market": market, "sig": sig})
        except Exception:
            print(f"  ✗ 指标计算失败:")
            traceback.print_exc()
            results.append({"strat": strat, "ind": None, "market": None, "sig": None})

        print()

    # 生成 HTML
    html = build_html(results)
    out_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ index.html 已生成")

    # 保存 JSON（历史记录）
    history_dir = os.path.join(os.path.dirname(__file__), "..", "history")
    os.makedirs(history_dir, exist_ok=True)
    history_data = []
    for r in results:
        history_data.append({
            "date":    TODAY_STR,
            "symbol":  r["strat"]["id"],
            "name":    r["strat"]["name"],
            "ind":     r.get("ind"),
            "state":   r["market"]["state"] if r.get("market") else None,
            "signal":  r["sig"]["signal"] if r.get("sig") else None,
            "window":  r["sig"]["window"] if r.get("sig") else None,
        })
    with open(os.path.join(history_dir, f"{TODAY_STR}.json"), "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    print(f"✓ history/{TODAY_STR}.json 已保存")

    # 持仓追踪
    print("\n[持仓追踪]")
    positions, closed = update_positions(results)
    closed_trades = load_closed_trades()

    # 飞书推送
    push_feishu(results, positions, closed, closed_trades)
    print("✓ 完成")


if __name__ == "__main__":
    main()
