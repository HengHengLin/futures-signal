"""
期货季节性策略回测
- 数据范围：2015年1月 → 上个月底（自动计算）
- 每月1号自动跑 / 随时手动触发
- 输出 backtest.html 到根目录
"""

import json
import os
import traceback
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta

import akshare as ak
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 时间范围：2015-01-01 → 上个月底
# ─────────────────────────────────────────────

CST       = timezone(timedelta(hours=8))
TODAY     = datetime.now(CST)
END_DATE  = (TODAY.replace(day=1) - timedelta(days=1))  # 上个月最后一天
START_DATE = datetime(2015, 1, 1, tzinfo=CST)

START_STR = START_DATE.strftime("%Y%m%d")
END_STR   = END_DATE.strftime("%Y%m%d")
END_LABEL = END_DATE.strftime("%Y年%m月")

START_YEAR = 2015
END_YEAR   = END_DATE.year  # 回测到去年底（完整年份）
# 当年也纳入，但只统计已完成的窗口
INCLUDE_CURRENT_YEAR = True

print(f"回测范围：{START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")

# ─────────────────────────────────────────────
# 品种配置
# ─────────────────────────────────────────────

STRATEGIES = [
    {
        "id": "RB",
        "name": "螺纹钢",
        "ak_symbol": "RB0",
        "direction": "long",
        "entry_month": 2,
        "entry_day": 10,        # 窗口中间日，找最近交易日
        "exit_month": 4,
        "exit_day": 15,
        "cross_year": False,
        "stated_winrate": 80,
        "logic": "春节后复工需求确定性回归",
    },
    {
        "id": "BU",
        "name": "沥青",
        "ak_symbol": "BU0",
        "direction": "short",
        "entry_month": 11,
        "entry_day": 5,
        "exit_month": 12,
        "exit_day": 25,
        "cross_year": False,
        "stated_winrate": 80,
        "logic": "北方入冬道路停工，需求物理冰封",
    },
    {
        "id": "I",
        "name": "铁矿石",
        "ak_symbol": "I0",
        "direction": "long",
        "entry_month": 11,
        "entry_day": 15,
        "exit_month": 1,
        "exit_day": 15,
        "cross_year": True,     # 跨年：11月入场，次年1月出场
        "stated_winrate": 82,
        "logic": "钢厂冬储补库，澳洲巴西雨季供应收缩",
    },
    {
        "id": "JD_short",
        "name": "鸡蛋(秋季空)",
        "ak_symbol": "JD0",
        "direction": "short",
        "entry_month": 9,
        "entry_day": 20,
        "exit_month": 10,
        "exit_day": 25,
        "cross_year": False,
        "stated_winrate": 85,
        "logic": "中秋国庆后需求坍塌，秋季产蛋率全年高位",
    },
    {
        "id": "JD_long",
        "name": "鸡蛋(夏季多)",
        "ak_symbol": "JD0",
        "direction": "long",
        "entry_month": 6,
        "entry_day": 25,
        "exit_month": 8,
        "exit_day": 15,
        "cross_year": False,
        "stated_winrate": 80,
        "logic": "高温减产+中秋月饼厂提前采购",
    },
    {
        "id": "JD_spring",
        "name": "鸡蛋(春季空)",
        "ak_symbol": "JD0",
        "direction": "short",
        "entry_month": 2,
        "entry_day": 25,
        "exit_month": 4,
        "exit_day": 25,
        "cross_year": False,
        "stated_winrate": 79,
        "logic": "春节后淡季，梅雨季贸易商被迫低价去库",
    },
    {
        "id": "M",
        "name": "豆粕",
        "ak_symbol": "M0",
        "direction": "long",
        "entry_month": 6,
        "entry_day": 25,
        "exit_month": 8,
        "exit_day": 15,
        "cross_year": False,
        "stated_winrate": 80,
        "logic": "美豆天气升水炒作+国内饲料旺季",
    },
]

# ─────────────────────────────────────────────
# 数据获取
# ─────────────────────────────────────────────

def fetch_data(ak_symbol: str) -> pd.DataFrame:
    try:
        df = ak.futures_main_sina(symbol=ak_symbol, start_date=START_STR, end_date=END_STR)
        df.columns = [c.lower() for c in df.columns]
        col_map = {}
        for c in df.columns:
            if any(k in c for k in ["date","日期","time"]):  col_map[c] = "date"
            elif any(k in c for k in ["open","开盘"]):        col_map[c] = "open"
            elif any(k in c for k in ["high","最高"]):        col_map[c] = "high"
            elif any(k in c for k in ["low","最低"]):         col_map[c] = "low"
            elif any(k in c for k in ["close","收盘"]):       col_map[c] = "close"
            elif any(k in c for k in ["vol","成交量"]):       col_map[c] = "volume"
        df = df.rename(columns=col_map)
        df["date"]  = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["high"]  = pd.to_numeric(df.get("high",  df["close"]), errors="coerce")
        df["low"]   = pd.to_numeric(df.get("low",   df["close"]), errors="coerce")
        df["volume"]= pd.to_numeric(df.get("volume", pd.Series([0]*len(df))), errors="coerce")
        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        print(f"  ✓ {ak_symbol}: {len(df)}条  {df['date'].min().date()} ~ {df['date'].max().date()}")
        return df
    except Exception as e:
        print(f"  ✗ {ak_symbol} 失败: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
# 找最近交易日
# ─────────────────────────────────────────────

def find_nearest_trading_day(df: pd.DataFrame, year: int, month: int, day: int) -> pd.Series | None:
    """找指定日期附近（±10天内）的第一个交易日"""
    target = pd.Timestamp(year=year, month=month, day=day)
    # 先找当天或之后
    mask = df["date"] >= target
    subset = df[mask]
    if len(subset) > 0 and (subset.iloc[0]["date"] - target).days <= 10:
        return subset.iloc[0]
    # 再找之前
    mask = df["date"] < target
    subset = df[mask]
    if len(subset) > 0 and (target - subset.iloc[-1]["date"]).days <= 10:
        return subset.iloc[-1]
    return None

# ─────────────────────────────────────────────
# 计算持仓期最大回撤
# ─────────────────────────────────────────────

def calc_max_drawdown(df: pd.DataFrame, entry_date, exit_date, direction: str) -> float:
    mask   = (df["date"] >= entry_date) & (df["date"] <= exit_date)
    prices = df[mask]["close"]
    if len(prices) < 2:
        return 0.0
    if direction == "long":
        peak = prices.cummax()
        dd   = ((prices - peak) / peak * 100).min()
    else:
        trough = prices.cummin()
        dd     = ((trough - prices) / trough * 100).min()
    return round(float(dd), 2)

# ─────────────────────────────────────────────
# 单品种回测
# ─────────────────────────────────────────────

def backtest_one(df: pd.DataFrame, strat: dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    records = []
    years   = range(START_YEAR, TODAY.year + 1)

    for year in years:
        entry_year = year
        exit_year  = year + 1 if strat["cross_year"] else year

        # 入场日
        try:
            entry_row = find_nearest_trading_day(df, entry_year, strat["entry_month"], strat["entry_day"])
        except Exception:
            continue
        if entry_row is None:
            continue

        # 出场日
        try:
            exit_row = find_nearest_trading_day(df, exit_year, strat["exit_month"], strat["exit_day"])
        except Exception:
            continue
        if exit_row is None:
            continue

        # 出场日必须晚于入场日，且不超过END_DATE
        if exit_row["date"] <= entry_row["date"]:
            continue
        if exit_row["date"] > pd.Timestamp(END_DATE):
            continue

        entry_price = float(entry_row["close"])
        exit_price  = float(exit_row["close"])

        if strat["direction"] == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        max_dd   = calc_max_drawdown(df, entry_row["date"], exit_row["date"], strat["direction"])
        hold_days = (exit_row["date"] - entry_row["date"]).days

        records.append({
            "year":         entry_year,
            "entry_date":   entry_row["date"].strftime("%Y-%m-%d"),
            "exit_date":    exit_row["date"].strftime("%Y-%m-%d"),
            "entry_price":  round(entry_price, 1),
            "exit_price":   round(exit_price, 1),
            "pnl_pct":      round(pnl_pct, 2),
            "win":          pnl_pct > 0,
            "max_drawdown": max_dd,
            "hold_days":    hold_days,
        })

    return pd.DataFrame(records)

# ─────────────────────────────────────────────
# 汇总统计
# ─────────────────────────────────────────────

def summarize(results: pd.DataFrame, strat: dict) -> dict:
    if results.empty:
        return {"name": strat["name"], "error": "无数据"}

    total    = len(results)
    wins     = int(results["win"].sum())
    winrate  = round(wins / total * 100, 1)
    avg_pnl  = round(results["pnl_pct"].mean(), 2)
    avg_win  = round(results[results["win"]]["pnl_pct"].mean(), 2) if wins > 0 else 0
    avg_loss = round(results[~results["win"]]["pnl_pct"].mean(), 2) if (total - wins) > 0 else 0
    max_dd   = round(results["max_drawdown"].min(), 2)
    avg_hold = int(results["hold_days"].mean())
    rr       = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 999
    # 最大连亏
    streak = max_consec_loss(results["win"].tolist())

    return {
        "name":           strat["name"],
        "direction":      strat["direction"],
        "total":          total,
        "wins":           wins,
        "winrate":        winrate,
        "stated_winrate": strat["stated_winrate"],
        "diff":           round(winrate - strat["stated_winrate"], 1),
        "avg_pnl":        avg_pnl,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "risk_reward":    rr,
        "max_drawdown":   max_dd,
        "avg_hold":       avg_hold,
        "max_loss_streak": streak,
        "logic":          strat["logic"],
    }

def max_consec_loss(wins: list) -> int:
    max_s = cur_s = 0
    for w in wins:
        if not w:
            cur_s += 1
            max_s = max(max_s, cur_s)
        else:
            cur_s = 0
    return max_s

# ─────────────────────────────────────────────
# HTML报告生成
# ─────────────────────────────────────────────

def build_html(all_summaries: list, all_details: list) -> str:
    now = TODAY.strftime("%Y-%m-%d %H:%M")

    # ── 汇总表 ──
    summary_rows = ""
    for s in all_summaries:
        if "error" in s:
            summary_rows += f"<tr><td>{s['name']}</td><td colspan='10' style='color:#4a5a6a'>数据获取失败</td></tr>"
            continue
        d_color  = "#00e5a0" if s["direction"] == "long" else "#ff4d6a"
        d_text   = "做多↑" if s["direction"] == "long" else "做空↓"
        wr_color = "#00e5a0" if s["winrate"] >= 75 else "#f5c842"
        diff_color = "#00e5a0" if s["diff"] >= 0 else "#ff4d6a"
        diff_str = f"+{s['diff']}%" if s["diff"] >= 0 else f"{s['diff']}%"
        pnl_color = "#00e5a0" if s["avg_pnl"] > 0 else "#ff4d6a"

        summary_rows += f"""
        <tr>
          <td><b>{s['name']}</b></td>
          <td style="color:{d_color}">{d_text}</td>
          <td>{s['total']}</td>
          <td style="color:{wr_color};font-weight:700">{s['winrate']}%</td>
          <td>{s['stated_winrate']}% <span style="color:{diff_color};font-size:11px">({diff_str})</span></td>
          <td style="color:{pnl_color}">{s['avg_pnl']}%</td>
          <td style="color:#00e5a0">{s['avg_win']}%</td>
          <td style="color:#ff4d6a">{s['avg_loss']}%</td>
          <td>{s['risk_reward']}</td>
          <td style="color:#ff4d6a">{s['max_drawdown']}%</td>
          <td>{s['avg_hold']}天</td>
          <td>{s['max_loss_streak']}</td>
        </tr>"""

    # ── 逐年明细 ──
    detail_html = ""
    for strat, df in zip(STRATEGIES, all_details):
        if df is None or df.empty:
            continue
        rows = ""
        for _, r in df.iterrows():
            pnl_color = "#00e5a0" if r["pnl_pct"] > 0 else "#ff4d6a"
            win_icon  = "✓" if r["win"] else "✗"
            win_color = "#00e5a0" if r["win"] else "#ff4d6a"
            rows += f"""
            <tr>
              <td>{r['year']}</td>
              <td>{r['entry_date']}</td>
              <td>{r['exit_date']}</td>
              <td>{r['entry_price']}</td>
              <td>{r['exit_price']}</td>
              <td style="color:{pnl_color};font-weight:600">{r['pnl_pct']:+.2f}%</td>
              <td style="color:{win_color};font-weight:700">{win_icon}</td>
              <td style="color:#ff4d6a">{r['max_drawdown']}%</td>
              <td>{r['hold_days']}天</td>
            </tr>"""

        s = next(x for x in all_summaries if x["name"] == strat["name"])
        wr_color = "#00e5a0" if s.get("winrate", 0) >= 75 else "#f5c842"

        detail_html += f"""
        <div class="detail-block">
          <div class="detail-header">
            <span class="detail-name">{strat['name']}</span>
            <span class="detail-dir {'long' if strat['direction']=='long' else 'short'}">
              {'做多↑' if strat['direction']=='long' else '做空↓'}
            </span>
            <span class="detail-winrate" style="color:{wr_color}">
              实测胜率 {s.get('winrate','—')}%
            </span>
            <span class="detail-logic">{strat['logic']}</span>
          </div>
          <table>
            <thead><tr>
              <th>年份</th><th>建仓日</th><th>平仓日</th>
              <th>建仓价</th><th>平仓价</th>
              <th>收益%</th><th>盈亏</th>
              <th>最大回撤</th><th>持仓天数</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>期货季节性回测报告</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
:root{{
  --bg:#0a0c0f;--surface:#111418;--border:#1e2530;
  --text:#c8d4e0;--bright:#e8f0f8;--dim:#4a5a6a;
  --green:#00e5a0;--red:#ff4d6a;--yellow:#f5c842;--blue:#4da6ff;
  --mono:'JetBrains Mono',monospace;--sans:'Noto Sans SC',sans-serif;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;padding:32px 40px;max-width:1100px;margin:0 auto}}
body::before{{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(30,37,48,.3) 1px,transparent 1px),linear-gradient(90deg,rgba(30,37,48,.3) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}}
.wrap{{position:relative;z-index:1}}
.header{{margin-bottom:28px;padding-bottom:16px;border-bottom:1px solid var(--border)}}
h1{{font-family:var(--mono);font-size:16px;font-weight:700;color:var(--bright);letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}}
.meta{{font-family:var(--mono);font-size:11px;color:var(--dim)}}
.meta span{{color:var(--yellow);margin:0 8px}}
h2{{font-family:var(--mono);font-size:11px;font-weight:600;color:var(--dim);letter-spacing:.12em;text-transform:uppercase;margin:28px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.nav-link{{font-family:var(--mono);font-size:11px;color:var(--blue);text-decoration:none;margin-right:16px}}
.nav-link:hover{{color:var(--bright)}}
table{{width:100%;border-collapse:collapse;margin-bottom:8px;font-size:12px}}
th{{font-family:var(--mono);font-size:10px;color:var(--dim);text-align:left;padding:8px 10px;border-bottom:1px solid var(--border);letter-spacing:.05em;text-transform:uppercase;white-space:nowrap}}
td{{padding:9px 10px;border-bottom:1px solid rgba(30,37,48,.8);font-family:var(--mono);color:var(--text);white-space:nowrap}}
tr:hover td{{background:rgba(255,255,255,.02)}}
.detail-block{{margin-bottom:36px;background:var(--surface);border:1px solid var(--border);border-radius:6px;overflow:hidden}}
.detail-header{{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.detail-name{{font-family:var(--mono);font-size:13px;font-weight:700;color:var(--bright)}}
.detail-dir{{font-family:var(--mono);font-size:10px;font-weight:700;padding:2px 8px;border-radius:3px}}
.detail-dir.long{{background:rgba(0,229,160,.1);color:var(--green);border:1px solid var(--green)}}
.detail-dir.short{{background:rgba(255,77,106,.1);color:var(--red);border:1px solid var(--red)}}
.detail-winrate{{font-family:var(--mono);font-size:12px;font-weight:600}}
.detail-logic{{font-size:11px;color:var(--dim)}}
.detail-block table{{margin:0}}
.summary-table{{background:var(--surface);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-bottom:8px}}
.footer{{margin-top:32px;padding-top:14px;border-top:1px solid var(--border);font-family:var(--mono);font-size:10px;color:var(--dim);display:flex;justify-content:space-between}}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>期货季节性策略 — 回测报告</h1>
    <div class="meta">
      数据范围：2015-01 → {END_LABEL}
      <span>·</span>
      生成时间：{now}
      <span>·</span>
      共{len([s for s in all_summaries if 'error' not in s])}个品种
    </div>
    <div style="margin-top:10px">
      <a class="nav-link" href="index.html">← 返回每日信号</a>
    </div>
  </div>

  <h2>汇总对比</h2>
  <div class="summary-table">
    <table>
      <thead><tr>
        <th>品种</th><th>方向</th><th>笔数</th>
        <th>实测胜率</th><th>表格胜率</th>
        <th>平均收益</th><th>均盈</th><th>均亏</th>
        <th>盈亏比</th><th>最大回撤</th><th>均持仓</th><th>最大连亏</th>
      </tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>
  </div>

  <h2>逐年明细</h2>
  {detail_html}

  <div class="footer">
    <span>收益率基于收盘价点位变动 · 未含保证金杠杆 · 仅供参考</span>
    <span>自动更新 via GitHub Actions · 每月1日</span>
  </div>
</div>
</body>
</html>"""

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*50}")
    print(f"  期货季节性回测")
    print(f"  {START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")
    print(f"{'='*50}\n")

    # 缓存已拉取的数据（同一ak_symbol不重复拉）
    data_cache = {}
    all_summaries = []
    all_details   = []

    for strat in STRATEGIES:
        print(f"[{strat['name']}]")
        sym = strat["ak_symbol"]
        if sym not in data_cache:
            data_cache[sym] = fetch_data(sym)
        df = data_cache[sym]

        try:
            detail  = backtest_one(df, strat)
            summary = summarize(detail, strat)
            if "error" not in summary:
                print(f"  实测胜率: {summary['winrate']}%  "
                      f"(表格: {strat['stated_winrate']}%  diff: {summary['diff']:+.1f}%)")
                print(f"  平均收益: {summary['avg_pnl']}%  "
                      f"盈亏比: {summary['risk_reward']}  "
                      f"最大回撤: {summary['max_drawdown']}%")
        except Exception:
            detail  = pd.DataFrame()
            summary = {"name": strat["name"], "error": "计算失败"}
            traceback.print_exc()

        all_summaries.append(summary)
        all_details.append(detail)
        print()

    # 生成HTML
    html     = build_html(all_summaries, all_details)
    out_path = os.path.join(os.path.dirname(__file__), "..", "backtest.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ backtest.html 已生成")

    # 保存JSON（供其他脚本读取）
    summary_path = os.path.join(os.path.dirname(__file__), "..", "backtest_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"✓ backtest_summary.json 已保存")
    print("✓ 完成")


if __name__ == "__main__":
    main()
