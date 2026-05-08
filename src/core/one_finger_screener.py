# -*- coding: utf-8 -*-
"""
一阳指战法 — 全A股票筛选引擎（经典版）

核心定义：
  在长期下跌或充分盘整后，出现一根放量、实体长的大阳线，
  有效突破前期整理平台，标志着主升浪的开始。

三段式筛选：
  Phase 1: 初筛 — 全A涨幅>5% + 大阳线形态 + 放量
  Phase 2: 精选 — 识别盘整区 + 验证前期趋势 + K线质量
  Phase 3: 输出 — 生成选股报告 + 买卖点参考

经典参数：
  MIN_PRICE_INCREASE = 5%   (最小涨幅)
  VOLUME_MULTIPLIER  = 2.0x (量能倍数是前5日均量)
  CONSOLIDATION_PERIOD = 10 (盘整期交易日数)
  MAX_CONSOLIDATION_RANGE = 10% (最大盘整振幅)
  PRIOR_TREND = DOWNTREND | CONSOLIDATION (排除高位)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Global Parameters (Classic Definition)
# ═══════════════════════════════════════════════════════════════

MIN_PRICE_INCREASE = 0.05       # 大阳线最小涨幅 5%
VOLUME_MULTIPLIER = 2.0         # 量能倍数：当日量 > 前5日均量 × 2.0
CONSOLIDATION_PERIOD = 10       # 盘整期长度（交易日）
MAX_CONSOLIDATION_RANGE = 0.10  # 盘整区最大振幅 10%
KLINE_BODY_RATIO = 0.70         # 实体占总振幅比例 ≥ 70%
PRIOR_TREND_LOOKBACK = 20       # 前期趋势回看天数
KLINE_FETCH_DAYS = 80           # K线拉取天数（需覆盖MA60+盘整+前期趋势）
MIN_TURNOVER = 2.0              # 初筛最低换手率
MAX_TURNOVER = 20.0             # 初筛最高换手率
A_GRADE_VOL_RATIO = 3.5         # A级量能倍数
A_GRADE_BODY_RATIO = 0.85       # A级实体占比
B_GRADE_VOL_RATIO = 2.5         # B级量能倍数
B_GRADE_BODY_RATIO = 0.75       # B级实体占比
MAX_LOWER_SHADOW = 0.20         # 下影线占比上限
MAX_UPPER_SHADOW = 0.25         # 上影线占比上限
PINCH_BODY_RATIO_MAX = 0.50     # 前日收窄：实体占比上限
PINCH_RANGE_MAX = 0.03          # 前日收窄：振幅上限
MA_STICKY_MAX = 0.03            # 均线粘合：最大离散度


# ═══════════════════════════════════════════════════════════════
# MarketGuard — 大盘安检
# ═══════════════════════════════════════════════════════════════

def run_one_finger_guard(
    ss_index: Optional[pd.DataFrame],
    market_stats: Dict,
    emotion_cycle: str,
) -> Dict:
    """
    一阳指大盘安检（参考飞龙在天 MarketGuard）

    Args:
        ss_index: 上证指数日线DataFrame（含 close, volume）
        market_stats: {'up_count','down_count','limit_down_count','total_amount'}
        emotion_cycle: 情绪周期标签

    Returns:
        {'passed': bool, 'score': int, 'warnings': [...], 'position_advice': str,
         'sse_close': float, 'up_ratio': str, 'limit_down': int}
    """
    score = 100
    warnings = []

    up_count = market_stats.get('up_count', 0)
    down_count = market_stats.get('down_count', 1)
    limit_down = market_stats.get('limit_down_count', 0)
    up_ratio = up_count / max(up_count + down_count, 1)
    sse_close = 0.0

    # ① 沪指 vs MA20
    if ss_index is not None and not ss_index.empty and len(ss_index) >= 22:
        sse_close = float(ss_index['close'].iloc[-1])
        ma20 = float(ss_index['close'].tail(20).mean())
        if sse_close < ma20 * 0.98:
            warnings.append(f"沪指{sse_close:.0f}低于MA20({ma20:.0f})，偏弱")
            score -= 25
        elif sse_close < ma20:
            warnings.append(f"沪指贴近MA20，方向不明")
            score -= 10

    # ② 涨跌比
    if up_ratio < 0.4:
        warnings.append(f"涨跌比仅{up_ratio:.0%}，情绪冰点")
        score -= 20
    elif up_ratio < 0.5:
        warnings.append(f"涨跌比{up_ratio:.0%}，偏弱")
        score -= 10

    # ③ 跌停数
    if limit_down >= 100:
        warnings.append(f"跌停{limit_down}家，恐慌踩踏")
        score -= 30
    elif limit_down >= 60:
        warnings.append(f"跌停{limit_down}家偏高，分歧加剧")
        score -= 15
    elif limit_down >= 40:
        warnings.append(f"跌停{limit_down}家，需警惕")
        score -= 5

    # ④ 情绪周期
    if emotion_cycle in ('恐慌', '退潮'):
        warnings.append(f"情绪{emotion_cycle}，追高风险极大")
        score -= 20

    # 仓位建议
    if score >= 80:
        position_advice = "3-5成"
    elif score >= 60:
        position_advice = "2-3成"
    elif score >= 40:
        position_advice = "1-2成或空仓"
    else:
        position_advice = "空仓观望"

    return {
        'passed': score >= 40,
        'score': score,
        'warnings': warnings[:3],
        'position_advice': position_advice,
        'sse_close': sse_close,
        'up_ratio': f"{up_ratio:.0%}",
        'limit_down': limit_down,
    }


# ═══════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class OneFingerStock:
    """一阳指入选股票（经典版）"""
    code: str
    name: str
    close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    change_pct: float = 0.0       # 今日涨幅
    body_ratio: float = 0.0       # 实体/总振幅
    vol_ratio_5d: float = 0.0     # 量/前5日均量
    turn_over: float = 0.0        # 换手率
    industry: str = ""
    is_limit_up: bool = False     # 是否涨停
    # 盘整区信息
    consolidation_high: float = 0.0
    consolidation_low: float = 0.0
    consolidation_range: float = 0.0  # 盘整区振幅
    vol_contracting: bool = False     # 盘整区后段缩量
    # 前期趋势
    prior_trend: str = ""          # "DOWNTREND" | "CONSOLIDATION" | "UPTREND(排除)"
    # 一阳K线增强指标
    lower_shadow_ratio: float = 0.0   # 下影线/总振幅
    upper_shadow_ratio: float = 0.0   # 上影线/总振幅
    ma_penetrations: int = 0          # 穿透均线条数
    prev_day_pinching: bool = False   # 前日收窄
    ma_sticky: bool = False           # 均线粘合
    sector_in_top: bool = False       # 板块共振
    guard_score: int = 100            # 大盘安检分数
    # 买卖参考
    entry_zone: str = ""
    stop_loss: str = ""
    # 质量评级
    quality: str = ""              # "A" | "B" | "C"
    quality_reasons: List[str] = field(default_factory=list)
    # 条件检查清单
    conditions: Dict[str, bool] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# K-Line Fetching
# ═══════════════════════════════════════════════════════════════

def _fetch_single_kline(code: str, days: int = KLINE_FETCH_DAYS) -> Optional[pd.DataFrame]:
    """拉取单只股票的日K线（yfinance→东财→腾讯→baostock），返回标准化的DataFrame"""
    try:
        import akshare as ak
        from datetime import date
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=days + 30)
        sd = start_dt.strftime('%Y%m%d')
        ed = end_dt.strftime('%Y%m%d')

        df = None
        # yfinance 优先（全球CDN，GH Actions可用）→ 东财 → 腾讯 → baostock
        for src in ['yf', 'em', 'tx', 'bs']:
            try:
                if src == 'yf':
                    import yfinance as yf
                    sym = f'{code}.SS' if code.startswith('6') else f'{code}.SZ'
                    ticker = yf.Ticker(sym)
                    df = ticker.history(period=f'{days+30}d')
                    if df is not None and not df.empty:
                        df = df.reset_index()
                        df = df.rename(columns={
                            'Date': 'date', 'Open': 'open', 'Close': 'close',
                            'High': 'high', 'Low': 'low', 'Volume': 'volume',
                        })
                        df['amount'] = 0
                        df['amplitude'] = 0.0
                        df['change_pct'] = df['close'].pct_change() * 100
                        df['turnover'] = 0.0
                elif src == 'em':
                    df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=sd, end_date=ed, adjust='qfq')
                elif src == 'tx':
                    df = ak.stock_zh_a_hist_tx(symbol=code, start_date=sd, end_date=ed, adjust='qfq')
                else:
                    # baostock — 重置连接避免状态问题
                    import baostock as bs
                    bs_code = f'sh.{code}' if code.startswith('6') else f'sz.{code}'
                    bs.login()
                    rs = bs.query_history_k_data_plus(bs_code,
                        'date,open,close,high,low,volume,amount,amplitude,pctChg,turn',
                        start_date=start_dt.strftime('%Y-%m-%d'), end_date=end_dt.strftime('%Y-%m-%d'),
                        frequency='d', adjustflag='2')
                    if rs.error_code == '0':
                        rows = []
                        while rs.next():
                            rows.append(rs.get_row_data())
                        if rows:
                            df = pd.DataFrame(rows, columns=['date','open','close','high','low','volume','amount','amplitude','change_pct','turnover'])
                            for col in ['open','close','high','low','volume','amount','amplitude','change_pct','turnover']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            df = df.dropna(subset=['close'])
                    bs.logout()
                if df is not None and not df.empty:
                    break
            except Exception:
                continue

        if df is None or df.empty:
            return None
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume',
            '成交额': 'amount', '振幅': 'amplitude',
            '涨跌幅': 'change_pct', '换手率': 'turnover',
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').tail(days).copy()
        return df
    except Exception:
        return None


def _batch_fetch_klines(codes: List[str], max_workers: int = 20, days: int = 60) -> Dict[str, pd.DataFrame]:
    """并行拉取多只股票的日K线"""
    results = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_single_kline, code, days): code for code in codes}
        for f in as_completed(futures):
            code = futures[f]
            try:
                df = f.result(timeout=30)
                if df is not None and not df.empty:
                    results[code] = df
                else:
                    failed += 1
            except Exception:
                failed += 1
    if failed > 0:
        logger.warning(f"K线拉取: {len(results)}成功 / {failed}失败")
    return results


# ═══════════════════════════════════════════════════════════════
# Function 1: 6/21均线系统检查（阶段二精选）
# ═══════════════════════════════════════════════════════════════

def _calc_ma(closes: pd.Series, days: int) -> Optional[float]:
    """计算移动平均线"""
    if len(closes) < days:
        return None
    return float(closes.tail(days).mean())


def _is_ma_flat_or_up(closes: pd.Series, days: int) -> bool:
    """判断均线是否走平或向上（最近4个MA值不下降超2%）"""
    n = len(closes)
    if n < days + 4:
        return False
    ma_vals = []
    for i in range(4):
        seg = closes.iloc[n - days - i - 1:n - i]
        if len(seg) < days:
            return False
        ma_vals.append(float(seg.mean()))
    return ma_vals[0] >= ma_vals[3] * 0.98


def check_ma_system(df: pd.DataFrame) -> Tuple[bool, float, float, List[str]]:
    """
    阶段二 · 6/21均线系统检查

    条件:
      ① 股价上穿/站稳6日均线 (今日收盘 > MA6)
      ② 股价上穿/站稳21日均线 (今日收盘 > MA21)
      ③ MA6走平或向上
      ④ MA21走平或向上
      ⑤ 昨日收盘在MA6或MA21下方附近 → "上穿"信号确认

    Returns: (passed, ma6, ma21, check_details)
    """
    closes = df['close']
    ma6 = _calc_ma(closes, 6)
    ma21 = _calc_ma(closes, 21)

    if ma6 is None or ma21 is None:
        return False, 0, 0, ["MA数据不足"]

    close_today = float(closes.iloc[-1])
    details = []

    # ① 站上MA6
    c1 = close_today > ma6
    details.append(f"{'✅' if c1 else '❌'}站上MA6({ma6:.2f})")
    if not c1:
        return False, ma6, ma21, details

    # ② 站上MA21
    c2 = close_today > ma21
    details.append(f"{'✅' if c2 else '❌'}站上MA21({ma21:.2f})")
    if not c2:
        return False, ma6, ma21, details

    # ③ MA6走平/向上
    c3 = _is_ma_flat_or_up(closes, 6)
    details.append(f"{'✅' if c3 else '❌'}MA6走平/向上")
    if not c3:
        return False, ma6, ma21, details

    # ④ MA21走平/向上
    c4 = _is_ma_flat_or_up(closes, 21)
    details.append(f"{'✅' if c4 else '❌'}MA21走平/向上")
    if not c4:
        return False, ma6, ma21, details

    # ⑤ 上穿确认（昨收在MA6或MA21下方）
    if len(closes) >= 2:
        prev_close = float(closes.iloc[-2])
        cross_6 = prev_close <= ma6 < close_today
        cross_21 = prev_close <= ma21 < close_today
        if cross_6 or cross_21:
            details.append("✅上穿信号")
        else:
            details.append("⚠️非首日上穿(已在线上)")
    else:
        details.append("⚠️无法判断上穿")

    return True, ma6, ma21, details


# ═══════════════════════════════════════════════════════════════
# Function 2: 识别盘整区（阶段二精选）
# ═══════════════════════════════════════════════════════════════

def identify_consolidation_zone(df: pd.DataFrame) -> Tuple[bool, float, float, float]:
    """
    识别过去 N 个交易日是否形成有效盘整区。

    输入: df (日K线DataFrame)
    输出: (is_consolidating, high, low, total_range)
      - is_consolidating: 是否在盘整
      - high: 盘整区最高价
      - low: 盘整区最低价
      - total_range: 总振幅

    逻辑:
      取过去 CONSOLIDATION_PERIOD 天（不含今天）的最高价和最低价
      计算总振幅 = (max_high - min_low) / min_low
      如果总振幅 <= MAX_CONSOLIDATION_RANGE，则判定为盘整
    """
    if len(df) < CONSOLIDATION_PERIOD + 1:
        return False, 0, 0, 0

    # 取过去N天（不含今天），因为今天可能是突破日
    lookback = df.iloc[-(CONSOLIDATION_PERIOD + 1):-1]
    if len(lookback) < CONSOLIDATION_PERIOD:
        return False, 0, 0, 0

    max_high = float(lookback['high'].max())
    min_low = float(lookback['low'].min())

    if min_low <= 0:
        return False, 0, 0, 0, False

    total_range = (max_high - min_low) / min_low

    # 振幅 <= 10% 判定为盘整
    is_consolidating = total_range <= MAX_CONSOLIDATION_RANGE

    # 量能收缩：盘整区后5天均量 < 前5天均量（供给被吸收）
    vol_contracting = False
    if is_consolidating and len(lookback) >= 10:
        front_vol = float(lookback['volume'].iloc[:5].mean())
        back_vol = float(lookback['volume'].iloc[5:].mean())
        vol_contracting = back_vol < front_vol if front_vol > 0 else False

    return is_consolidating, max_high, min_low, total_range, vol_contracting


# ═══════════════════════════════════════════════════════════════
# Function 2: 验证前期趋势
# ═══════════════════════════════════════════════════════════════

def check_prior_trend_condition(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    验证前期趋势：必须是在下跌或盘整后出现，排除高位假突破。

    逻辑:
      取过去 PRIOR_TREND_LOOKBACK 天的最高价和收盘价变化
      如果是上升趋势（涨幅>15%），返回 False（高位盘整，排除）
      如果是下跌或横盘，返回 True

    Returns: (pass, trend_label)
    """
    if len(df) < PRIOR_TREND_LOOKBACK + CONSOLIDATION_PERIOD + 2:
        return True, "CONSOLIDATION"  # 数据不足，默认通过

    # 看 CONSOLIDATION_PERIOD 之前的那段趋势
    # 取 day[-20-10:-10] 这段作为前期趋势
    prior_start = -(PRIOR_TREND_LOOKBACK + CONSOLIDATION_PERIOD + 1)
    prior_end = -(CONSOLIDATION_PERIOD + 1)
    prior_seg = df.iloc[prior_start:prior_end]

    if len(prior_seg) < 10:
        return True, "CONSOLIDATION"

    prior_start_price = float(prior_seg['close'].iloc[0])
    prior_end_price = float(prior_seg['close'].iloc[-1])

    if prior_start_price <= 0:
        return True, "CONSOLIDATION"

    period_change = (prior_end_price - prior_start_price) / prior_start_price

    if period_change > 0.15:
        # 前期涨幅>15%，高位风险
        return False, "UPTREND(高位排除)"
    elif period_change < -0.10:
        return True, "DOWNTREND"
    else:
        return True, "CONSOLIDATION"


# ═══════════════════════════════════════════════════════════════
# Function 3: 验证一阳K线
# ═══════════════════════════════════════════════════════════════

def validate_one_yang_kline(df: pd.DataFrame, consolidation_high: float) -> Tuple[bool, Dict]:
    """
    验证最新K线是否构成合格的一阳指。

    条件A: (C-O)/O >= 5%         — 大阳线涨幅
    条件B: (C-O)/(H-L) >= 0.70   — 实体占总振幅≥70%
    条件C: Vol >= 前5日均量 × 2.0 — 放巨量
    条件D: C > consolidation_high — 突破盘整区

    Returns: (passed, metrics_dict)
    """
    if len(df) < 6:
        return False, {}

    latest = df.iloc[-1]
    O = float(latest['open'])
    C = float(latest['close'])
    H = float(latest['high'])
    L = float(latest['low'])
    Vol = float(latest['volume'])

    if O <= 0 or H <= L:
        return False, {}

    # 条件A: 大阳线涨幅 ≥ 5%
    day_change = (C - O) / O
    cond_a = day_change >= MIN_PRICE_INCREASE

    # 条件B: 实体占比 ≥ 70%
    body = C - O
    total_range = H - L
    body_ratio = body / total_range if total_range > 0 else 0
    cond_b = body_ratio >= KLINE_BODY_RATIO

    # 条件C: 放巨量（>前5日均量 × 2.0）
    prev_5_vols = df['volume'].iloc[-6:-1]
    avg_vol_5d = float(prev_5_vols.mean()) if len(prev_5_vols) >= 3 else Vol
    vol_ratio = Vol / avg_vol_5d if avg_vol_5d > 0 else 0
    cond_c = vol_ratio >= VOLUME_MULTIPLIER

    # 条件D: 突破盘整区高点
    cond_d = C > consolidation_high if consolidation_high > 0 else cond_a and cond_c

    metrics = {
        'day_change': day_change,
        'body_ratio': body_ratio,
        'vol_ratio_5d': vol_ratio,
        'breakout': C - consolidation_high if consolidation_high > 0 else 0,
        'cond_a': cond_a,
        'cond_b': cond_b,
        'cond_c': cond_c,
        'cond_d': cond_d,
    }

    passed = cond_a and cond_b and cond_c and cond_d
    return passed, metrics


# ═══════════════════════════════════════════════════════════════
# Enhanced Analysis Functions (Optimizations)
# ═══════════════════════════════════════════════════════════════

def _count_ma_penetrations(closes: pd.Series, O: float, C: float) -> int:
    """统计大阳线实体穿透了多少条关键均线（MA6/MA10/MA21/MA60）。"""
    mas = {'MA6': 6, 'MA10': 10, 'MA21': 21, 'MA60': 60}
    penetrated = 0
    for ma_name, period in mas.items():
        ma_val = _calc_ma(closes, period)
        if ma_val is not None and O < ma_val < C:
            penetrated += 1
    return penetrated


def _check_prev_day_pinching(df: pd.DataFrame) -> bool:
    """检查前日K线是否极致收敛（盘整末端信号）。"""
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    prev_O = float(prev['open'])
    prev_C = float(prev['close'])
    prev_H = float(prev['high'])
    prev_L = float(prev['low'])
    if prev_H <= prev_L or prev_L <= 0:
        return False
    prev_range = (prev_H - prev_L) / prev_L
    prev_body = abs(prev_C - prev_O) / (prev_H - prev_L)
    return prev_range < PINCH_RANGE_MAX and prev_body < PINCH_BODY_RATIO_MAX


def _calc_shadow_ratios(O: float, C: float, H: float, L: float) -> Tuple[float, float]:
    """计算上下影线占比。返回 (lower_shadow_ratio, upper_shadow_ratio)。"""
    total_range = H - L
    if total_range <= 0:
        return 1.0, 1.0
    lower_shadow = (min(O, C) - L) / total_range
    upper_shadow = (H - max(O, C)) / total_range
    return lower_shadow, upper_shadow


def _calc_ma_stickiness(closes: pd.Series, close_today: float) -> Tuple[bool, float]:
    """计算MA5/MA10/MA21的粘合度。返回 (is_sticky, spread_pct)。"""
    ma5 = _calc_ma(closes, 5)
    ma10 = _calc_ma(closes, 10)
    ma21_val = _calc_ma(closes, 21)
    if ma5 is None or ma10 is None or ma21_val is None or close_today <= 0:
        return False, 1.0
    ma_values = [ma5, ma10, ma21_val]
    spread = (max(ma_values) - min(ma_values)) / close_today
    return spread < MA_STICKY_MAX, spread


# ═══════════════════════════════════════════════════════════════
# Main Screener
# ═══════════════════════════════════════════════════════════════

def screen_one_finger_stocks(
    limit_up_pool: Optional[List[Dict]] = None,
    top_sectors: Optional[List[Dict]] = None,
    bottom_sectors: Optional[List[Dict]] = None,
    guard_score: int = 100,
    max_candidates: int = 200,
) -> List[OneFingerStock]:
    """
    一阳指战法全A选股（经典版）

    三段式流程：
      STEP 1: 初筛 — 全A涨幅>5% + 非ST
      STEP 2: 精选 — 盘整区识别 + 前期趋势 + K线质量
      STEP 3: 输出 — 质量评级 + 买卖参考

    Returns:
        符合经典一阳指条件的股票列表
    """
    logger.info("========== 一阳指战法选股（经典版）==========")

    # ── STEP 1: 初筛 ──
    import akshare as ak

    # 先试东财（字段全），再试新浪，最后用 baostock+腾讯行情兜底
    snapshot = None
    src_used = ''
    for src_name, fetch_fn in [
        ('东财', lambda: ak.stock_zh_a_spot_em()),
        ('新浪', lambda: ak.stock_zh_a_spot()),
    ]:
        try:
            snapshot = fetch_fn()
            if snapshot is not None and not snapshot.empty:
                src_used = src_name
                logger.info(f"全A快照({src_name}): {len(snapshot)} 只")
                break
        except Exception as e:
            logger.warning(f"全A快照({src_name})失败: {e}")

    # 第三兜底: baostock + 腾讯行情
    if snapshot is None or snapshot.empty:
        try:
            import baostock as bs
            import requests as _req
            bs.login()
            rs = bs.query_stock_basic()
            all_codes = []
            while rs.next():
                row = rs.get_row_data()
                code_full = row[0]  # e.g. "sh.603095"
                name = row[1]
                if 'ST' in name or '指数' in name or code_full.startswith('bj.'):
                    continue
                code = code_full.split('.')[-1]
                all_codes.append((code, name))
            bs.logout()
            logger.info(f"baostock 股票列表: {len(all_codes)} 只")

            # 腾讯行情批量拉取（每批80只）
            import pandas as pd
            rows = []
            batch_size = 80
            for i in range(0, len(all_codes), batch_size):
                batch = all_codes[i:i+batch_size]
                tx_parts = [f"{'sh' if c[0]=='6' else 'sz'}{c[0]}" for c in batch]
                url = f"http://qt.gtimg.cn/q={','.join(tx_parts)}"
                resp = _req.get(url, timeout=15)
                resp.encoding = 'gbk'
                for line in resp.text.strip().split('\n'):
                    if '~' not in line:
                        continue
                    parts = line.split('~')
                    if len(parts) < 35:
                        continue
                    tc = parts[2]  # pure numeric code
                    price = float(parts[3]) if parts[3] else 0
                    prev_close = float(parts[4]) if parts[4] else 0
                    if price > 0 and prev_close > 0:
                        change_pct = (price - prev_close) / prev_close * 100
                        turn_over = float(parts[37]) if len(parts) > 37 and parts[37] else 0
                        industry = parts[56].strip() if len(parts) > 56 and parts[56] and parts[56].strip() not in ('GP-A', 'GP') else ''
                        rows.append({
                            '代码': tc, '名称': parts[1], '涨跌幅': change_pct,
                            '量比': 1.0, '换手率': turn_over, '所属行业': industry,
                        })
                if i % (batch_size * 5) == 0:
                    logger.info(f"  腾讯行情: {i}/{len(all_codes)}")
            snapshot = pd.DataFrame(rows)
            src_used = '腾讯行情'
            logger.info(f"全A快照({src_used}): {len(snapshot)} 只")
        except Exception as e:
            logger.error(f"全A快照(baostock+腾讯)失败: {e}")

    if snapshot is None or snapshot.empty:
        logger.error("全A快照获取失败（所有数据源均不可用）")
        return []

    # 涨停池查找表
    limit_up_map = {}
    if limit_up_pool:
        for s in limit_up_pool:
            code = s.get('code', '')
            limit_up_map[code] = {
                'consecutive': s.get('consecutive', 1),
                'name': s.get('name', ''),
            }

    # 初筛：涨幅>5% + 非ST + 换手率(有数据时)（兼容东财/新浪两种列名）
    candidates = []
    turnover_available = False  # 检测是否有换手率数据
    for _, row in snapshot.iterrows():
        code = str(row.get('代码', '')).strip()
        name = str(row.get('名称', '')).strip()
        change_pct = float(row.get('涨跌幅', 0) or 0)
        volume_ratio = float(row.get('量比', 1.0) or 1.0)
        turn_over = float(row.get('换手率', 0) or 0)
        industry = str(row.get('所属行业', '') or '')

        if turn_over > 0:
            turnover_available = True

        # 新浪代码带前缀（bj920000/sh603095/sz002081），去掉前2位
        if code[:2] in ('bj', 'sh', 'sz', 'BJ', 'SH', 'SZ'):
            code = code[2:]

        if 'ST' in name or '*ST' in name or 'st' in name.lower():
            continue
        if len(code) < 6:
            continue
        if change_pct < MIN_PRICE_INCREASE * 100:
            continue

        candidates.append({
            'code': code, 'name': name, 'change_pct': change_pct,
            'volume_ratio': volume_ratio, 'turn_over': turn_over,
            'industry': industry,
            'is_limit_up': code in limit_up_map,
            'consecutive_limit': limit_up_map.get(code, {}).get('consecutive', 0),
        })

    # 换手率过滤（仅当数据源提供换手率时生效）
    if turnover_available:
        candidates = [c for c in candidates if MIN_TURNOVER <= c['turn_over'] <= MAX_TURNOVER]
        logger.info(f"STEP1 初筛(含换手率{MIN_TURNOVER}-{MAX_TURNOVER}%): {len(candidates)} 只")
    else:
        logger.info(f"STEP1 初筛(换手率数据不可用): {len(candidates)} 只")

    if not candidates:
        logger.info("无候选股")
        return []

    # 按涨幅排序，限制候选数量
    candidates.sort(key=lambda x: x['change_pct'], reverse=True)
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
        logger.info(f"截取前{max_candidates}只候选股")

    # ── 构建板块共振查找表 ──
    sector_top_set: set = set()
    if top_sectors:
        for s in top_sectors:
            name = s[0] if isinstance(s, (tuple, list)) else (s.get('name', '') if isinstance(s, dict) else str(s))
            sector_top_set.add(name)

    # ── STEP 2: 精选（多维校验）──
    codes = [c['code'] for c in candidates]
    logger.info(f"并行拉取 {len(codes)} 只K线...")
    kline_map = _batch_fetch_klines(codes, max_workers=12, days=KLINE_FETCH_DAYS)
    logger.info(f"成功拉取 {len(kline_map)} 只K线")

    results: List[OneFingerStock] = []
    for c in candidates:
        code = c['code']
        df = kline_map.get(code)
        if df is None or df.empty or len(df) < CONSOLIDATION_PERIOD + PRIOR_TREND_LOOKBACK + 2:
            continue

        conditions = {}

        # 2.1 6/21均线系统检查
        ma_ok, ma6, ma21, ma_details = check_ma_system(df)
        conditions['ma_system'] = ma_ok
        if not ma_ok:
            continue

        # 2.2 识别盘整区 + 量能收缩
        is_consolidating, cons_high, cons_low, cons_range, vol_contracting = identify_consolidation_zone(df)
        conditions['consolidation'] = is_consolidating
        if not is_consolidating:
            continue

        # 2.3 验证前期趋势（非高位）
        trend_ok, trend_label = check_prior_trend_condition(df)
        conditions['prior_trend'] = trend_ok
        if not trend_ok:
            continue

        # 2.4 验证K线质量（大阳线+实体+放量+突破）
        kline_ok, metrics = validate_one_yang_kline(df, cons_high)
        conditions['big_yang'] = metrics.get('cond_a', False)
        conditions['solid_body'] = metrics.get('cond_b', False)
        conditions['heavy_volume'] = metrics.get('cond_c', False)
        conditions['breakout'] = metrics.get('cond_d', False)
        if not kline_ok:
            continue

        # ── 增强指标计算 ──
        O = float(df['open'].iloc[-1])
        C = float(df['close'].iloc[-1])
        H = float(df['high'].iloc[-1])
        L = float(df['low'].iloc[-1])
        closes_series = df['close']

        # 一阳穿多线
        ma_penetrations = _count_ma_penetrations(closes_series, O, C)

        # 前日收窄
        prev_pinching = _check_prev_day_pinching(df)

        # 影线占比
        lower_shadow, upper_shadow = _calc_shadow_ratios(O, C, H, L)
        shadow_ok = lower_shadow <= MAX_LOWER_SHADOW and upper_shadow <= MAX_UPPER_SHADOW

        # 均线粘合
        ma_sticky, ma_spread = _calc_ma_stickiness(closes_series, C)

        # 板块共振（有行业数据时）
        sector_in_top = False
        ind = c.get('industry', '')
        if ind and sector_top_set:
            sector_in_top = ind in sector_top_set or any(
                ind in tn or tn in ind for tn in sector_top_set
            )

        # ── 多维度质量评级 ──
        vol_r = metrics['vol_ratio_5d']
        body_r = metrics['body_ratio']
        turn_over = c['turn_over']
        # guard_score 来自 MarketGuard（外部传入）

        # 打分细则（每项通过+1）
        score = 0
        reasons = []
        score_details = []

        # 量能维度
        if vol_r >= A_GRADE_VOL_RATIO:
            score += 2; reasons.append('超强放量'); score_details.append(f'量能{(vol_r*100):.0f}%')
        elif vol_r >= B_GRADE_VOL_RATIO:
            score += 1; reasons.append('放量充分'); score_details.append(f'量能{(vol_r*100):.0f}%')

        # 实体维度
        if body_r >= A_GRADE_BODY_RATIO:
            score += 2; reasons.append('实体饱满'); score_details.append(f'实体{(body_r*100):.0f}%')
        elif body_r >= B_GRADE_BODY_RATIO:
            score += 1; reasons.append('实体扎实')

        # 影线维度
        if shadow_ok:
            score += 1; reasons.append('影线健康')
        else:
            if lower_shadow > MAX_LOWER_SHADOW:
                reasons.append('下影偏长')
            if upper_shadow > MAX_UPPER_SHADOW:
                reasons.append('上影偏长')

        # 趋势维度（下跌后转折最强）
        if trend_label == 'DOWNTREND':
            score += 2; reasons.append('下跌转折')
        elif trend_label == 'CONSOLIDATION':
            score += 1

        # 盘整质量维度
        if vol_contracting:
            score += 1; reasons.append('盘整缩量蓄力')

        # 均线维度
        if ma_penetrations >= 3:
            score += 2; reasons.append(f'一阳穿{ma_penetrations}线')
        elif ma_penetrations >= 2:
            score += 1; reasons.append(f'穿透{ma_penetrations}线')

        if ma_sticky:
            score += 1; reasons.append('均线粘合')

        # 前日形态
        if prev_pinching:
            score += 1; reasons.append('前日收窄')

        # 换手率维度（有数据时）
        if turn_over >= 3 and turn_over <= 15:
            score += 1; reasons.append('换手适中')

        # 板块共振
        if sector_in_top:
            score += 1; reasons.append('板块共振')

        # ── 大盘环境软分级 ──
        guard_tier = ''
        if guard_score >= 80:
            guard_tier = 'warm'
            # 大盘偏暖：B/C 门槛略放宽
            if score >= 5:
                reasons.append('大盘偏暖')
        elif guard_score < 40:
            guard_tier = 'cold'
            reasons.append('⚠️大盘偏冷')
            # 大盘偏冷：A 级独立性反而加分
            if trend_label == 'DOWNTREND' and vol_r >= B_GRADE_VOL_RATIO:
                score += 1
                reasons.append('逆势抗跌')

        # ── 定级 ──
        if guard_tier == 'warm':
            # 暖市：B/C门槛各降1分
            if score >= 9:
                quality = 'A'
            elif score >= 5:
                quality = 'B'
            else:
                quality = 'C'
        elif guard_tier == 'cold':
            # 冷市：A门槛维持，B门槛提高
            if score >= 10:
                quality = 'A'
            elif score >= 7:
                quality = 'B'
            else:
                quality = 'C'
        else:
            if score >= 10:
                quality = 'A'
            elif score >= 6:
                quality = 'B'
            else:
                quality = 'C'

        # 买卖参考
        stop_loss = round(cons_low * 0.97, 2)
        entry_low = round(cons_high, 2)
        entry_high = round(C * 1.02, 2)

        results.append(OneFingerStock(
            code=code, name=c['name'],
            close=C, open=O, high=H, low=L,
            change_pct=c['change_pct'],
            body_ratio=body_r,
            vol_ratio_5d=vol_r,
            turn_over=turn_over,
            industry=c['industry'],
            is_limit_up=c['is_limit_up'],
            consolidation_high=cons_high,
            consolidation_low=cons_low,
            consolidation_range=cons_range,
            vol_contracting=vol_contracting,
            prior_trend=trend_label,
            lower_shadow_ratio=lower_shadow,
            upper_shadow_ratio=upper_shadow,
            ma_penetrations=ma_penetrations,
            prev_day_pinching=prev_pinching,
            ma_sticky=ma_sticky,
            sector_in_top=sector_in_top,
            guard_score=guard_score,
            entry_zone=f"{entry_low:.2f}~{entry_high:.2f}",
            stop_loss=f"{stop_loss:.2f}（盘整低{cons_low:.2f}下方3%）",
            quality=quality,
            quality_reasons=reasons,
            conditions=conditions,
        ))

    # 按质量排序（score隐含在quality中，同级别按量能排序）
    quality_order = {'A': 0, 'B': 1, 'C': 2}
    results.sort(key=lambda s: (quality_order.get(s.quality, 9), -s.vol_ratio_5d))

    a_count = len([r for r in results if r.quality == 'A'])
    b_count = len([r for r in results if r.quality == 'B'])
    c_count = len([r for r in results if r.quality == 'C'])
    logger.info(
        f"STEP2 精选完成: {len(results)} 只 "
        f"(A级{a_count}只, B级{b_count}只, C级{c_count}只)"
    )
    if a_count > 0:
        a_stocks = [(r.name, r.code, r.quality_reasons) for r in results if r.quality == 'A']
        for name, code, reasons_str in a_stocks:
            logger.info(f"  🏆 A级: {name}({code}) — {', '.join(reasons_str)}")

    return results


# ═══════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════

def build_one_finger_report(
    stocks: List[OneFingerStock],
    emotion_cycle: str = "",
    market_guard: Optional[Dict] = None,
) -> str:
    """构建一阳指选股报告（午盘版 — 条件清单 + 安检 + 板块分布）"""
    today = datetime.now().strftime('%m-%d')

    a_stocks = [s for s in stocks if s.quality == 'A']
    b_stocks = [s for s in stocks if s.quality == 'B']
    c_stocks = [s for s in stocks if s.quality == 'C']

    # ── 板块集中度 ──
    sector_count: Dict[str, int] = {}
    for s in stocks:
        ind = s.industry if s.industry else "其他"
        sector_count[ind] = sector_count.get(ind, 0) + 1
    sector_parts = [f"{k}({v}只)" for k, v in sorted(sector_count.items(), key=lambda x: -x[1])]
    sector_line = " | ".join(sector_parts[:5])
    concentrated = [k for k, v in sector_count.items() if v >= 3]

    buf = [
        f"## 一阳指 · {today} 午盘",
        f"情绪: **{emotion_cycle}** | 选股: **{len(stocks)}**只 "
        f"(A{len(a_stocks)}/B{len(b_stocks)}/C{len(c_stocks)})",
        "",
        f"> 板块: {sector_line}",
    ]
    if concentrated:
        buf.append(f"> ⚠️ {'/'.join(concentrated)}集中度偏高，注意板块贝塔风险")
    buf.append("")

    # ── 安检结果 ──
    if market_guard:
        g = market_guard
        guard_icon = "✅" if g.get('passed') else "⚠️"
        buf.append(f"> 安检: {guard_icon} 沪指{g.get('sse_close','?')} | 涨跌比{g.get('up_ratio','?')} | 跌停{g.get('limit_down','?')}家 | 仓位{g.get('position_advice','?')}")
        if g.get('warnings'):
            for w in g['warnings']:
                buf.append(f"> ⚠️ {w}")
        buf.append("")

    if not stocks:
        buf.append("今日无一阳指信号，等待盘整突破形态。\n")
        buf.append("*经典一阳指：长期下跌/盘整后，放量长阳突破盘整平台。*\n")
        return "\n".join(buf)

    # A级（最多3只）
    if a_stocks:
        buf.append("**A级 — 强转折（优先关注）**")
        for i, s in enumerate(a_stocks[:3]):
            buf.append(_stock_condition_line(i, s))
            buf.append("")
        buf.append("")

    # B级（最多3只）
    if b_stocks:
        buf.append("**B级 — 标准突破**")
        for i, s in enumerate(b_stocks[:3]):
            buf.append(_stock_condition_line(i, s))
            buf.append("")
        buf.append("")

    # C级（最多1只）
    if c_stocks:
        buf.append("**C级 — 待观察（需次日确认）**")
        for i, s in enumerate(c_stocks[:1]):
            buf.append(_stock_condition_line(i, s))
            buf.append("")
        buf.append("")

    # 战法纪律
    buf.append("---")
    buf.append(
        "纪律: 突破盘整高点介入 | "
        "盘整低下方3%止损 | "
        "前期涨>15%不追 | "
        "仅供参考不构成投资建议"
    )
    buf.append("")
    buf.append("⏰ 午盘盘中信号，收盘需确认")

    # ── 回测段（如果上午已有选股文件） ──
    backtest = build_one_finger_backtest("data/one_finger_picks_today.json")
    if backtest:
        buf.append("\n\n---\n")
        buf.append(backtest)

    return "\n".join(buf)


def _stock_condition_line(idx: int, s: OneFingerStock) -> str:
    """单行格式：股票 + 关键指标 + 入场/止损"""
    label = ["①", "②", "③", "④", "⑤"][idx]
    industry = s.industry if s.industry else "—"

    # 关键指标（替换掉无意义的✅清单）
    metrics = []
    metrics.append(f"量{(s.vol_ratio_5d*100):.0f}%")      # 量能倍数
    metrics.append(f"实体{(s.body_ratio*100):.0f}%")       # 实体占比
    if s.turn_over > 0:
        metrics.append(f"换手{s.turn_over:.1f}%")          # 换手率
    if s.ma_penetrations >= 2:
        metrics.append(f"穿{s.ma_penetrations}线")         # 均线穿透
    if s.vol_contracting:
        metrics.append("缩量蓄力")
    if s.prev_day_pinching:
        metrics.append("前日收窄")
    if s.ma_sticky:
        metrics.append("均线粘合")
    if s.sector_in_top:
        metrics.append("板块共振")
    if s.prior_trend == 'DOWNTREND':
        metrics.append("下跌转折")

    return (
        f"{label} **{s.name}**({s.code}) +{s.change_pct:.1f}% | {industry}\n"
        f"   {' · '.join(metrics)} | 入场{s.consolidation_high:.2f} 止损{s.stop_loss.split('（')[0]}"
    )


# ═══════════════════════════════════════════════════════════════
# Save/Load for Backtest
# ═══════════════════════════════════════════════════════════════

def save_one_finger_picks(stocks: List[OneFingerStock], filepath: str = "data/one_finger_picks_today.json"):
    """保存一阳指选股结果，供下午复盘回测"""
    picks = []
    for s in stocks:
        picks.append({
            'code': s.code,
            'name': s.name,
            'quality': s.quality,
            'score_reasons': s.quality_reasons,
            'close_price': s.close,
            'change_pct': s.change_pct,
            'body_ratio': s.body_ratio,
            'vol_ratio_5d': s.vol_ratio_5d,
            'prior_trend': s.prior_trend,
            'consolidation_range': s.consolidation_range,
            'vol_contracting': s.vol_contracting,
            'ma_penetrations': s.ma_penetrations,
            'prev_day_pinching': s.prev_day_pinching,
            'ma_sticky': s.ma_sticky,
            'sector_in_top': s.sector_in_top,
            'turn_over': s.turn_over,
            'date': datetime.now().strftime('%Y-%m-%d'),
        })
    try:
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(picks, f, ensure_ascii=False, indent=2)
        logger.info(f"一阳指选股结果已保存: {filepath} ({len(picks)}只)")
    except Exception as e:
        logger.warning(f"保存一阳指选股失败: {e}")


def build_one_finger_backtest(filepath: str = "data/one_finger_picks_today.json") -> str:
    """生成一阳指回测段"""
    if not os.path.exists(filepath):
        return ""

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            picks = json.load(f)
        if not picks:
            return ""

        buf = ["\n### 今日一阳指回测\n"]
        buf.append("今早推的一阳指表现：\n")

        import requests as _requests

        codes_tx = []
        for p in picks:
            code = p['code']
            if code.startswith('6'):
                codes_tx.append(f'sh{code}')
            else:
                codes_tx.append(f'sz{code}')

        tx_codes = ','.join(codes_tx)
        url = f'http://qt.gtimg.cn/q={tx_codes}'
        resp = _requests.get(url, timeout=10)
        resp.encoding = 'gbk'
        raw_lines = resp.text.strip().split('\n')
        quote_map = {}
        for line in raw_lines:
            if '~' not in line:
                continue
            parts = line.split('~')
            if len(parts) < 35:
                continue
            tc = parts[2]
            quote_map[tc] = {
                'name': parts[1],
                'price': float(parts[3]) if parts[3] else 0,
                'change_pct': float(parts[32]) if parts[32] else 0,
            }

        hit = 0
        total_pnl = 0.0
        for p in picks:
            code = p['code']
            name = p['name']
            morning_price = p.get('close_price', 0)
            q = quote_map.get(code)
            if q and q['price'] and q['price'] > 0:
                close_p = q['price']
                day_change = (close_p - morning_price) / morning_price * 100 if morning_price > 0 else q['change_pct']

                if day_change > 5:
                    status, advice = "✅", "强势延续"
                elif day_change > 0:
                    status, advice = "✅", "收涨持有"
                elif day_change > -3:
                    status, advice = "⚠️", "小幅回调"
                else:
                    status, advice = "❌", "破位止损"

                if day_change > 0:
                    hit += 1
                quality = p.get('quality', '')
                buf.append(f"  {status} {name}({code}) [{quality}级] → {close_p:.2f}({day_change:+.1f}%) — {advice}\n")
                total_pnl += day_change
            else:
                buf.append(f"  ⚪ {name}({code}) — 数据暂不可用\n")

        if len(picks) > 0:
            avg_pnl = total_pnl / len(picks)
            hit_rate = hit / len(picks) * 100
            buf.append(f"\n一阳指命中率: {hit}/{len(picks)} ({hit_rate:.0f}%) | 平均收益: {avg_pnl:+.1f}%\n")

        return "\n".join(buf)
    except Exception as e:
        logger.warning(f"一阳指回测生成失败: {e}")
        return ""
