# -*- coding: utf-8 -*-
"""
飞龙在天战法 — 全A股票筛选引擎

Pipeline:
  Phase 1: 大盘安检（动态标准）
  Phase 2: 主线确认（五项全过）
  Phase 3: 飞龙选股（四大条件 + 产业地位 + 伪龙过滤）
  Phase 4: 三线分类 + 仓位建议
  Phase 5: 生成选股报告(LLM)
  Phase 6: 持仓跟踪
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 指数/板块代码映射
INDEX_MAP = {
    'sh000001': '上证指数',
}

@dataclass
class MarketGuardResult:
    """大盘安检结果"""
    passed: bool
    score: int = 0
    details: List[str] = field(default_factory=list)
    failed_items: List[str] = field(default_factory=list)
    advice: str = ""

@dataclass
class MainLineSector:
    """主线板块"""
    name: str
    score: int = 0
    limit_up_count: int = 0
    max_consecutive: int = 0
    consecutive_days_top10: int = 0
    event_stage: str = ""  # 萌芽/扩散/高潮/兑现
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    is_mainline: bool = False

@dataclass
class FlyingDragon:
    """飞龙候选股"""
    code: str
    name: str
    industry: str
    mainline: str
    consecutive: int
    close_price: float
    change_pct: float
    ma5: float
    ma10: float
    ma20: float
    ma_alignment: str  # 多头排列/空头排列/缠绕
    volume_ratio: float  # 今日量/MA5均量
    has_limit_up_5d: bool
    sector_rank: int
    industry_position: str = ""  # LLM判断的产业地位
    entry_line: str = ""  # 一号线/二号线/三号线/观望
    entry_signal: str = ""
    suggested_position: str = ""
    is_real_dragon: bool = True  # 真龙头 vs 伪龙


class MarketGuard:
    """Phase 1: 大盘安检"""

    @staticmethod
    def check(ss_index, market_stats, emotion_cycle: str) -> MarketGuardResult:
        """执行大盘安检，五项动态标准。"""
        result = MarketGuardResult(passed=True)

        # --- 1. 成交量 ---
        # 从指数dataframe中获取近20日均量
        if ss_index is not None and len(ss_index) >= 20:
            avg_vol_20 = ss_index['volume'].tail(20).mean()
            today_vol = ss_index['volume'].iloc[-1]
            vol_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 0
            if vol_ratio >= 0.75:
                result.details.append(f"成交额>近20日均量75% ✅ ({vol_ratio:.0%})")
                result.score += 20
            else:
                result.details.append(f"成交额不足 ❌ ({vol_ratio:.0%} < 75%)")
                result.failed_items.append("成交量萎缩")
                result.passed = False
        else:
            result.details.append("成交额数据不足 ⚠️")

        # --- 2. 指数位置 ---
        if ss_index is not None and len(ss_index) >= 20:
            close = ss_index['close'].iloc[-1]
            ma20 = ss_index['close'].rolling(20).mean().iloc[-1]
            if close > ma20:
                # 检查MA20斜率
                ma20_prev = ss_index['close'].rolling(20).mean().iloc[-2]
                slope_ok = ma20 >= ma20_prev
                result.details.append(
                    f"沪指{close:.0f}站上MA20({ma20:.0f}) ✅" if slope_ok
                    else f"沪指站上MA20但MA20走平 ⚠️"
                )
                result.score += 20 if slope_ok else 10
            else:
                result.details.append(f"沪指{close:.0f}低于MA20({ma20:.0f}) ❌")
                result.failed_items.append("指数破位")
                result.passed = False
        else:
            result.details.append("指数数据不足 ⚠️")

        # --- 3. 赚钱效应 ---
        up = market_stats.get('up_count', 0)
        down = market_stats.get('down_count', 0)
        if down > 0:
            ratio = up / down
            if ratio > 0.7:
                result.details.append(f"涨跌比{ratio:.2f}>0.7 ✅")
                result.score += 20
            else:
                result.details.append(f"涨跌比{ratio:.2f}≤0.7 ❌")
                result.failed_items.append("赚钱效应差")
                result.passed = False
        else:
            result.details.append("涨跌数据不足 ⚠️")

        # --- 4. 恐慌程度 ---
        limit_up = market_stats.get('limit_up_count', 0)
        limit_down = market_stats.get('limit_down_count', 0)
        if limit_down < limit_up * 0.6:
            result.details.append(
                f"跌停{limit_down}<涨停×60%({limit_up*0.6:.0f}) ✅"
                if limit_down < limit_up * 0.5
                else f"跌停{limit_down}略高但未达恐慌({limit_down/limit_up:.0%}) ⚠️"
            )
            result.score += 15 if limit_down < limit_up * 0.5 else 10
        else:
            result.details.append(f"跌停{limit_down}≥涨停×60% ❌ 恐慌踩踏")
            result.failed_items.append("恐慌踩踏")
            result.passed = False

        # --- 5. 情绪周期 ---
        if emotion_cycle not in ('冰点', '退潮'):
            result.details.append(f"情绪周期:{emotion_cycle} ✅")
            result.score += 20
        else:
            result.details.append(f"情绪周期:{emotion_cycle} ❌")
            result.failed_items.append("系统性概率不利")
            result.passed = False

        # 综合建议
        if result.passed:
            if result.score >= 80:
                result.advice = "飞龙战法可以出手，仓位可积极"
            else:
                result.advice = "安检通过但部分指标偏弱，仓位需保守"
        else:
            result.advice = f"安检不通过({', '.join(result.failed_items)})，建议空仓/轻仓观望"

        return result


class SectorConfirmer:
    """Phase 2: 主线确认"""

    @staticmethod
    def confirm(limit_up_pool: List[Dict], sector_rankings_data, analyzer=None) -> List[MainLineSector]:
        """确认当前主线板块，五项标准判断。"""
        # 按行业聚合涨停股
        industry_lu = {}  # {industry: [stock_dicts]}
        for s in limit_up_pool:
            ind = s.get('industry', '其他')
            industry_lu.setdefault(ind, []).append(s)

        total_lu = len(limit_up_pool)
        all_top10_sectors = {s['name'] for s in sector_rankings_data[0][:10]} if sector_rankings_data and len(sector_rankings_data) > 0 and sector_rankings_data[0] else set()

        results = []
        for ind, stocks in industry_lu.items():
            if len(stocks) < 3:  # 至少3只涨停
                continue

            consecutive_max = max(s.get('consecutive', 1) for s in stocks)
            sector = MainLineSector(
                name=ind,
                limit_up_count=len(stocks),
                max_consecutive=consecutive_max,
            )

            # ① 涨停集中度（3只+ 或 占比>8%）
            if len(stocks) >= 3:
                sector.checks_passed.append("涨停集中度(≥3只)")
            else:
                sector.checks_failed.append(f"涨停仅{len(stocks)}只")

            # ② 连板梯度（≥3板龙头 或 2板但涨停≥5只）
            if consecutive_max >= 3 or (consecutive_max >= 2 and len(stocks) >= 5):
                sector.checks_passed.append("连板梯度")
            else:
                sector.checks_failed.append(f"连板力度不足(最高{consecutive_max}板)")

            # ③ 持续性（在涨幅前10 或 涨停数≥5）
            if ind in all_top10_sectors or len(stocks) >= 5:
                sector.checks_passed.append("持续性/热度")
                sector.consecutive_days_top10 = 1 if ind in all_top10_sectors else 0
            else:
                sector.checks_failed.append("未进涨幅前10且涨停<5只")

            # 评分
            sector.score = (len(sector.checks_passed) + 1) * 20 + min(consecutive_max, 5) * 3 + len(stocks)

            # 至少2项通过 + 事件发酵/产业逻辑(LLM后续确认)
            if len(sector.checks_passed) >= 2:
                sector.is_mainline = True

            results.append(sector)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:5]


class FlyingDragonScreener:
    """Phase 3-4: 飞龙选股 + 伪龙过滤 + 三线分类"""

    @staticmethod
    def screen(
        limit_up_pool: List[Dict],
        mainline_sectors: List[MainLineSector],
        data_manager,
    ) -> List[FlyingDragon]:
        """从涨停池中筛选飞龙候选。"""
        mainline_names = {m.name for m in mainline_sectors if m.is_mainline}
        if not mainline_names:
            return []

        dragons = []
        for stock in limit_up_pool:
            ind = stock.get('industry', '')
            if ind not in mainline_names:  # 只筛主线板块内的
                continue

            code = stock.get('code', '')
            name = stock.get('name', '')
            consecutive = stock.get('consecutive', 1)

            # 获取日线数据
            try:
                df, _ = data_manager.get_daily_data(code, days=60)
                if df is None or df.empty or len(df) < 10:
                    continue
            except Exception:
                continue

            # 计算均线和指标
            df = df.copy()
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma10'] = df['close'].rolling(10).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            df['vol_ma5'] = df['volume'].rolling(5).mean()

            last = df.iloc[-1]
            ma5 = last.get('ma5', 0)
            ma10 = last.get('ma10', 0)
            ma20 = last.get('ma20', 0)
            close = last['close']
            volume = last['volume']
            avg_vol = last.get('vol_ma5', volume) or volume
            vol_ratio = volume / avg_vol if avg_vol > 0 else 1

            # ① 均线昂首: MA5>MA10>MA20, MA5斜率>0
            if not (ma5 > ma10 > ma20 and pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20)):
                continue

            # MA5斜率：最近5日MA5是否持续上升（或连板中维持向上）
            df['ma5_slope'] = df['ma5'].diff()
            ma5_trend = df['ma5_slope'].tail(3).mean()  # 近3日平均斜率
            if ma5_trend <= 0:
                continue

            # ② 底部放量: 连板≥2天可放宽标准
            open_p = last['open']
            k_body_pct = abs(close - open_p) / open_p * 100 if open_p > 0 else 0

            if consecutive >= 4:
                # ≥4板龙头：惜售极度缩量是强信号，不限量比
                pass
            elif consecutive >= 3:
                # 3板：惜售开始，量比>0.5x即可
                if vol_ratio < 0.5:
                    continue
            elif consecutive >= 2:
                # 中位连板：量比>1.2x, 实体>1%
                if vol_ratio < 1.2:
                    continue
                if k_body_pct < 1:
                    continue
            else:
                # 首板：需要明显的底部放量，量比>1.5x, 实体>2%
                if vol_ratio < 1.5:
                    continue
                if k_body_pct < 2:
                    continue

            # ③ 近期涨停: 已在涨停池中满足

            # ④ 辨识前排: 在主线板块中排名
            sector_lu_stocks = [s for s in limit_up_pool if s.get('industry', '') == ind]
            sector_lu_stocks.sort(key=lambda x: x.get('consecutive', 0), reverse=True)
            rank = next((i+1 for i, s in enumerate(sector_lu_stocks) if s.get('code') == code), 99)
            if rank > 10:
                continue

            # ⑥ 伪龙过滤
            # 跟风龙: 板块内最晚涨停且涨幅最低
            # 反包龙: 前日跌停今日涨停(从K线判断)
            prev_close = df['close'].iloc[-2] if len(df) >= 2 else close
            prev_pct = (prev_close - df['close'].iloc[-3]) / df['close'].iloc[-3] * 100 if len(df) >= 3 else 0
            if prev_pct < -9.5:  # 前日跌停，今日涨停 = 反包龙
                continue

            # 庄股龙: 换手率极低但涨停（没有换手率字段则跳过此检查）
            # 消息龙标记留给LLM判断

            # 确定三线分类
            entry_line, entry_signal, suggested_position = _classify_entry(
                df, ma5, ma10, ma20, close, vol_ratio
            )
            if not entry_line:
                continue

            alignment = "多头排列" if ma5 > ma10 > ma20 else "缠绕"

            dragon = FlyingDragon(
                code=code,
                name=name,
                industry=ind,
                mainline=ind,
                consecutive=consecutive,
                close_price=close,
                change_pct=stock.get('change_pct', 0),
                ma5=ma5,
                ma10=ma10,
                ma20=ma20,
                ma_alignment=alignment,
                volume_ratio=vol_ratio,
                has_limit_up_5d=True,
                sector_rank=rank,
                entry_line=entry_line,
                entry_signal=entry_signal,
                suggested_position=suggested_position,
            )
            dragons.append(dragon)

        dragons.sort(key=lambda d: d.consecutive, reverse=True)
        return dragons


def _classify_entry(df, ma5, ma10, ma20, close, vol_ratio) -> Tuple[str, str, str]:
    """三线分类判断"""
    # 计算MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    dif = ema12.iloc[-1] - ema26.iloc[-1]
    dea = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]
    macd_golden = dif > dea if pd.notna(dif) and pd.notna(dea) else False

    # 量缩（<昨日60%）
    if len(df) >= 2:
        prev_vol = df['volume'].iloc[-2]
        vol_shrinking = df['volume'].iloc[-1] < prev_vol * 0.6
    else:
        vol_shrinking = False

    dist_ma5 = abs(close - ma5) / ma5 * 100 if ma5 > 0 else 100

    # 飞龙运行中: MA多头 + 远离MA5(>3%) → 持股待涨，回踩加仓
    if pd.notna(ma5) and close > ma5 * 1.03 and ma5 > ma10 > ma20:
        wait_line = "等待回踩5日线加仓" if dist_ma5 > 5 else "观察5日线支撑"
        return "运行中", f"飞龙运行中(偏离MA5 {dist_ma5:.0f}%)，{wait_line}", "10%"

    # 一号线: close距MA5<1.5% + 缩量 + MACD金叉
    if dist_ma5 < 1.5 and vol_shrinking and macd_golden:
        return "一号线", "回踩5日线企稳+缩量+MACD金叉", "15%"

    # 二号线: 前日<MA10, 今日>MA10 + 放量
    if len(df) >= 2 and vol_ratio > 1.2:
        prev_close = df['close'].iloc[-2]
        prev_ma10 = df['ma10'].iloc[-2]
        if pd.notna(prev_ma10) and prev_close < prev_ma10 and close > ma10:
            return "二号线", "突破10日线+放量确认", "10%"

    # 三号线: 高点回撤>15% + 站上MA20 + 放量
    if len(df) >= 30 and vol_ratio > 1.0:
        high_20 = df['high'].tail(20).max()
        drawdown = (high_20 - close) / high_20 * 100 if high_20 > 0 else 0
        if drawdown > 15 and close > ma20:
            return "三号线", f"高点回撤{drawdown:.0f}%+站上20日线", "8%"

    return "", "", ""


def build_flying_dragon_report(
    guard_result: MarketGuardResult,
    mainline_sectors: List[MainLineSector],
    dragons: List[FlyingDragon],
    analyzer,
    emotion_cycle: str = "",
) -> str:
    """Phase 5: 构建飞龙选股报告（交易员早盘风格）"""
    today = datetime.now().strftime('%m-%d')

    if not guard_result.passed:
        return f"## 飞龙在天 · {today}\n安检不通过（{', '.join(guard_result.failed_items)}），今日不筛飞龙，空仓/轻仓。\n"

    # 核心矛盾推断
    max_consec = max((d.consecutive for d in dragons), default=0)
    total_dragons = len(dragons)
    if max_consec >= 4:
        tension = "龙头加速，关注分歧换手"
    elif max_consec >= 3:
        tension = "梯队完整，寻找回踩切入点"
    elif total_dragons <= 2:
        tension = "飞龙稀少，耐心等待信号"
    else:
        tension = "分散轮动，聚焦量价最优"

    risk_hints = [f for f in guard_result.failed_items]
    if guard_result.score < 80:
        risk_hints.append(f"安检仅{guard_result.score}分")

    buf = [
        f"## 飞龙在天 · {today}",
        f"情绪: **{emotion_cycle}** | 飞龙: **{total_dragons}**只 | {tension}",
        "",
    ]

    if not dragons:
        buf.append("无满足四大条件的飞龙，等待更好信号。\n")
        return "\n".join(buf)

    # 按优先级排序：连板>板块排名>量比
    sorted_dragons = sorted(dragons, key=lambda d: (d.consecutive, -d.sector_rank, -d.volume_ratio), reverse=True)

    # Top 2 重点关注
    top2 = sorted_dragons[:2]
    buf.append("**今日关注：**")
    for i, d in enumerate(top2):
        label = ["①", "②"][i]
        ma5_target = d.ma5 * 1.02 if d.close_price > d.ma5 * 1.05 else d.ma5
        buf.append(
            f"  {label} **{d.name}**({d.code}) — {d.consecutive}连板·{d.industry}龙{'头' if d.sector_rank<=2 else '二'}"
            f" | 偏离MA5 {abs(d.close_price/d.ma5*100-100):.0f}%"
            f" | {'等回踩5日线≈{:.0f}元'.format(d.ma5) if d.close_price > d.ma5*1.03 else '回踩企稳中'}"
        )

    # 其余等回踩观察
    rest = sorted_dragons[2:6]
    if rest:
        buf.append("")
        buf.append("**等回踩观察：**")
        for i, d in enumerate(rest):
            label = ["③", "④", "⑤", "⑥"][i]
            buf.append(
                f"  {label} {d.name}({d.code}) — {d.consecutive}连板·{d.industry}"
                f" | 量比{d.volume_ratio:.1f}x | {d.entry_signal}"
            )

    # 仓位与风险
    risk_str = "、".join(risk_hints) if risk_hints else "暂无显著风险"
    buf.append("")
    buf.append(f"仓位: 试错期3-4成 | 风险: {risk_str}")
    buf.append(f"\n*飞龙在天战法选股，仅供参考，不构成投资建议。*\n")

    return "\n".join(buf)


def save_dragon_picks_for_backtest(dragons: List[FlyingDragon], filepath: str = "data/dragon_picks_today.json"):
    """保存今日飞龙选股结果，供复盘回测使用。"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        picks = []
        for d in dragons:
            picks.append({
                'code': d.code,
                'name': d.name,
                'consecutive': d.consecutive,
                'industry': d.industry,
                'close_price': d.close_price,
                'entry_line': d.entry_line,
                'date': datetime.now().strftime('%Y-%m-%d'),
            })
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(picks, f, ensure_ascii=False, indent=2)
        logger.info(f"飞龙选股已保存: {filepath} ({len(picks)}只)")
    except Exception as e:
        logger.warning(f"飞龙选股保存失败: {e}")


def build_dragon_backtest_section(filepath: str = "data/dragon_picks_today.json") -> str:
    """读取今早飞龙选股，对比市场收盘数据生成回测段。仅查询飞龙候选股，秒级完成。"""
    if not os.path.exists(filepath):
        return ""

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            picks = json.load(f)
        if not picks:
            return ""

        buf = ["\n### 今日飞龙回测\n"]
        buf.append("今早推的飞龙表现：\n")

        # 批量从腾讯财经获取实时行情（一次一只，但只查飞龙候选，7-8只秒级）
        import requests as _requests
        hit = 0
        total_pnl = 0.0

        codes_tx = []
        for p in picks:
            code = p['code']
            if code.startswith('6'):
                codes_tx.append(f'sh{code}')
            else:
                codes_tx.append(f'sz{code}')

        # 批量查询腾讯行情
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
            tx_code = parts[2]
            tc = tx_code  # parts[2] 已经是纯数字代码，无需去前缀
            quote_map[tc] = {
                'name': parts[1],
                'price': float(parts[3]) if parts[3] else 0,
                'change_pct': float(parts[32]) if parts[32] else 0,
            }

        for p in picks:
            code = p['code']
            name = p['name']
            morning_price = p.get('close_price', 0)
            q = quote_map.get(code)
            if q and q['price'] and q['price'] > 0:
                close_p = q['price']
                day_change = q['change_pct'] if morning_price == 0 else (close_p - morning_price) / morning_price * 100

                if day_change > 5:
                    status, advice = "✅", "强势，明天关注加速"
                elif day_change > 0:
                    status, advice = "✅", "收涨，持有观察"
                elif day_change > -3:
                    status, advice = "⚠️", "小幅回调，明天减仓"
                else:
                    status, advice = "❌", "大幅回落，明天清仓"

                if day_change > 0:
                    hit += 1
                buf.append(f"  {status} {name}({code}) → {close_p:.2f}({day_change:+.1f}%) — {advice}\n")
                total_pnl += day_change
            else:
                buf.append(f"  ⚪ {name}({code}) — 数据暂不可用\n")

        if len(picks) > 0:
            avg_pnl = total_pnl / len(picks)
            hit_rate = hit / len(picks) * 100
            buf.append(f"\n命中率: {hit}/{len(picks)} ({hit_rate:.0f}%) | 平均收益: {avg_pnl:+.1f}%\n")

        return "\n".join(buf)
    except Exception as e:
        logger.warning(f"飞龙回测失败: {e}")
        return ""


def _build_llm_confirm_prompt(raw_report, mainline_sectors, dragons) -> str:
    """构建LLM确认Prompt"""
    sector_info = "\n".join(
        f"- {m.name}: 涨停{m.limit_up_count}只, 最高{m.max_consecutive}板"
        for m in mainline_sectors[:3]
    )

    dragon_info = "\n".join(
        f"- {d.name}({d.code}): {d.consecutive}连板, {d.ma_alignment}, "
        f"量比{d.volume_ratio:.1f}x, {d.entry_line}"
        for d in dragons[:10]
    )

    return f"""请以飞龙在天战法视角，对以下选股结果做简短评注(纯Markdown，<1000字)：

{sector_info}

{dragon_info}

评注要点：
1. 主线板块的可持续性判断（产业逻辑是否支持持续走强）
2. 前排飞龙的产业地位（是不是真正的行业龙头/细分唯一）
3. 事件发酵阶段和操作建议
4. 风险提示（哪些可能是伪龙/跟风）
"""
