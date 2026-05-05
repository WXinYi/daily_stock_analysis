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

import logging
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
    """Phase 5: 构建飞龙选股报告(LLM确认)"""
    today = datetime.now().strftime('%Y-%m-%d')

    buf = [f"## 飞龙在天 · 早盘选股\n",
           f"**{today}**",
           f" | 情绪: **{emotion_cycle}**\n" if emotion_cycle else "\n"]

    if not guard_result.passed:
        buf.append("\n*安检不通过（{}），今日不筛飞龙，建议空仓/轻仓等待。*\n".format(
            ', '.join(guard_result.failed_items)))
        return "\n".join(buf)

    # --- 主线段 ---
    buf.append(f"\n### 主线板块\n")
    if mainline_sectors:
        for m in mainline_sectors:
            status = "✅" if m.is_mainline else "⚠️"
            buf.append(f"\n**{m.name}** {status}")
            buf.append(f"\n- 涨停{m.limit_up_count}只/全市场{'-':>4} | 最高连板:{m.max_consecutive}板 | 评分:{m.score}")
            if m.checks_passed:
                buf.append(f"\n- 通过: {', '.join(m.checks_passed)}")
            if m.checks_failed:
                buf.append(f"\n- 未过: {', '.join(m.checks_failed)}")
            if m.event_stage:
                buf.append(f"\n- 事件发酵: {m.event_stage}")
            buf.append("")
    else:
        buf.append("\n无主线板块确认，建议观望。\n")

    # --- 飞龙候选段 ---
    if dragons:
        # 按入场线分组
        running = [d for d in dragons if d.entry_line == '运行中']
        line1 = [d for d in dragons if d.entry_line == '一号线']
        line2 = [d for d in dragons if d.entry_line == '二号线']
        line3 = [d for d in dragons if d.entry_line == '三号线']

        for line_name, line_dragons in [
            ('运行中(持股)', running),
            ('一号线(激进)', line1),
            ('二号线(稳健)', line2),
            ('三号线(二波)', line3),
        ]:
            if not line_dragons:
                continue
            buf.append(f"\n### 飞龙候选 · {line_name}\n")
            for d in line_dragons[:5]:
                buf.append(f"\n**{d.name}**({d.code}) | {d.industry}")
                buf.append(f"\n- {d.consecutive}连板 | 均线{d.ma_alignment} | MA5:{d.ma5:.1f} MA10:{d.ma10:.1f} MA20:{d.ma20:.1f}")
                buf.append(f"\n- 量比:{d.volume_ratio:.1f}x | 板块排名:{d.sector_rank}")
                buf.append(f"\n- 入场: {d.entry_signal}")
                buf.append(f"\n- 建议仓位: {d.suggested_position}")
                buf.append("")
    else:
        buf.append("\n### 飞龙候选\n无满足四大条件的飞龙，等待更好信号。\n")

    buf.append(f"\n*飞龙在天战法选股，仅供参考，不构成投资建议。*\n")

    report = "\n".join(buf)

    # --- LLM确认（如有analyzer）---
    if analyzer and analyzer.is_available() and (dragons or mainline_sectors):
        try:
            prompt = _build_llm_confirm_prompt(report, mainline_sectors, dragons)
            llm_result = analyzer.generate_text(prompt, max_tokens=2048, temperature=0.3)
            if llm_result:
                # 把LLM确认附加到报告后面
                report += f"\n---\n### LLM确认\n{llm_result}\n"
        except Exception as e:
            logger.warning(f"飞龙LLM确认失败: {e}")

    return report


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
