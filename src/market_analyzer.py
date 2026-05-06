# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from src.config import get_config
from src.report_language import normalize_report_language
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)


_ENGLISH_SECTION_PATTERNS = {
    "market_summary": r"###\s*(?:1\.\s*)?Market Summary",
    "index_commentary": r"###\s*(?:2\.\s*)?(?:Index Commentary|Major Indices)",
    "sector_highlights": r"###\s*(?:4\.\s*)?(?:Sector Highlights|Sector/Theme Highlights)",
}

_CHINESE_SECTION_PATTERNS = {
    "market_summary": r"###\s*一、(?:盘面总览|市场总结)",
    "index_commentary": r"###\s*二、(?:指数结构|指数点评|主要指数)",
    "sector_highlights": r"###\s*三、(?:板块主线|热点解读|板块表现)",
    "funds_sentiment": r"###\s*四、(?:资金与情绪|资金动向)",
    "news_catalysts": r"###\s*五、(?:消息催化|后市展望)",
}


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    # north_flow: float = 0.0           # 北向资金净流入（亿元）- 已废弃，接口不可用
    
    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块


@dataclass
class LimitUpStock:
    """涨停股信息"""
    code: str
    name: str
    consecutive: int = 1           # 连板数（1=首板, 2=2连板...）
    industry: str = ""             # 所属行业
    first_time: str = ""           # 首次封板时间


@dataclass
class LimitUpLadder:
    """连板天梯数据"""
    total: int = 0                          # 涨停总数
    consecutive_stats: Dict[int, int] = field(default_factory=dict)  # {板数: 数量}
    height_leaders: List[LimitUpStock] = field(default_factory=list)  # 高度板龙头（3板+）
    industry_ladders: List[Dict] = field(default_factory=list)  # 行业天梯 [{sector, stocks_by_board: {板数: [股票]}, total}]
    concept_ladders: List[Dict] = field(default_factory=list)   # 概念天梯


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 搜索市场新闻
    5. 生成大盘复盘报告
    """
    
    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
    ):
        """
        初始化大盘分析器

        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
            region: 市场区域 cn=A股 us=美股
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us", "hk") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def _get_review_language(self) -> str:
        configured = normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )
        if self.region == "us":
            return "en"
        return configured

    def _get_template_review_language(self) -> str:
        return normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )

    def _get_market_scope_name(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "us":
            return "US market"
        if self.region == "hk":
            return "Hong Kong market" if review_language == "en" else "港股市场"
        if review_language == "en":
            return "A-share market"
        return "A股市场"

    def _get_turnover_unit_label(self) -> str:
        """Return the turnover unit label for the current market/language."""
        if self.region == "us":
            return "USD bn" if self._get_review_language() == "en" else "十亿美元"
        if self.region == "hk":
            return "HKD bn" if self._get_review_language() == "en" else "十亿港元"
        return "CNY 100m" if self._get_review_language() == "en" else "亿"

    def _format_turnover_value(self, amount_raw: float) -> str:
        """Format raw turnover according to market-specific units."""
        if amount_raw == 0.0:
            return "N/A"
        if self.region in ("us", "hk"):
            return f"{amount_raw / 1e9:.2f}"
        if amount_raw > 1e6:
            return f"{amount_raw / 1e8:.0f}"
        return f"{amount_raw:.0f}"

    def _get_review_title(self, date: str) -> str:
        if self._get_review_language() == "en":
            market_names = {"us": "US Market Recap", "hk": "HK Market Recap"}
            market_name = market_names.get(self.region, "A-share Market Recap")
            return f"## {date} {market_name}"
        return f"## {date} 大盘复盘"

    def _get_index_hint(self) -> str:
        if self._get_review_language() == "en":
            if self.region == "us":
                return "Analyze the key moves in the S&P 500, Nasdaq, Dow, and other major indices."
            if self.region == "hk":
                return "Analyze the key moves in the HSI, Hang Seng Tech, HSCEI, and other major indices."
            return "Analyze the price action in the SSE, SZSE, ChiNext, and other major indices."
        return self.profile.prompt_index_hint

    def _get_strategy_prompt_block(self) -> str:
        if self.region == "hk" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Hong Kong Market Regime Strategy
Focus on HSI trend, southbound flow dynamics, and sector rotation to define next-session risk posture.

### Strategy Principles
- Read market regime from HSI, HSTECH, and HSCEI alignment first.
- Track southbound capital flow as a key sentiment driver.
- Translate recap into actionable risk-on/risk-off stance with clear invalidation points.

### Analysis Dimensions
- Trend Regime: Classify the market as momentum, range, or risk-off.
  - Are HSI/HSTECH/HSCEI directionally aligned
  - Did volume confirm the move
  - Are key index levels reclaimed or lost
- Capital Flows: Map southbound flow and macro narrative into equity risk appetite.
  - Southbound net flow direction and magnitude
  - USD/HKD and China policy implications
  - Breadth and leadership concentration
- Sector Themes: Identify persistent leaders and vulnerable laggards.
  - Tech/internet platform trend persistence
  - Financials/property sensitivity to policy shifts
  - Defensive vs growth factor rotation

### Action Framework
- Risk-on: broad index breakout with expanding southbound participation.
- Neutral: mixed index signals; focus on selective relative strength.
- Risk-off: failed breakouts and rising volatility; prioritize capital preservation."""
        if not (self.region == "cn" and self._get_review_language() == "en"):
            return self.strategy.to_prompt_block()
        return """## Strategy Blueprint: A-share Three-Phase Recap Strategy
Focus on index trend, liquidity, and sector rotation to shape the next-session trading plan.

### Strategy Principles
- Read index direction first, then confirm liquidity structure, and finally test sector persistence.
- Every conclusion must map to position sizing, trading pace, and risk-control actions.
- Base judgments on today's data and the latest 3-day news flow without inventing unverified information.

### Analysis Dimensions
- Trend Structure: Determine whether the market is in an uptrend, range, or defensive phase.
  - Are the SSE, SZSE, and ChiNext moving in the same direction
  - Is the market advancing on expanding volume or slipping on contracting volume
  - Have key support or resistance levels been reclaimed or broken
- Liquidity & Sentiment: Identify near-term risk appetite and market temperature.
  - Advance/decline breadth and limit-up/limit-down structure
  - Whether turnover is expanding or fading
  - Whether high-beta leaders are showing divergence
- Leading Themes: Distill tradable leadership and areas to avoid.
  - Whether leading sectors have clear event catalysts
  - Whether sector leaders are pulling the group higher
  - Whether weakness is broadening across lagging sectors

### Action Framework
- Offensive: indices rise in sync, turnover expands, and core themes strengthen.
- Balanced: index divergence or low-volume consolidation; keep sizing controlled and wait for confirmation.
- Defensive: indices weaken and laggards broaden; prioritize risk control and de-risking."""

    def _get_strategy_markdown_block(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "hk" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify the market as momentum, range, or risk-off based on HSI/HSTECH/HSCEI alignment.
- **Capital Flows**: Track southbound flow direction and macro narrative for risk appetite signals.
- **Sector Themes**: Focus on tech/internet platform persistence and financials/property policy sensitivity.
"""
        if not (self.region == "cn" and review_language == "en"):
            return self.strategy.to_markdown_block()
        return """### 6. Strategy Framework
- **Trend Structure**: Determine whether the market is in an uptrend, range, or defensive phase.
- **Liquidity & Sentiment**: Track breadth, turnover expansion, and whether leaders are diverging.
- **Leading Themes**: Focus on sectors with catalysts and sustained leadership while avoiding broadening weakness.
"""

    def _get_market_mood_text(self, mood_key: str, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if review_language == "en":
            mapping = {
                "strong_up": "strong gains",
                "mild_up": "moderate gains",
                "mild_down": "mild losses",
                "strong_down": "clear weakness",
                "range": "range-bound trading",
            }
        else:
            mapping = {
                "strong_up": "强势上涨",
                "mild_up": "小幅上涨",
                "mild_down": "小幅下跌",
                "strong_down": "明显下跌",
                "range": "震荡整理",
            }
        return mapping[mood_key]

    def get_market_overview(self) -> MarketOverview:
        """
        获取市场概览数据
        
        Returns:
            MarketOverview: 市场概览数据对象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        # 1. 获取主要指数行情（按 region 切换 A 股/美股）
        overview.indices = self._get_main_indices()

        # 2. 获取涨跌统计（A 股有，美股无等效数据）
        if self.profile.has_market_stats:
            self._get_market_statistics(overview)

        # 3. 获取板块涨跌榜（A 股有，美股暂无）
        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)
        
        # 4. 获取北向资金（可选）
        # self._get_north_flow(overview)
        
        return overview

    
    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情"""
        indices = []

        try:
            logger.info("[大盘] 获取主要指数实时行情...")

            # 使用 DataFetcherManager 获取指数行情（按 region 切换）
            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[大盘] 所有行情数据源失败，将依赖新闻搜索进行分析")
            else:
                logger.info(f"[大盘] 获取到 {len(indices)} 个指数行情")

        except Exception as e:
            logger.error(f"[大盘] 获取指数行情失败: {e}")

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计"""
        try:
            logger.info("[大盘] 获取市场涨跌统计...")

            stats = self.data_manager.get_market_stats()

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(f"[大盘] 涨:{overview.up_count} 跌:{overview.down_count} 平:{overview.flat_count} "
                          f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                          f"成交额:{overview.total_amount:.0f}亿")

        except Exception as e:
            logger.error(f"[大盘] 获取涨跌统计失败: {e}")

    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] 获取板块涨跌榜...")

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(f"[大盘] 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                logger.info(f"[大盘] 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")

        except Exception as e:
            logger.error(f"[大盘] 获取板块涨跌榜失败: {e}")
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """获取北向资金流入"""
    #     try:
    #         logger.info("[大盘] 获取北向资金...")
    #         
    #         # 获取北向资金数据
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    #         
    #         if df is not None and not df.empty:
    #             # 取最新一条数据
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
    #                 
    #             logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
    #
    #     except Exception as e:
    #         logger.warning(f"[大盘] 获取北向资金失败: {e}")

    def _build_limit_up_ladder(self) -> Optional[LimitUpLadder]:
        """获取涨停池数据并构建连板天梯（行业+概念双维度）。"""
        try:
            logger.info("[大盘] 获取涨停池连板数据...")

            # 从 AkShare 获取涨停池
            from data_provider.akshare_fetcher import AkshareFetcher
            akshare = AkshareFetcher()
            pool = akshare.get_limit_up_pool()

            if pool is None:
                logger.warning("[连板] 涨停池接口失败")
                return None
            if not pool:
                logger.info("[连板] 今日无涨停数据")
                return LimitUpLadder(total=0)

            # 构建股票列表
            stocks = []
            for item in pool:
                stock = LimitUpStock(
                    code=str(item.get('code', '')),
                    name=str(item.get('name', '')),
                    consecutive=int(item.get('consecutive', 1)),
                    industry=str(item.get('industry', '')),
                    first_time=str(item.get('first_time', '')),
                )
                stocks.append(stock)

            # 连板梯度统计
            from collections import Counter
            consec_dist = Counter(s.consecutive for s in stocks)
            consecutive_stats = {}
            for k in sorted(consec_dist):
                consecutive_stats[k] = consec_dist[k]

            # 高度板龙头（3连板及以上，按连板数降序）
            height_leaders = sorted(
                [s for s in stocks if s.consecutive >= 3],
                key=lambda s: s.consecutive, reverse=True
            )[:10]

            # ---- 行业天梯 ----
            industry_map = {}  # {sector_name: {consecutive: [LimitUpStock]}}
            for s in stocks:
                ind = s.industry or "其他"
                if ind not in industry_map:
                    industry_map[ind] = {}
                industry_map[ind].setdefault(s.consecutive, []).append(s)

            industry_ladders = []
            for sector, by_board in industry_map.items():
                total_in_sector = sum(len(v) for v in by_board.values())
                industry_ladders.append({
                    'sector': sector,
                    'total': total_in_sector,
                    'stocks_by_board': dict(sorted(by_board.items(), key=lambda x: x[0], reverse=True)),
                })
            # 按涨停数量降序，取 Top 5
            industry_ladders.sort(key=lambda x: x['total'], reverse=True)
            industry_ladders = industry_ladders[:5]

            result = LimitUpLadder(
                total=len(stocks),
                consecutive_stats=consecutive_stats,
                height_leaders=height_leaders,
                industry_ladders=industry_ladders,
                concept_ladders=[],
            )
            logger.info(f"[连板] 天梯构建完成: {len(stocks)}只, {len(height_leaders)}只高度板, "
                       f"{len(industry_ladders)}个行业板块")
            return result

        except Exception as e:
            logger.warning(f"[连板] 构建天梯失败: {e}")
            return None

    def _format_limit_up_section(self, ladeder: LimitUpLadder) -> str:
        """生成连板情绪 + 天梯 Markdown 文本，注入到复盘报告。"""
        if ladeder is None:
            return "\n### 连板情绪\n今日涨停数据暂不可用。\n"
        if ladeder.total == 0:
            return "\n### 连板情绪\n今日无涨停数据。\n"

        lines = ["\n### 连板情绪\n"]

        # 梯度统计
        parts = []
        for k in sorted(ladeder.consecutive_stats.keys(), reverse=True):
            cnt = ladeder.consecutive_stats[k]
            if k == 1:
                label = "首板"
            elif k >= 5:
                label = f"{k}板及以上"
            else:
                label = f"{k}连板"
            parts.append(f"{label}{cnt}只")
        lines.append(f"今日涨停{ladeder.total}只，" + "、".join(parts) + "。")

        # 高度板龙头
        if ladeder.height_leaders:
            leaders_str = "、".join(f"{s.name}({s.consecutive}连板)" for s in ladeder.height_leaders[:5])
            lines.append(f"高度板龙头：{leaders_str}。")

        # 情绪判断
        if ladeder.height_leaders and ladeder.height_leaders[0].consecutive >= 5:
            lines.append("短线情绪强势，赚钱效应扩散。")
        elif ladeder.height_leaders and ladeder.height_leaders[0].consecutive >= 3:
            lines.append("短线情绪偏暖，连板梯队完整。")
        else:
            lines.append("短线情绪偏弱，连板高度有限。")

        # 天梯（涨停<10家跳过）
        if ladeder.total < 10:
            lines.append("\n*涨停数不足10家，跳过连板天梯明细。*")
            return "\n".join(lines) + "\n"

        # ---- 行业天梯 ----
        if ladeder.industry_ladders:
            lines.append("\n### 连板天梯 · 行业板块\n")
            for ladder in ladeder.industry_ladders:
                lines.append(f"**{ladder['sector']}**（涨停{ladder['total']}只）\n")
                for board, stock_list in sorted(ladder['stocks_by_board'].items(), key=lambda x: x[0], reverse=True):
                    if board == 1:
                        label = "首板"
                    else:
                        label = f"{board}板"
                    names = "、".join(s.name for s in stock_list[:5])
                    if len(stock_list) > 5:
                        names += f"等{len(stock_list)}只"
                    lines.append(f"  - {label}: {names}\n")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _build_limit_up_prompt_block(ladder: Optional[LimitUpLadder]) -> str:
        """构建 Prompt 中的连板数据块（供 LLM 分析参考）。"""
        if ladder is None:
            return ""
        if ladder.total == 0:
            return ""
        lines = ["## 连板数据"]
        # 梯度概览
        parts = []
        for k in sorted(ladder.consecutive_stats.keys(), reverse=True):
            cnt = ladder.consecutive_stats[k]
            label = f"{k}连板" if k > 1 else "首板"
            parts.append(f"{label}{cnt}只")
        lines.append("- 涨停" + "，".join(parts))
        if ladder.height_leaders:
            lines.append("- 高度板: " + "、".join(
                f"{s.name}({s.consecutive}板)" for s in ladder.height_leaders[:5]))
        # 行业分布
        if ladder.industry_ladders:
            ind_parts = []
            for il in ladder.industry_ladders[:5]:
                top_board = max(il['stocks_by_board'].keys())
                ind_parts.append(f"{il['sector']}({il['total']}只/最高{top_board}板)")
            lines.append("- 涨停集中行业: " + "；".join(ind_parts))
        lines.append("")
        return "\n".join(lines) + "\n"

    def _inject_limit_up_section(self, review: str, limit_up_section: str) -> str:
        """将连板天梯段落注入到四、资金与情绪段之后。"""
        import re
        # 匹配 "### 四、资金与情绪" 或 "### 四、资金与情绪\n..."
        pattern = r"(### 四、资金与情绪.*?)(?=\n### 五|\n### \S|\Z)"
        match = re.search(pattern, review, re.DOTALL)
        if match:
            idx = match.end()
            # 找到该段落后下一个 ## 或 ### 标题之前插入
            next_heading = re.search(r"\n###?\s", review[idx:])
            if next_heading:
                insert_pos = idx + next_heading.start()
            else:
                insert_pos = len(review)
            return review[:insert_pos] + "\n\n---\n\n" + limit_up_section + "\n" + review[insert_pos:]
        # 没找到资金与情绪段，直接追加到末尾
        return review + "\n\n---\n\n" + limit_up_section + "\n"

    def search_market_news(self) -> List[Dict]:
        """
        搜索市场新闻
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning("[大盘] 搜索服务未配置，跳过新闻搜索")
            return []
        
        all_news = []

        # 按 region 使用不同的新闻搜索词
        search_queries = self.profile.news_queries
        
        try:
            logger.info("[大盘] 开始搜索市场新闻...")
            
            # 根据 region 设置搜索上下文名称，避免美股搜索被解读为 A 股语境
            market_names = {"cn": "大盘", "us": "US market", "hk": "HK market"}
            market_name = market_names.get(self.region, "大盘")
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(f"[大盘] 搜索 '{query}' 获取 {len(response.results)} 条结果")
            
            logger.info(f"[大盘] 共获取 {len(all_news)} 条市场新闻")
            
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[大盘] AI分析器未配置或不可用，使用模板生成报告")
            return self._generate_template_review(overview, news)

        # 获取连板天梯数据
        limit_up_ladder = None
        limit_up_section = ""
        try:
            limit_up_ladder = self._build_limit_up_ladder()
            if limit_up_ladder is not None:
                limit_up_section = self._format_limit_up_section(limit_up_ladder)
        except Exception as e:
            logger.warning(f"[大盘] 连板数据获取失败，跳过: {e}")

        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news, limit_up_ladder)

        logger.info("[大盘] 调用大模型生成复盘报告...")
        # Use the public generate_text() entry point — never access private analyzer attributes.
        review = self.analyzer.generate_text(prompt, max_tokens=8192, temperature=0.7)

        if review:
            logger.info("[大盘] 复盘报告生成成功，长度: %d 字符", len(review))
            # Inject structured data tables into LLM prose sections
            review = self._inject_data_into_review(review, overview, news)
            # 注入连板天梯到资金与情绪段之后
            if limit_up_section:
                review = self._inject_limit_up_section(review, limit_up_section)
            return review
        else:
            logger.warning("[大盘] 大模型返回为空，使用模板报告")
            return self._generate_template_review(overview, news)
    
    def _inject_data_into_review(
        self,
        review: str,
        overview: MarketOverview,
        news: Optional[List] = None,
    ) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        # Build data blocks
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        news_block = self._build_news_block(news or [])
        patterns = (
            _ENGLISH_SECTION_PATTERNS
            if self._get_review_language() == "en"
            else _CHINESE_SECTION_PATTERNS
        )

        if stats_block:
            review = self._insert_after_section(
                review,
                patterns["market_summary"],
                stats_block,
            )

        if indices_block:
            review = self._insert_after_section(
                review,
                patterns["index_commentary"],
                indices_block,
            )

        if sector_block:
            review = self._insert_after_section(
                review,
                patterns["sector_highlights"],
                sector_block,
            )

        if news_block and "news_catalysts" in patterns:
            review = self._insert_after_section(
                review,
                patterns["news_catalysts"],
                news_block,
            )

        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading — append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing and visual separator
        return text[:insert_pos].rstrip() + '\n\n---\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        if self._get_review_language() == "en":
            light = self.build_market_light_snapshot(overview)
            return "\n".join(
                [
                    f"> **Market Light**: {light['status']} ({light['label']}) | "
                    f"**{light['score']}/100** {self._build_temperature_bar(light['score'])}",
                    f"> **Reasons**: {'; '.join(light['reasons'])}",
                    f"> **Guidance**: {light['guidance']}",
                    "",
                    f"> 📈 Advancers **{overview.up_count}** / Decliners **{overview.down_count}** / "
                    f"Flat **{overview.flat_count}** | "
                    f"Limit-up **{overview.limit_up_count}** / Limit-down **{overview.limit_down_count}** | "
                    f"Turnover **{overview.total_amount:.0f}** ({self._get_turnover_unit_label()})",
                ]
            )
        light = self.build_market_light_snapshot(overview)
        score, label = light["score"], light["temperature_label"]
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else 0.0
        limit_spread = overview.limit_up_count - overview.limit_down_count
        lines = [
            f"> **大盘红绿灯**：{light['status']}（{light['label']}） | **{score}/100** {self._build_temperature_bar(score)}",
            f"> **核心原因**：{'；'.join(light['reasons'])}",
            f"> **操作建议**：{light['guidance']}",
            "",
            "| 指标 | 数值 | 观察 |",
            "|------|------|------|",
            f"| 上涨/下跌/平盘 | {overview.up_count} / {overview.down_count} / {overview.flat_count} | 上涨占比(不含平盘) {up_ratio:.1%} |",
            f"| 涨停/跌停 | {overview.limit_up_count} / {overview.limit_down_count} | 涨跌停差 {limit_spread:+d} |",
            f"| 两市成交额 | {overview.total_amount:.0f} 亿 | {self._describe_turnover(overview.total_amount)} |",
        ]
        return "\n".join(lines)

    def build_market_light_snapshot(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build a deterministic market-light snapshot from structured breadth data."""
        score, temperature_label = self._build_market_temperature(overview)
        if score >= 60:
            status = "green"
        elif score >= 40:
            status = "yellow"
        else:
            status = "red"

        if self._get_review_language() == "en":
            label_map = {
                "green": "constructive",
                "yellow": "watch",
                "red": "defensive",
            }
            guidance_map = {
                "green": "Risk appetite is acceptable; focus on leading themes and position discipline.",
                "yellow": "Signals are mixed; keep position sizing moderate and wait for confirmation.",
                "red": "Risk is elevated; prioritize drawdown control and avoid chasing weak rebounds.",
            }
            reasons = self._build_market_light_reasons_en(overview, score)
        else:
            label_map = {
                "green": "可进攻",
                "yellow": "需观察",
                "red": "偏防守",
            }
            guidance_map = {
                "green": "风险偏好尚可，关注主线延续与仓位纪律。",
                "yellow": "信号分化，控制仓位并等待量价确认。",
                "red": "风险偏高，优先控制回撤，避免追高弱反弹。",
            }
            reasons = self._build_market_light_reasons_zh(overview, score)

        return {
            "status": status,
            "label": label_map[status],
            "score": score,
            "temperature_label": temperature_label,
            "reasons": reasons,
            "guidance": guidance_map[status],
        }

    def _build_market_light_reasons_zh(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = [f"盘面温度 {score}/100"]
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，赚钱效应扩散")
            elif up_ratio <= 0.4:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，亏钱效应较强")
            else:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，市场分化")
        if overview.indices:
            avg_change = sum(idx.change_pct for idx in overview.indices) / len(overview.indices)
            reasons.append(f"主要指数平均涨跌幅 {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"涨跌停差 {overview.limit_up_count - overview.limit_down_count:+d}")
        return reasons[:4]

    def _build_market_light_reasons_en(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = [f"market temperature {score}/100"]
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is expanding")
            elif up_ratio <= 0.4:
                reasons.append(f"advancers ratio {up_ratio:.0%}, downside pressure dominates")
            else:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is mixed")
        if overview.indices:
            avg_change = sum(idx.change_pct for idx in overview.indices) / len(overview.indices)
            reasons.append(f"average major-index change {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"limit-up/down spread {overview.limit_up_count - overview.limit_down_count:+d}")
        return reasons[:4]

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """构建指数行情表格"""
        if not overview.indices:
            return ""
        if self._get_review_language() == "en":
            lines = [
                f"| Index | Last | Change % | Open | High | Low | Amplitude | Turnover ({self._get_turnover_unit_label()}) |",
                "|-------|------|----------|------|------|-----|-----------|-----------------|",
            ]
        else:
            lines = [
                "| 指数 | 最新 | 涨跌幅 | 开盘 | 最高 | 最低 | 振幅 | 成交额(亿) |",
                "|------|------|--------|------|------|------|------|-----------|",
            ]
        for idx in overview.indices:
            arrow = "🔴" if idx.change_pct > 0 else "🟢" if idx.change_pct < 0 else "⚪"
            amount_raw = idx.amount or 0.0
            amount_str = self._format_turnover_value(amount_raw)
            lines.append(
                f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | "
                f"{self._format_optional_number(idx.open)} | {self._format_optional_number(idx.high)} | "
                f"{self._format_optional_number(idx.low)} | {self._format_optional_pct(idx.amplitude)} | {amount_str} |"
            )
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build sector ranking block."""
        if not overview.top_sectors and not overview.bottom_sectors:
            return ""
        lines = []
        if overview.top_sectors:
            if self._get_review_language() == "en":
                lines.extend([
                    "#### Leading Sectors",
                    "| Rank | Sector | Change |",
                    "|------|--------|--------|",
                ])
            else:
                lines.extend([
                    "#### 领涨板块 Top 5",
                    "| 排名 | 板块 | 涨跌幅 |",
                    "|------|------|--------|",
                ])
            for rank, sector in enumerate(overview.top_sectors[:5], 1):
                lines.append(
                    f"| {rank} | {sector.get('name', '-')} | {self._format_signed_pct(sector.get('change_pct'))} |"
                )
        if overview.bottom_sectors:
            if lines:
                lines.append("")
            if self._get_review_language() == "en":
                lines.extend([
                    "#### Lagging Sectors",
                    "| Rank | Sector | Change |",
                    "|------|--------|--------|",
                ])
            else:
                lines.extend([
                    "#### 领跌板块 Top 5",
                    "| 排名 | 板块 | 涨跌幅 |",
                    "|------|------|--------|",
                ])
            for rank, sector in enumerate(overview.bottom_sectors[:5], 1):
                lines.append(
                    f"| {rank} | {sector.get('name', '-')} | {self._format_signed_pct(sector.get('change_pct'))} |"
                )
        return "\n".join(lines)

    def _build_news_block(self, news: List) -> str:
        """Build a compact news catalyst list (no table — wraps naturally in DingTalk)."""
        if not news:
            return ""
        if self._get_review_language() == "en":
            lines = ["#### News Catalysts"]
        else:
            lines = ["#### 近三日催化线索"]

        for idx, item in enumerate(news[:5], 1):
            if hasattr(item, "title"):
                title = getattr(item, "title", "") or "-"
                snippet = getattr(item, "snippet", "") or ""
            else:
                title = item.get("title", "-") or "-"
                snippet = item.get("snippet", "") or ""
            title = str(title).strip()[:60]
            snippet = str(snippet).strip().replace("\n", " ")[:120] or "-"
            lines.append(f"{idx}. **{title}** — {snippet}")
        return "\n".join(lines)

    @staticmethod
    def _format_optional_number(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}"

    @staticmethod
    def _format_optional_pct(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}%"

    @staticmethod
    def _format_signed_pct(value: Any) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric_value:+.2f}%"

    @staticmethod
    def _escape_table_cell(value: str) -> str:
        return value.replace("|", "\\|")

    @staticmethod
    def _build_temperature_bar(score: int) -> str:
        filled = max(0, min(10, round(score / 10)))
        return "█" * filled + "░" * (10 - filled)

    @staticmethod
    def _describe_turnover(total_amount: float) -> str:
        if total_amount >= 15000:
            return "高活跃度"
        if total_amount >= 9000:
            return "中等活跃"
        if total_amount > 0:
            return "缩量观望"
        return "暂无数据"

    def _build_market_temperature(self, overview: MarketOverview) -> tuple[int, str]:
        participants = overview.up_count + overview.down_count
        breadth_score = 50
        if participants:
            breadth_score = int(overview.up_count / participants * 100)

        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        index_score = 50
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            index_score = int(max(0, min(100, 50 + avg_change * 12)))

        limit_total = overview.limit_up_count + overview.limit_down_count
        limit_score = 50
        if limit_total:
            limit_score = int(overview.limit_up_count / limit_total * 100)

        score = int(round(breadth_score * 0.45 + index_score * 0.35 + limit_score * 0.20))
        if self._get_review_language() == "en":
            if score >= 70:
                label = "risk-on"
            elif score >= 55:
                label = "constructive"
            elif score >= 40:
                label = "mixed"
            else:
                label = "defensive"
        else:
            if score >= 70:
                label = "强势"
            elif score >= 55:
                label = "偏暖"
            elif score >= 40:
                label = "震荡"
            else:
                label = "偏弱"
        return score, label

    def _build_review_prompt(self, overview: MarketOverview, news: List, limit_up_ladder: Optional[LimitUpLadder] = None) -> str:
        """构建复盘报告 Prompt"""
        review_language = self._get_review_language()

        # 指数行情信息（简洁格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])
        
        # 新闻信息 - 支持 SearchResult 对象或字典
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # 兼容 SearchResult 对象和字典
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"
        
        # 按 region 组装市场概况与板块区块（美股无涨跌家数、板块数据）
        stats_block = ""
        sector_block = ""
        if review_language == "en":
            if self.profile.has_market_stats:
                stats_block = f"""## Market Breadth
- Advancers: {overview.up_count} | Decliners: {overview.down_count} | Flat: {overview.flat_count}
- Limit-up: {overview.limit_up_count} | Limit-down: {overview.limit_down_count}
- Turnover: {overview.total_amount:.0f} ({self._get_turnover_unit_label()})"""
            else:
                stats_block = "## Market Breadth\n(No equivalent advance/decline statistics are available for this market.)"

            if self.profile.has_sector_rankings:
                sector_block = f"""## Sector Performance
Leading: {top_sectors_text if top_sectors_text else "N/A"}
Lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}"""
            else:
                sector_block = "## Sector Performance\n(Sector data not available for this market.)"
        else:
            if self.profile.has_market_stats:
                stats_block = f"""## 市场概况
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元"""
            else:
                stats_block = "## 市场概况\n（该市场暂无涨跌家数等统计）"

            if self.profile.has_sector_rankings:
                sector_block = f"""## 板块表现
领涨: {top_sectors_text if top_sectors_text else "暂无数据"}
领跌: {bottom_sectors_text if bottom_sectors_text else "暂无数据"}"""
            else:
                sector_block = "## 板块表现\n（该市场暂无板块涨跌数据）"

        data_no_indices_hint = (
            "注意：由于行情数据获取失败，请主要根据【市场新闻】进行定性分析和总结，不要编造具体的指数点位。"
            if not indices_text
            else ""
        )
        if review_language == "en":
            data_no_indices_hint = (
                "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels."
                if not indices_text
                else ""
            )
            indices_placeholder = indices_text if indices_text else "No index data (API error)"
            news_placeholder = news_text if news_text else "No relevant news"
        else:
            indices_placeholder = indices_text if indices_text else "暂无指数数据（接口异常）"
            news_placeholder = news_text if news_text else "暂无相关新闻"

        if review_language == "en":
            report_title = self._get_review_title(overview.date).removeprefix("## ").strip()
            return f"""You are a professional US/A/H market analyst. Please produce a concise market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)
- The entire fixed shell, headings, guidance, and conclusion must be in English

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

{self._build_limit_up_prompt_block(limit_up_ladder)}

## Market News
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# Output Template (follow this structure)

## {report_title}

### 1. Market Summary
(2-3 sentences summarizing overall market tone, index moves, and liquidity.)

### 2. Index Commentary
({self._get_index_hint()})

### 3. Fund Flows
(Interpret what turnover, participation, and flow signals imply.)

### 4. Sector Highlights
(Analyze the drivers behind the leading and lagging sectors or themes.)

### 5. Outlook
(Provide the near-term outlook based on price action and news.)

### 6. Risk Alerts
(List the main risks to monitor.)

### 7. Strategy Plan
(Provide an offensive/balanced/defensive stance, a position-sizing guideline, one invalidation trigger, and end with “For reference only, not investment advice.”)

---

Output the report content directly, no extra commentary.
"""

        # A 股场景使用中文提示语
        return f"""你是一位专业的A/H/美股市场分析师，请根据以下数据生成一份结构化的{self._get_market_scope_name('zh')}大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）
- 报告要像交易员盘后工作台：先给结论，再按数据表、主线、催化、计划展开
- 不要重复列出已由系统注入的表格数据；正文负责解释表格背后的含义

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_placeholder}

{stats_block}

{sector_block}

{self._build_limit_up_prompt_block(limit_up_ladder)}

## 市场新闻
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# 输出格式模板（请严格按此格式输出，每条指引都要有对应内容）

## {overview.date} 大盘复盘

> 一句话定调：今日市场状态 + 核心矛盾 + 明日最需关注的一个变量。

### 一、盘面总览
（2-3句：指数方向、涨跌结构、量能水平，明确给出”强势/偏暖/震荡/偏弱”定性）

### 二、指数结构
（{self._get_index_hint()}，点明谁护盘谁拖累，给出具体支撑位/压力位数字）

### 三、板块主线
（领涨方向+驱动逻辑+持续性评估；领跌方向+扩散风险。必须明确回答：有主线吗？主线是谁？）

### 四、资金与情绪
（量能信号、涨跌停结构、高标股状态，给情绪定性：亢奋/偏暖/中性/降温/恐慌）

### 五、消息催化
（只列对明日交易有实质影响的催化，无实质影响则写”今日无重大催化”）

### 六、明日交易计划
- **结论**：进攻/均衡/防守，一句话理由
- **仓位**：建议几成仓
- **关注**：2-3个方向
- **回避**：1-2个方向
- **失效条件**：触发什么信号则立即转防守

### 七、风险提示
（2-3条具体风险，最后一行：”建议仅供参考，不构成投资建议。”）

### 八、策略总结
（今日1条核心教训 + 1条可操作优化建议，每条≤20字，精炼不废话）

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成复盘报告（无大模型时的备选方案）"""
        template_language = self._get_template_review_language()
        mood_code = self.profile.mood_index_code
        # 根据 mood_index_code 查找对应指数
        # cn: mood_code="000001"，idx.code 可能为 "sh000001"（以 mood_code 结尾）
        # us: mood_code="SPX"，idx.code 直接为 "SPX"
        mood_index = next(
            (
                idx
                for idx in overview.indices
                if idx.code == mood_code or idx.code.endswith(mood_code)
            ),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = self._get_market_mood_text("strong_up", template_language)
            elif mood_index.change_pct > 0:
                market_mood = self._get_market_mood_text("mild_up", template_language)
            elif mood_index.change_pct > -1:
                market_mood = self._get_market_mood_text("mild_down", template_language)
            else:
                market_mood = self._get_market_mood_text("strong_down", template_language)
        else:
            market_mood = self._get_market_mood_text("range", template_language)
        
        # 指数行情（简洁格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        separator = ", " if template_language == "en" else "、"
        top_text = separator.join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = separator.join([s['name'] for s in overview.bottom_sectors[:3]])

        if template_language == "en":
            stats_section = ""
            if self.profile.has_market_stats:
                stats_section = f"""
### 3. Breadth & Liquidity
| Metric | Value |
|--------|-------|
| Advancers | {overview.up_count} |
| Decliners | {overview.down_count} |
| Limit-up | {overview.limit_up_count} |
| Limit-down | {overview.limit_down_count} |
| Turnover ({self._get_turnover_unit_label()}) | {overview.total_amount:.0f} |
"""
            sector_section = ""
            if self.profile.has_sector_rankings and (top_text or bottom_text):
                sector_section = f"""
### 4. Sector Highlights
- **Leaders**: {top_text or "N/A"}
- **Laggards**: {bottom_text or "N/A"}
"""
            market_names = {"us": "US Market Recap", "hk": "HK Market Recap"}
            market_name = market_names.get(self.region, "A-share Market Recap")
            report = f"""## {overview.date} {market_name}

### 1. Market Summary
Today's {self._get_market_scope_name(template_language)} showed **{market_mood}**.

### 2. Major Indices
{indices_text or "- No index data available"}
{stats_section}
{sector_section}
### 5. Risk Alerts
Market conditions can change quickly. The data above is for reference only and does not constitute investment advice.

{self._get_strategy_markdown_block(template_language)}

---
*Review Time: {datetime.now().strftime('%H:%M')}*
"""
            return report

        market_labels = {"cn": "A股", "us": "美股", "hk": "港股"}
        market_label = market_labels.get(self.region, "A股")
        dashboard_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        return f"""## {overview.date} 大盘复盘

> 今日{market_label}市场整体呈现**{market_mood}**态势，优先观察指数承接、成交额变化和板块持续性。

### 一、盘面总览
{dashboard_block or "暂无市场宽度数据。"}

### 二、指数结构
{indices_block or indices_text or "暂无指数数据。"}

### 三、板块主线
{sector_block or "- 暂无板块涨跌榜数据。"}

### 四、资金与情绪
- 结合成交额和涨跌家数看，当前更适合等待确认，避免仅凭单一热点追高。

### 五、消息催化
- 暂无可用新闻时，应降低对题材持续性的确定性判断。

### 六、明日交易计划
- **结论**：均衡观察。
- **仓位**：控制在中性区间，等待指数与主线共振。
- **关注方向**：{top_text or "强于指数的主线板块"}。
- **回避方向**：{bottom_text or "连续走弱且缺少修复信号的方向"}。

### 七、风险提示
- 市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
    
    def run_daily_review(self) -> str:
        """
        执行每日大盘复盘流程
        
        Returns:
            复盘报告文本
        """
        logger.info("========== 开始大盘复盘分析 ==========")
        
        # 1. 获取市场概览
        overview = self.get_market_overview()
        
        # 2. 搜索市场新闻
        news = self.search_market_news()
        
        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        
        logger.info("========== 大盘复盘分析完成 ==========")

        return report

    def run_chan_review(self) -> str:
        """
        执行缠论分析（上证指数 + Top 3 热门板块），独立推送。

        Returns:
            缠论分析报告 Markdown 文本
        """
        import akshare as ak
        from datetime import datetime as _dt, timedelta as _td
        import numpy as np

        logger.info("========== 开始缠论分析 ==========")

        def _calc_macd(close_series, fast=12, slow=26, signal=9):
            """简易 MACD 计算"""
            ema_fast = close_series.ewm(span=fast, adjust=False).mean()
            ema_slow = close_series.ewm(span=slow, adjust=False).mean()
            dif = ema_fast - ema_slow
            dea = dif.ewm(span=signal, adjust=False).mean()
            hist = (dif - dea) * 2
            return dif, dea, hist

        def _describe_trend(df, name):
            """描述均线排列和 MACD 状态"""
            last = df.iloc[-1]
            ma5 = last['ma5']
            ma10 = last['ma10']
            ma20 = last['ma20']
            ma60 = last['ma60']

            # 均线排列
            if pd.isna(ma60):
                ma60 = ma20  # fallback
            if ma5 > ma10 > ma20 > ma60:
                alignment = "多头排列"
            elif ma5 < ma10 < ma20 < ma60:
                alignment = "空头排列"
            else:
                alignment = "缠绕"

            # MACD
            last_dif = last.get('dif', 0)
            last_dea = last.get('dea', 0)
            last_hist = last.get('macd_hist', 0)
            prev_hist = 0
            if len(df) >= 2:
                prev_hist = df.iloc[-2].get('macd_hist', 0)

            if last_dif > last_dea:
                macd_state = "金叉（多头）"
            else:
                macd_state = "死叉（空头）"

            # 背驰判断
            divergence = ""
            if last_hist < prev_hist and last_hist > 0:
                divergence = "，红柱缩短（潜在顶背驰）"
            elif last_hist > prev_hist and last_hist < 0:
                divergence = "，绿柱收窄（潜在底背驰）"

            # 近5日K线概要
            recent = df.tail(5)
            k_summary = ", ".join(
                f"{r['date'].strftime('%m-%d'):} {r['close']:.1f}" for _, r in recent.iterrows()
            ) if not recent.empty else "无数据"

            return {
                'name': name,
                'close': last['close'],
                'ma5': f"{ma5:.1f}",
                'ma10': f"{ma10:.1f}",
                'ma20': f"{ma20:.1f}",
                'ma60': f"{ma60:.1f}" if not pd.isna(ma60) else "N/A",
                'alignment': alignment,
                'dif': f"{last_dif:.2f}",
                'dea': f"{last_dea:.2f}",
                'macd_state': macd_state + divergence,
                'k_summary': k_summary,
                'high_10d': f"{df.tail(10)['high'].max():.1f}",
                'low_10d': f"{df.tail(10)['low'].min():.1f}",
                'high_30d': f"{df.tail(30)['high'].max():.1f}",
                'low_30d': f"{df.tail(30)['low'].min():.1f}",
            }

        chan_data = []

        # ---- 1. 上证指数 ----
        try:
            df_ss = ak.stock_zh_index_daily(symbol='sh000001')
            if df_ss is not None and not df_ss.empty:
                df_ss = df_ss.rename(columns={'date': 'date', 'open': 'open', 'close': 'close',
                                               'high': 'high', 'low': 'low', 'volume': 'volume'})
                df_ss['date'] = pd.to_datetime(df_ss['date'])
                df_ss = df_ss.sort_values('date').tail(80).copy()

                # 如果最新日期不是今天，用腾讯行情补上今日实时价
                today_dt = _dt.now()
                last_date = df_ss['date'].iloc[-1]
                if last_date.date() < today_dt.date():
                    try:
                        import requests
                        resp = requests.get('http://qt.gtimg.cn/q=sh000001', timeout=10)
                        resp.encoding = 'gbk'
                        parts = resp.text.split('~')
                        if len(parts) > 35:
                            today_price = float(parts[3])
                            today_open = float(parts[5]) if parts[5] else today_price
                            today_high = float(parts[33]) if parts[33] else today_price
                            today_low = float(parts[34]) if parts[34] else today_price
                            today_vol = float(parts[36]) if len(parts) > 36 and parts[36] else (
                                float(parts[6]) if parts[6] else 0)
                            if today_price > 0:
                                df_ss = pd.concat([df_ss, pd.DataFrame([{
                                    'date': pd.Timestamp(today_dt.date()),
                                    'open': today_open, 'close': today_price,
                                    'high': today_high, 'low': today_low,
                                    'volume': today_vol,
                                }])], ignore_index=True)
                                logger.info(f"[缠论] 已补充今日实时价: {today_price:.2f}")
                    except Exception as e:
                        logger.warning(f"[缠论] 补今日实时价失败: {e}")

                df_ss['ma5'] = df_ss['close'].rolling(5).mean()
                df_ss['ma10'] = df_ss['close'].rolling(10).mean()
                df_ss['ma20'] = df_ss['close'].rolling(20).mean()
                df_ss['ma60'] = df_ss['close'].rolling(60).mean()
                dif, dea, hist = _calc_macd(df_ss['close'])
                df_ss['dif'] = dif
                df_ss['dea'] = dea
                df_ss['macd_hist'] = hist
                chan_data.append(_describe_trend(df_ss, '上证指数'))
                logger.info(f"[缠论] 上证指数: {len(df_ss)} 条日线数据 (最新: {df_ss['date'].iloc[-1].date()})")
        except Exception as e:
            logger.warning(f"[缠论] 上证指数获取失败: {e}")

        # ---- 2. 热门板块（同花顺数据，按成交额 Top 5）----
        sector_data = []
        hot_sectors = []
        try:
            df_ths = ak.stock_board_industry_summary_ths()
            if df_ths is not None and not df_ths.empty:
                name_col = '板块'
                amt_col = '总成交额'
                df_ths[amt_col] = pd.to_numeric(df_ths[amt_col], errors='coerce')
                df_valid = df_ths.dropna(subset=[amt_col])
                top5_hot = df_valid.nlargest(5, amt_col)
                for _, row in top5_hot.iterrows():
                    hot_sectors.append(row[name_col])
                logger.info(f"[缠论] 同花顺热门板块: {hot_sectors}")
        except Exception as e:
            logger.warning(f"[缠论] 同花顺板块排行获取失败: {e}")

        # 同花顺行业K线拉取
        all_sector_names = hot_sectors
        for sec_name in all_sector_names:
            if sec_name in {d['name'] for d in sector_data}:
                continue
            try:
                df_sec = ak.stock_board_industry_index_ths(
                    symbol=sec_name,
                    start_date=(_dt.now() - _td(days=120)).strftime('%Y%m%d'),
                    end_date=_dt.now().strftime('%Y%m%d'),
                )
                if df_sec is not None and not df_sec.empty:
                    df_sec = df_sec.rename(columns={
                        '日期': 'date', '开盘价': 'open', '收盘价': 'close',
                        '最高价': 'high', '最低价': 'low', '成交量': 'volume',
                    })
                    df_sec['date'] = pd.to_datetime(df_sec['date'])
                    df_sec = df_sec.sort_values('date').tail(80).copy()
                    df_sec['ma5'] = df_sec['close'].rolling(5).mean()
                    df_sec['ma10'] = df_sec['close'].rolling(10).mean()
                    df_sec['ma20'] = df_sec['close'].rolling(20).mean()
                    df_sec['ma60'] = df_sec['close'].rolling(60).mean()
                    dif, dea, hist = _calc_macd(df_sec['close'])
                    df_sec['dif'] = dif
                    df_sec['dea'] = dea
                    df_sec['macd_hist'] = hist
                    sector_desc = _describe_trend(df_sec, sec_name)
                    sector_data.append(sector_desc)
                    chan_data.append(sector_desc)
                    logger.info(f"[缠论] 板块 {sec_name}: {len(df_sec)} 条日线数据")
            except Exception as e:
                logger.warning(f"[缠论] 板块 {sec_name} 获取失败: {e}")

        # ---- 3. 构建 Prompt ----
        if not chan_data:
            logger.warning("[缠论] 无可用数据")
            return "今日缠论数据暂不可用。"

        today = _dt.now().strftime('%Y-%m-%d')
        buf = ["# 缠论技术分析数据\n"]
        for d in chan_data:
            tag = "🔥热门" if d['name'] in hot_sectors else ""
            buf.append(f"## {d['name']} {tag}")
            buf.append(f"- 收盘价: {d['close']:.2f}")
            buf.append(f"- 均线: MA5={d['ma5']}, MA10={d['ma10']}, MA20={d['ma20']}, MA60={d['ma60']}（{d['alignment']}）")
            buf.append(f"- MACD: DIF={d['dif']}, DEA={d['dea']}（{d['macd_state']}）")
            buf.append(f"- 近5日K线: {d['k_summary']}")
            buf.append(f"- 近10日高低: {d['low_10d']} ~ {d['high_10d']}")
            buf.append(f"- 近30日高低: {d['low_30d']} ~ {d['high_30d']}")
            buf.append("")

        chan_data_block = "\n".join(buf)

        hot_list = "\n".join(f"  - {n}" for n in hot_sectors) if hot_sectors else "（无数据）"

        prompt = f"""你是一位精通缠中说禅（缠论）的技术分析师。请根据下方提供的上证指数和热门板块（按今日成交额排序）的日线数据，输出一份纯 Markdown 格式的缠论分析报告。

【要求】
- 纯 Markdown，禁止 JSON/代码块
- 对每个标的识别：当前笔/线段方向、中枢区间、背驰状态
- 给出关键点位：中枢上沿、中枢下沿、支撑位、压力位
- 对每个标的做走势完全分类（强/中/弱三种路径），每种给出边界条件和操作建议
- 语言简洁，像交易员早盘笔记

---

{chan_data_block}

---

# 输出格式

## 缠论早盘 · 行业趋势

> 一句话总览今日市场缠论状态 + 热门板块轮动方向

### 上证指数
- **当前结构**:
- **关键点位**: 中枢上沿/下沿/支撑/压力
- **走势分类**: 强/中/弱

### 热门板块（按成交额 Top 5）

{hot_list}

（对每个板块：当前结构+关键点位+走势分类，重点关注二买/三买机会）

---
请直接输出报告。
"""

        # ---- 4. LLM 生成 ----
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[缠论] AI分析器不可用")
            return "AI 分析器未配置，无法生成缠论报告。"

        logger.info("[缠论] 调用大模型生成报告...")
        review = self.analyzer.generate_text(prompt, max_tokens=4096, temperature=0.3)
        if review:
            logger.info("[缠论] 报告生成成功，长度: %d 字符", len(review))
            logger.info("========== 缠论分析完成 ==========")
            return review
        else:
            logger.warning("[缠论] 大模型返回为空")
            return "缠论分析生成失败。"


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 测试获取市场概览
    overview = analyzer.get_market_overview()
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")
    
    # 测试生成模板报告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)
