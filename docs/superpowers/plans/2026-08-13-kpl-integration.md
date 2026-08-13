# KPL 接口集成 + 推送优化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入 KPL（开盘啦）私有接口替换/补强大盘复盘的脆弱数据源，新增概念天梯/情绪指标/新高+大面股段落，瘦身新闻段，并修复钉钉消息折叠。

**Architecture:** 新建 `data_provider/kpl_client.py` 封装 KPL 私有接口（单入口 `_post` + 解析函数 + 熔断），大盘复盘各数据点在 `market_analyzer.py` 处优先走 KPL、失败回退现有 akshare 链；新增段落渲染为 Markdown 注入 LLM 输出；钉钉分片改为配置化分片大小。全量 KPL_ENABLED=false 可一键回退。

**Tech Stack:** Python 3 / requests / akshare / unittest.mock / pytest

## Global Constraints

- 遵循 `AGENTS.md` 硬规则：不加 `Co-Authored-By`；不写死密钥（DeviceID 用文档公开演示值，可配置覆盖）。
- **Commit 规则（AGENTS.md 硬规则）**：未经用户明确确认，不执行 `git commit`。本计划中每个任务末尾的 Commit 步骤为**待确认项**——执行时先跳过提交，完成 1-2 个任务后向用户请求一次提交确认，获批后再执行。
- 改配置项必须同步 `.env.example` 与 `docs/CHANGELOG.md`（`[Unreleased]` 扁平格式，禁止 `### 类目标题`）。
- KPL 是大陆服务，`longhuvip.com` 必须直连（加入 NO_PROXY），禁止走海外代理。
- 每个 KPL 抓取函数独立 try/except；KPL 失败静默回退 akshare，报告不报错；连续失败 3 次熔断。
- 验收验证：`./scripts/ci_gate.sh` + `python -m pytest -m "not network"`。
- 设计文档：`docs/superpowers/specs/2026-08-13-kpl-integration-design.md`（实现细节以此计划为准，计划是设计文档的细化与澄清）。

---

### Task 1: 配置项 + KPLClient 基础模块

**Files:**
- Modify: `src/config.py`（dataclass 字段 + `_load_from_env` + `domestic_domains`）
- Create: `data_provider/kpl_client.py`
- Test: `tests/test_kpl_client.py`
- Modify: `.env.example`、`docs/CHANGELOG.md`

**Interfaces:**
- Consumes: `Config` dataclass（`kpl_enabled`/`kpl_timeout_seconds`/`kpl_device_id`/`dingtalk_chunk_max_bytes`）
- Produces: `KPLClient(config: Config)`，方法 `_post(host: str, params: dict) -> Optional[dict]`；`available` 属性（KPL_ENABLED）

- [ ] **Step 1: 写失败测试 —— KPLClient 基础请求与熔断**

```python
# tests/test_kpl_client.py
# -*- coding: utf-8 -*-
import sys, os, unittest
from unittest import mock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from data_provider.kpl_client import KPLClient


def _config(**kw):
    base = dict(
        stock_list=[],
        kpl_enabled=True, kpl_timeout_seconds=10,
        kpl_device_id="d66474b3-fd78-3a95-a56d-76e29e765ea3",
        dingtalk_chunk_max_bytes=2000,
    )
    base.update(kw)
    return Config(**base)


class KPLClientBaseTest(unittest.TestCase):
    def test_post_builds_url_and_returns_json(self):
        client = KPLClient(_config())
        with mock.patch.object(client, "_session") as session:
            session.get.return_value.status_code = 200
            session.get.return_value.json.return_value = {"errcode": "0", "x": 1}
            result = client._post("apphwshhq.longhuvip.com", {"a": "MoodNumCount"})
        self.assertEqual(result, {"errcode": "0", "x": 1})
        url = session.get.call_args[0][0]
        self.assertIn("longhuvip.com", url)

    def test_post_returns_none_on_http_error(self):
        client = KPLClient(_config())
        with mock.patch.object(client, "_session") as session:
            session.get.return_value.status_code = 500
            self.assertIsNone(client._post("apphwshhq.longhuvip.com", {"a": "x"}))

    def test_disabled_by_config(self):
        client = KPLClient(_config(kpl_enabled=False))
        with mock.patch.object(client, "_session") as session:
            self.assertIsNone(client._post("apphwshhq.longhuvip.com", {"a": "x"}))
        session.get.assert_not_called()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_kpl_client.py -v`
Expected: FAIL（`ImportError: cannot import name 'KPLClient'`）

- [ ] **Step 3: 写 config.py 新字段与加载**

在 `src/config.py` 的 dataclass `Config` 中，「消息长度限制」节后追加：

```python
    # === KPL(开盘啦)数据源配置 ===
    kpl_enabled: bool = True            # KPL 总开关，false 全走 akshare
    kpl_timeout_seconds: int = 10       # 单接口超时
    kpl_device_id: str = "d66474b3-fd78-3a95-a56d-76e29e765ea3"  # 文档公开演示 ID，可覆盖

    # 钉钉单条消息字节上限（防客户端折叠，默认 2000B）
    dingtalk_chunk_max_bytes: int = 2000
```

在 `_load_from_env` 的 `domestic_domains` 列表（约 1010 行）追加 `'longhuvip.com',`。

在 `_load_from_env` 的 `return cls(...)` 中追加（放在 `dingtalk_webhook_secret=...` 之后）：

```python
            kpl_enabled=os.getenv('KPL_ENABLED', 'true').lower() != 'false',
            kpl_timeout_seconds=parse_env_int(
                os.getenv('KPL_TIMEOUT_SECONDS'), 10,
                field_name='KPL_TIMEOUT_SECONDS', minimum=1,
            ),
            kpl_device_id=(
                os.getenv('KPL_DEVICE_ID')
                or 'd66474b3-fd78-3a95-a56d-76e29e765ea3'
            ).strip(),
            dingtalk_chunk_max_bytes=parse_env_int(
                os.getenv('DINGTALK_CHUNK_MAX_BYTES'), 2000,
                field_name='DINGTALK_CHUNK_MAX_BYTES', minimum=100,
            ),
```

- [ ] **Step 4: 写 KPLClient 基础模块**

```python
# data_provider/kpl_client.py
# -*- coding: utf-8 -*-
"""KPL(开盘啦 longhuvip) 私有接口客户端。

无需鉴权，使用文档公开 DeviceID。所有接口大陆直连（NO_PROXY 已含 longhuvip.com）。
单一 HTTP 入口 `_post`，各业务方法负责解析；连续失败 3 次对该动作熔断。
"""
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# 三个域名族：实时/历史/实时辅助
KPL_HOSTS = {
    "realtime": "apphwshhq.longhuvip.com",
    "realtime2": "apphq.longhuvip.com",
    "history": "apphis.longhuvip.com",
}
_KPL_USER_AGENT = "Dalvik/2.1.0 (Linux; U; Android 14; V2178A Build/UP1A.231005.007)"
_CIRCUIT_BREAKER_LIMIT = 3


class KPLClient:
    """KPL 接口客户端（KPL 主数据源，失败由调用方回退 akshare）。"""

    def __init__(self, config) -> None:
        self._enabled = bool(getattr(config, "kpl_enabled", True))
        self._timeout = getattr(config, "kpl_timeout_seconds", 10)
        self._device_id = getattr(config, "kpl_device_id", "") or ""
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": _KPL_USER_AGENT,
        })
        self._fail_counts: Dict[str, int] = {}

    @property
    def available(self) -> bool:
        return self._enabled

    def _post(self, host: str, params: dict) -> Optional[dict]:
        """向 host 发起 GET，返回 JSON dict；失败/熔断返回 None。"""
        if not self._enabled:
            return None
        action = params.get("a", "unknown")
        if self._fail_counts.get(action, 0) >= _CIRCUIT_BREAKER_LIMIT:
            logger.warning("[KPL] %s 连续失败，熔断跳过", action)
            return None
        url = f"https://{host}/w1/api/index.php"
        params = dict(params)
        params.setdefault("PhoneOSNew", 1)
        params.setdefault("DeviceID", self._device_id)
        try:
            resp = self._session.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("errcode") not in (None, "0", 0):
                raise ValueError(f"errcode={data.get('errcode')}")
            self._fail_counts.pop(action, None)
            return data
        except Exception as e:  # noqa: BLE001 — 私有接口任何失败都回退
            self._fail_counts[action] = self._fail_counts.get(action, 0) + 1
            logger.warning("[KPL] %s 请求失败: %s", action, e)
            return None
```

- [ ] **Step 5: 更新 .env.example 与 CHANGELOG**

`.env.example` 追加：

```
# KPL(开盘啦) 数据源
KPL_ENABLED=true
KPL_TIMEOUT_SECONDS=10
# KPL_DEVICE_ID=d66474b3-fd78-3a95-a56d-76e29e765ea3
# 钉钉单条消息字节上限（防折叠）
DINGTALK_CHUNK_MAX_BYTES=2000
```

`docs/CHANGELOG.md` `[Unreleased]` 追加一行：`- [新功能] 接入 KPL(开盘啦) 数据源，替代部分 akshare 大盘数据`

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_kpl_client.py -v`
Expected: PASS（3 tests）

- [ ] **Step 7: Commit**

```bash
git add src/config.py data_provider/kpl_client.py tests/test_kpl_client.py .env.example docs/CHANGELOG.md
git commit -m "feat: add KPL client base module and config"
```

---

### Task 2: 大盘统计替换（MoodNumCount + RiseFallAnalysis）

**Files:**
- Modify: `data_provider/kpl_client.py`（新增 `get_market_stats`）
- Modify: `src/market_analyzer.py`（`_get_market_statistics` 约 406-426 行）
- Test: `tests/test_kpl_client.py`

**Interfaces:**
- Consumes: `KPLClient._post`
- Produces: `KPLClient.get_market_stats() -> Optional[Dict[str, Any]]`，keys：`up_count/down_count/limit_up_count/limit_down_count/natural_zt/po_ban_rate/zha_ban/total_amount`；`MarketAnalyzer._get_market_statistics` 内部先 KPL 后 `self.data_manager.get_market_stats()`

- [ ] **Step 1: 写失败测试**

```python
class KPLMarketStatsTest(unittest.TestCase):
    def _client(self, payloads):
        client = KPLClient(_config())
        client._post = mock.Mock(side_effect=payloads)
        return client

    def test_parses_mood_and_risefall(self):
        client = self._client([
            {"errcode": "0", "list": {"SZJS": 4279, "XDJS": 836, "ZTJS": 54,
                                       "DTJS": 2, "qscln": 137018971, "bl": 7.65}},
            {"errcode": "0", "info": [[47, 3, 43, 1, 18.9655, 11, "2026-04-10"]]},
        ])
        stats = client.get_market_stats()
        self.assertEqual(stats["up_count"], 4279)
        self.assertEqual(stats["down_count"], 836)
        self.assertEqual(stats["limit_up_count"], 54)
        self.assertEqual(stats["limit_down_count"], 2)
        self.assertAlmostEqual(stats["total_amount"], 13701.8971, places=2)  # qscln 万元→亿
        self.assertEqual(stats["natural_zt"], 43)
        self.assertAlmostEqual(stats["po_ban_rate"], 18.9655, places=4)
        self.assertEqual(stats["zha_ban"], 11)

    def test_returns_none_on_first_failure(self):
        client = self._client([None])
        self.assertIsNone(client.get_market_stats())
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_kpl_client.py::KPLMarketStatsTest -v`
Expected: FAIL（`AttributeError: 'KPLClient' object has no attribute 'get_market_stats'`）

- [ ] **Step 3: 实现 get_market_stats**

在 `data_provider/kpl_client.py` 追加：

```python
    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """市场涨跌统计：MoodNumCount(涨跌/涨停跌停家数+量能) + RiseFallAnalysis(自然涨停/破板率/炸板)。"""
        mood = self._post(KPL_HOSTS["realtime"], {
            "a": "MoodNumCount", "apiv": "w43", "c": "MarketMood", "VerSion": "5.22.0.2",
        })
        if mood is None:
            return None
        lst = mood.get("list") or {}
        up = _to_int(lst.get("SZJS")); down = _to_int(lst.get("XDJS"))
        zt = _to_int(lst.get("ZTJS")); dt = _to_int(lst.get("DTJS"))
        qscln = _to_float(lst.get("qscln")) or 0.0
        # qscln 为全市场量能(万元)，换算为亿元
        total_amount = qscln / 1e4
        rf = self._post(KPL_HOSTS["realtime"], {
            "a": "RiseFallAnalysis", "apiv": "w43", "c": "HomeDingPan", "VerSion": "5.22.0.2",
        })
        natural_zt = po_ban_rate = zha_ban = None
        if rf and rf.get("info"):
            row = rf["info"][0]
            natural_zt = _to_int(row[2]); po_ban_rate = _to_float(row[4]); zha_ban = _to_int(row[5])
        return {
            "up_count": up, "down_count": down, "limit_up_count": zt, "limit_down_count": dt,
            "total_amount": round(total_amount, 2),
            "natural_zt": natural_zt, "po_ban_rate": po_ban_rate, "zha_ban": zha_ban,
        }
```

在模块级追加辅助函数：

```python
def _to_int(v: Any) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0

def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
```

- [ ] **Step 4: 接入 _get_market_statistics**

修改 `src/market_analyzer.py::_get_market_statistics`，在 `self.data_manager.get_market_stats()` **之前**插入 KPL 优先：

```python
    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计（KPL 优先，失败回退 data_manager/akshare）。"""
        try:
            logger.info("[大盘] 获取市场涨跌统计...")
            stats = self._get_kpl_market_stats()
            if not stats:
                stats = self.data_manager.get_market_stats()
            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)
                # KPL 新增字段透传到 overview 供新段落使用
                overview.natural_zt = stats.get('natural_zt')
                overview.po_ban_rate = stats.get('po_ban_rate')
                overview.zha_ban = stats.get('zha_ban')
                logger.info(f"[大盘] 涨:{overview.up_count} 跌:{overview.down_count} "
                            f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                            f"成交额:{overview.total_amount:.0f}亿")
        except Exception as e:
            logger.error(f"[大盘] 获取涨跌统计失败: {e}")

    def _get_kpl_market_stats(self) -> Optional[Dict[str, Any]]:
        """KPL 大盘统计；KPL 不可用/失败返回 None。"""
        kpl = self._get_kpl_client()
        if kpl is None:
            return None
        try:
            return kpl.get_market_stats()
        except Exception as e:
            logger.warning(f"[KPL] 大盘统计失败: {e}")
            return None
```

在 `MarketOverview` dataclass（`src/market_analyzer.py` 约 83-89 行）追加三个字段，并在类中新增 `_get_kpl_client` 单例辅助：

```python
    # KPL 新增（无 akshare 等价物，失败为 None）
    natural_zt: Optional[int] = None
    po_ban_rate: Optional[float] = None
    zha_ban: Optional[int] = None
```

```python
    def _get_kpl_client(self):
        """懒加载 KPLClient（单实例）。"""
        if not getattr(self, "_kpl_client", None):
            from data_provider.kpl_client import KPLClient
            self._kpl_client = KPLClient(self.config)
        if not self._kpl_client.available:
            return None
        return self._kpl_client
```

> 已确认：`MarketAnalyzer.__init__`（[src/market_analyzer.py:128](src/market_analyzer.py#L128)）内部已有 `self.config = get_config()`，`_get_kpl_client` 直接使用即可，无需改构造签名。
> 已确认：`_get_market_statistics`/`_get_sector_rankings` 已由 `self.profile.has_market_stats`/`has_sector_rankings` 门控（仅 A 股生效，[src/market_analyzer.py:314-319](src/market_analyzer.py#L314-L319)），KPL 数据无需额外 region 判断。

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_kpl_client.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add data_provider/kpl_client.py src/market_analyzer.py tests/test_kpl_client.py
git commit -m "feat: replace market stats with KPL MoodNumCount/RiseFallAnalysis"
```

---

### Task 3: 概念涨跌榜替换（RealRankingInfo）

**Files:**
- Modify: `data_provider/kpl_client.py`（新增 `get_sector_rankings`）
- Modify: `src/market_analyzer.py`（`_get_sector_rankings` 约 460-481 行）
- Test: `tests/test_kpl_client.py`

**Interfaces:**
- Consumes: `KPLClient._post`
- Produces: `KPLClient.get_sector_rankings() -> Optional[List[Dict[str, Any]]]`，每项 keys：`code/name/change_pct/strength/turnover_amount/main_net/volume_ratio`，按 `strength` 降序

- [ ] **Step 1: 写失败测试**

```python
class KPLSectorTest(unittest.TestCase):
    def test_parses_ranking_info(self):
        client = KPLClient(_config())
        client._post = mock.Mock(return_value={"errcode": "0", "list": [
            ["801807", "算力", 12505, 2.503, 0.753, 524251307081, 350201029,
             27494806325, -27144605296, 1.114, 12058520900160, 2.66],
            ["801660", "通信", 7602, 0.126, 0, 1405620402, 89906965,
             293509488, -203602523, 2.42, 5253626856256, 0],
        ]})
        sectors = client.get_sector_rankings()
        self.assertEqual(len(sectors), 2)
        self.assertEqual(sectors[0]["name"], "算力")
        self.assertEqual(sectors[0]["strength"], 12505)
        self.assertAlmostEqual(sectors[0]["change_pct"], 2.503, places=3)
        self.assertEqual(sectors[1]["name"], "通信")

    def test_returns_none_on_failure(self):
        client = KPLClient(_config())
        client._post = mock.Mock(return_value=None)
        self.assertIsNone(client.get_sector_rankings())
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_kpl_client.py::KPLSectorTest -v`
Expected: FAIL

- [ ] **Step 3: 实现 get_sector_rankings**

```python
    def get_sector_rankings(self) -> Optional[List[Dict[str, Any]]]:
        """板块强度榜：RealRankingInfo(ZSType=7)，按强度降序返回全量板块。"""
        data = self._post(KPL_HOSTS["realtime2"], {
            "Order": 1, "a": "RealRankingInfo", "st": 60, "apiv": "w26",
            "Type": 1, "c": "ZhiShuRanking", "Index": 0, "ZSType": 7, "VerSion": "5.21.0.2",
        })
        if data is None:
            return None
        rows = data.get("list") or []
        result = []
        for r in rows:
            if not isinstance(r, list) or len(r) < 10:
                continue
            result.append({
                "code": str(r[0] or ""),
                "name": str(r[1] or ""),
                "strength": _to_float(r[2]) or 0.0,
                "change_pct": _to_float(r[3]),
                "turnover_amount": _to_float(r[5]),
                "main_net": _to_float(r[6]),
                "volume_ratio": _to_float(r[9]),
            })
        result.sort(key=lambda x: x["strength"], reverse=True)
        return result
```

- [ ] **Step 4: 接入 _get_sector_rankings**

修改 `src/market_analyzer.py::_get_sector_rankings`，在 `_fetch_concept_rankings()` 之前插入 KPL 优先（keep 现有 try/except 结构）：

```python
    def _get_sector_rankings(self, overview: MarketOverview):
        """获取概念涨跌榜（KPL RealRankingInfo 优先，失败回退同花顺）。"""
        try:
            kpl_sectors = self._get_kpl_sector_rankings()
            if kpl_sectors:
                overview.top_sectors = kpl_sectors[:5]
                overview.bottom_sectors = kpl_sectors[-5:][::-1]
                logger.info(f"[大盘] 概念涨跌榜(KPL): "
                            f"领涨 {[s['name'] for s in overview.top_sectors]}")
                return
        except Exception as e:
            logger.warning(f"[大盘] KPL 概念涨跌榜失败: {e}")

        try:
            logger.info("[大盘] 获取概念涨跌榜(同花顺)...")
            results = self._fetch_concept_rankings()
            if results:
                overview.top_sectors = [{'name': n, 'change_pct': c} for n, c in results[:5]]
                overview.bottom_sectors = [{'name': n, 'change_pct': c} for n, c in results[-5:][::-1]]
                logger.info(f"[大盘] 领涨概念: {[s['name'] for s in overview.top_sectors]}")
                return
        except Exception as e:
            logger.warning(f"[大盘] 概念涨跌榜失败: {e}，降级行业排行")

        try:
            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)
            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors
        except Exception:
            pass

    def _get_kpl_sector_rankings(self) -> Optional[List[Dict[str, Any]]]:
        kpl = self._get_kpl_client()
        if kpl is None:
            return None
        try:
            return kpl.get_sector_rankings()
        except Exception as e:
            logger.warning(f"[KPL] 板块强度榜失败: {e}")
            return None
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_kpl_client.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add data_provider/kpl_client.py src/market_analyzer.py tests/test_kpl_client.py
git commit -m "feat: replace concept rankings with KPL RealRankingInfo"
```

---

### Task 4: 概念天梯 + 涨停复盘（GetPlateInfo_w38）

**Files:**
- Modify: `data_provider/kpl_client.py`（新增 `get_limit_up_review`）
- Modify: `src/market_analyzer.py`（`LimitUpStock` 加字段、`_build_limit_up_ladder`、`_format_limit_up_section`）
- Test: `tests/test_kpl_client.py`、`tests/test_market_review.py`

**Interfaces:**
- Consumes: `KPLClient._post`；`LimitUpLadder` dataclass
- Produces: `KPLClient.get_limit_up_review() -> Optional[Dict[str, Any]]`，keys：`nums{zt,dt,zbl}`、`boards: List[{sector, total, stocks_by_board: {int: List[dict]}}]`，股票 dict：`code/name/consecutive/first_time/limit_reason/board/seal_amount`

> **设计细化**：KPL `GetPlateInfo_w38` 是复盘接口（18:00 定时运行时正是收盘数据），单请求同时给出涨停池（含连板数、板块、原因、封单）与板块分组，优于文档中的 `DailyLimitPerformance`（需 5 次 PidType 请求且无板块名）。本任务以 `GetPlateInfo_w38` 作为 KPL 涨停数据唯一来源，`concept_ladders`（题材天梯）由此填充；`industry_ladders`（东财行业天梯）保留 akshare 涨停池（KPL 无东财行业分类，此为设计第 4.1 节"两者并存"的落实）。

- [ ] **Step 1: 写失败测试**

```python
class KPLLimitUpReviewTest(unittest.TestCase):
    def _payload(self):
        return {"errcode": "0", "nums": {"ZT": 2, "DT": 1, "ZBL": 0.0},
                "list": [{"ZSCode": "801074", "ZSName": "核电", "StockList": [
                    ["000777", "中核科技", 0, "", 0, 0, 1759973841, 0, 70447232,
                     "首板", 1, "可控核聚变、核电", 239159833, 682199256, 11.73,
                     5877917502, "可控核聚变", "核电(可控核聚变)；根据互动易：...", 1],
                ]}]}

    def test_parses_boards(self):
        client = KPLClient(_config())
        client._post = mock.Mock(return_value=self._payload())
        review = client.get_limit_up_review()
        self.assertIsNotNone(review)
        self.assertEqual(review["nums"]["zt"], 2)
        board = review["boards"][0]
        self.assertEqual(board["sector"], "核电")
        self.assertEqual(board["total"], 1)
        stock = board["stocks_by_board"][1][0]
        self.assertEqual(stock["name"], "中核科技")
        self.assertEqual(stock["consecutive"], 1)
        self.assertIn("互动易", stock["limit_reason"])

    def test_returns_none_on_failure(self):
        client = KPLClient(_config())
        client._post = mock.Mock(return_value=None)
        self.assertIsNone(client.get_limit_up_review())
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_kpl_client.py::KPLLimitUpReviewTest -v`
Expected: FAIL

- [ ] **Step 3: 实现 get_limit_up_review**

```python
    def get_limit_up_review(self) -> Optional[Dict[str, Any]]:
        """涨停复盘 GetPlateInfo_w38：按板块分组的涨停池 + 涨停/跌停/炸板率概览。"""
        data = self._post(KPL_HOSTS["realtime"], {
            "a": "GetPlateInfo_w38", "st": 1000, "c": "DailyLimitResumption",
            "VerSion": "5.21.0.0", "Index": 0, "apiv": "w42",
        })
        if data is None:
            return None
        nums = data.get("nums") or {}
        boards = []
        for item in data.get("list") or []:
            sector = str(item.get("ZSName") or "").strip()
            stock_list = item.get("StockList") or []
            if not sector or not stock_list:
                continue
            by_board: Dict[int, list] = {}
            for s in stock_list:
                if not isinstance(s, list) or len(s) < 18:
                    continue
                consecutive = _to_int(s[10]) or 1
                by_board.setdefault(consecutive, []).append({
                    "code": str(s[0] or ""),
                    "name": str(s[1] or ""),
                    "consecutive": consecutive,
                    "first_time": _format_epoch(s[6]),
                    "limit_reason": str(s[17] or s[16] or "").replace("\r\n", " ").strip(),
                    "board": sector,
                    "seal_amount": _to_float(s[8]),
                })
            boards.append({
                "sector": sector,
                "total": len(stock_list),
                "stocks_by_board": by_board,
            })
        boards.sort(key=lambda b: b["total"], reverse=True)
        return {
            "nums": {"zt": _to_int(nums.get("ZT")), "dt": _to_int(nums.get("DT")),
                     "zbl": _to_float(nums.get("ZBL"))},
            "boards": boards,
        }
```

模块级追加辅助：

```python
def _format_epoch(v: Any) -> str:
    """epoch 秒 → 'HH:MM'（涨停时间展示用）；非法返回空串。"""
    try:
        ts = int(float(v))
    except (TypeError, ValueError):
        return ""
    if ts <= 0:
        return ""
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ts).strftime("%H:%M")
```

- [ ] **Step 4: LimitUpStock 加字段**

`src/market_analyzer.py` 的 `LimitUpStock` dataclass 追加（默认值向后兼容）：

```python
    limit_reason: str = ""           # 涨停原因（KPL 开盘啦原因）
```

- [ ] **Step 5: 改造 _build_limit_up_ladder**

重写 `_build_limit_up_ladder`，KPL 复盘优先（基础统计 + 概念天梯），akshare 涨停池补充行业天梯：

```python
    def _build_limit_up_ladder(self) -> Optional[LimitUpLadder]:
        """获取涨停池并构建连板天梯：KPL 复盘优先，akshare 兜底。"""
        from collections import Counter

        try:
            # ---- KPL 涨停复盘（基础统计 + 概念天梯）----
            kpl_pool = None
            kpl_concept_ladders = []
            kpl_client = self._get_kpl_client()
            if kpl_client is not None:
                try:
                    kpl_review = kpl_client.get_limit_up_review()
                    if kpl_review and kpl_review["boards"]:
                        kpl_pool = []
                        for board in kpl_review["boards"]:
                            for byb in board["stocks_by_board"].values():
                                for st in byb:
                                    kpl_pool.append(LimitUpStock(
                                        code=st["code"], name=st["name"],
                                        consecutive=st["consecutive"],
                                        industry=st["board"],
                                        first_time=st["first_time"],
                                        limit_reason=st["limit_reason"],
                                    ))
                        kpl_concept_ladders = [
                            {
                                'sector': b['sector'],
                                'total': b['total'],
                                'stocks_by_board': b['stocks_by_board'],
                            } for b in kpl_review["boards"][:5]
                        ]
                        logger.info(f"[连板] KPL 复盘成功: {len(kpl_pool)}只, "
                                    f"{len(kpl_concept_ladders)}个概念板块")
                except Exception as e:
                    logger.warning(f"[连板] KPL 复盘失败: {e}")

            # ---- akshare 涨停池（基础统计回退 + 行业天梯）----
            pool = kpl_pool
            industry_ladders = []
            try:
                from data_provider.akshare_fetcher import AkshareFetcher
                akshare = AkshareFetcher()
                ak_pool = akshare.get_limit_up_pool()
                if ak_pool:
                    ak_stocks = [
                        LimitUpStock(
                            code=str(item.get('code', '')),
                            name=str(item.get('name', '')),
                            consecutive=int(item.get('consecutive', 1)),
                            industry=str(item.get('industry', '')),
                            first_time=str(item.get('first_time', '')),
                            limit_reason=str(item.get('limit_reason', '')),
                        )
                        for item in ak_pool
                    ]
                    # 行业天梯始终以 akshare 东财行业为准（KPL 无行业分类）
                    industry_map = {}
                    for s in ak_stocks:
                        ind = s.industry or "其他"
                        industry_map.setdefault(ind, {}).setdefault(s.consecutive, []).append(s)
                    for sector, by_board in industry_map.items():
                        industry_ladders.append({
                            'sector': sector,
                            'total': sum(len(v) for v in by_board.values()),
                            'stocks_by_board': dict(sorted(by_board.items(), key=lambda x: x[0], reverse=True)),
                        })
                    industry_ladders.sort(key=lambda x: x['total'], reverse=True)
                    industry_ladders = industry_ladders[:5]
                    if pool is None:
                        pool = ak_stocks
                        logger.info("[连板] 使用 akshare 涨停池（KPL 不可用）")
            except Exception as e:
                logger.warning(f"[连板] akshare 涨停池失败: {e}")

            if pool is None:
                logger.warning("[连板] 涨停池接口全部失败")
                return None
            if not pool:
                logger.info("[连板] 今日无涨停数据")
                return LimitUpLadder(total=0)

            consec_dist = Counter(s.consecutive for s in pool)
            consecutive_stats = {k: consec_dist[k] for k in sorted(consec_dist)}
            height_leaders = sorted(
                [s for s in pool if s.consecutive >= 3],
                key=lambda s: s.consecutive, reverse=True
            )[:10]

            return LimitUpLadder(
                total=len(pool),
                consecutive_stats=consecutive_stats,
                height_leaders=height_leaders,
                industry_ladders=industry_ladders,
                concept_ladders=kpl_concept_ladders,
            )
        except Exception as e:
            logger.warning(f"[连板] 构建天梯失败: {e}")
            return None
```

- [ ] **Step 6: 渲染概念天梯**

修改 `_format_limit_up_section`，在行业天梯块之后追加概念天梯渲染（复用同一格式）：

```python
        # ---- 概念天梯（KPL 题材）----
        if ladeder.concept_ladders:
            lines.append("\n### 连板天梯 · 概念板块\n")
            for ladder in ladeder.concept_ladders:
                sector = ladder['sector']
                total = ladder['total']
                lines.append(f"**{sector}**（涨停{total}只）\n")
                by_board = ladder['stocks_by_board']
                for board, stock_list in sorted(by_board.items(), key=lambda x: x[0], reverse=True):
                    label = "首板" if board == 1 else f"{board}板"
                    names = "、".join(s['name'] if isinstance(s, dict) else getattr(s, 'name', '')
                                      for s in stock_list[:5])
                    if len(stock_list) > 5:
                        names += f"等{len(stock_list)}只"
                    lines.append(f"  - {label}: {names}\n")
```

> 注意：`kpl_concept_ladders` 中 `stocks_by_board` 的值为 dict（来自 KPLClient），上方渲染兼容 dict 与 `LimitUpStock` 两种形态。

- [ ] **Step 7: 运行测试确认通过**

Run: `python -m pytest tests/test_kpl_client.py tests/test_market_review.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add data_provider/kpl_client.py src/market_analyzer.py tests/test_kpl_client.py
git commit -m "feat: fill concept ladder from KPL limit-up review"
```

---

### Task 5: 情绪定量指标 + 涨停原因 + 百日新高/大面股

**Files:**
- Modify: `data_provider/kpl_client.py`（新增 `get_market_emotion`/`get_limit_up_reasons`/`get_new_high_sectors`/`get_big_loss_stocks`）
- Modify: `src/market_analyzer.py`（新增 `_build_emotion_section`/`_build_risk_section`/`_inject_risk_section`；`_build_limit_up_prompt_block` 扩展；`generate_market_review` 接入）
- Test: `tests/test_kpl_client.py`、`tests/test_market_review.py`

**Interfaces:**
- Consumes: `KPLClient._post`；`_get_kpl_client`
- Produces: `KPLClient.get_market_emotion() -> Optional[Dict[str, Any]]`（`strong/limit_height/big_loss_count/yesterday_zt_pcp/yesterday_lb_pcp/yesterday_pb_pcp/day`）；`get_limit_up_reasons(stock_ids: List[str]) -> Dict[str, str]`；`get_new_high_sectors() -> Optional[List[Dict]]`（`name/count/ratio/code`）；`get_big_loss_stocks(date: str) -> Optional[List[Dict]]`（`code/name/change_pct/concept`）

- [ ] **Step 1: 写失败测试**

```python
class KPLEmotionTest(unittest.TestCase):
    def _client(self, payloads):
        client = KPLClient(_config())
        client._post = mock.Mock(side_effect=payloads)
        return client

    def test_parses_emotion(self):
        client = self._client([
            {"errcode": "0", "info": [{"ztjs": "9", "Day": "2026-04-10",
                                        "df_num": "0", "strong": "48", "lbgd": "3"}]},
            {"errcode": "0", "List": ["--", -55, 1, 0, 1.716, 0, 0, 0]},   # 昨涨停
            {"errcode": "0", "List": ["--", 175, 1, 0, 4.508, 0, 0, 0]},   # 昨连板
            {"errcode": "0", "List": ["--", -108, 1, 0, -0.803, 0, 0, 0]}, # 昨破板
        ])
        emo = client.get_market_emotion()
        self.assertEqual(emo["strong"], 48)
        self.assertEqual(emo["limit_height"], 3)
        self.assertEqual(emo["big_loss_count"], 0)
        self.assertAlmostEqual(emo["yesterday_zt_pcp"], 1.716, places=3)
        self.assertAlmostEqual(emo["yesterday_lb_pcp"], 4.508, places=3)
        self.assertAlmostEqual(emo["yesterday_pb_pcp"], -0.803, places=3)

    def test_emotion_fails_returns_none(self):
        client = self._client([None])
        self.assertIsNone(client.get_market_emotion())


class KPLRiskTest(unittest.TestCase):
    def test_parses_new_high_and_big_loss(self):
        client = KPLClient(_config())
        client._post = mock.Mock(side_effect=[
            {"errcode": "0", "List": [["芯片", "76,18", 801001], ["算力", "44,10", 801807]]},
            {"errcode": "0", "List": [["002676", "顺威股份", "-1.18%", -10.2, "", 0, "聚丙烯、壳资源"]]},
        ])
        highs = client.get_new_high_sectors()
        self.assertEqual(highs[0]["name"], "芯片")
        self.assertEqual(highs[0]["count"], 76)
        losses = client.get_big_loss_stocks("2026-04-10")
        self.assertEqual(losses[0]["name"], "顺威股份")

    def test_reasons(self):
        client = KPLClient(_config())
        client._post = mock.Mock(return_value={
            "errcode": "0", "StockID": "001337",
            "List": [{"Reason": "黄金；现货黄金创新高。\r\n黄金：公司金矿资源量...", "SCLT": "日内龙一"}]})
        reasons = client.get_limit_up_reasons(["001337"])
        self.assertEqual(reasons["001337"], "黄金；现货黄金创新高。 黄金：公司金矿资源量...")
```

追加到 `tests/test_market_review.py`（该文件顶部已有 `ensure_litellm_stub()`、`MagicMock`、`patch` 导入）：

```python
class EmotionRiskSectionTest(unittest.TestCase):
    def _make_analyzer(self):
        from src.market_analyzer import MarketAnalyzer
        return MarketAnalyzer(analyzer=MagicMock())

    def test_emotion_section_builds(self):
        from src.market_analyzer import MarketOverview
        analyzer = self._make_analyzer()
        overview = MarketOverview(
            date="2026-08-13", limit_up_count=54, limit_down_count=2,
            natural_zt=43, po_ban_rate=18.9655, zha_ban=11,
        )
        kpl = MagicMock()
        kpl.get_market_emotion.return_value = {
            "strong": 48, "limit_height": 3, "big_loss_count": 0, "day": "2026-08-13",
            "yesterday_zt_pcp": 1.716, "yesterday_lb_pcp": 4.508, "yesterday_pb_pcp": -0.803,
        }
        with patch.object(analyzer, "_get_kpl_client", return_value=kpl):
            section = analyzer._build_emotion_section(overview)
        self.assertIn("情绪值 **48**/100", section)
        self.assertIn("自然涨停43只", section)
        self.assertIn("昨涨停今表现: +1.72%", section)

    def test_emotion_section_empty_when_no_kpl(self):
        from src.market_analyzer import MarketOverview
        analyzer = self._make_analyzer()
        with patch.object(analyzer, "_get_kpl_client", return_value=None):
            section = analyzer._build_emotion_section(MarketOverview(date="2026-08-13"))
        self.assertEqual(section, "")

    def test_risk_section_and_injection(self):
        analyzer = self._make_analyzer()
        kpl = MagicMock()
        kpl.get_new_high_sectors.return_value = [{"name": "芯片", "count": 76, "ratio": 18}]
        kpl.get_big_loss_stocks.return_value = [
            {"name": "顺威股份", "code": "002676", "change_pct": "-10.2%", "concept": "壳资源"}]
        with patch.object(analyzer, "_get_kpl_client", return_value=kpl):
            section = analyzer._build_risk_section()
        self.assertIn("百日新高板块", section)
        self.assertIn("顺威股份", section)
        review = "### 五、消息催化\n\n1. 半导体走强\n\n### 六、风险提示\n\n内容"
        injected = analyzer._inject_risk_section(review, section)
        self.assertLess(injected.index("百日新高板块"), injected.index("### 六、风险提示"))
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_kpl_client.py::KPLEmotionTest tests/test_kpl_client.py::KPLRiskTest tests/test_market_review.py::EmotionRiskSectionTest -v`
Expected: FAIL（`AttributeError: 'KPLClient' object has no attribute 'get_market_emotion'` / `'MarketAnalyzer' object has no attribute '_build_emotion_section'`）

- [ ] **Step 3: 实现四个 KPL 方法**

```python
    def get_market_emotion(self) -> Optional[Dict[str, Any]]:
        """市场情绪：ChangeStatistics(情绪值/连板高度/大幅回撤) + 昨涨停/连板/破板今表现。"""
        cs = self._post(KPL_HOSTS["realtime2"], {
            "a": "ChangeStatistics", "st": 10, "c": "HomeDingPan", "VerSion": "5.21.0.2",
        })
        if cs is None or not cs.get("info"):
            return None
        row = cs["info"][0]
        def _plate_pcp(plate_id: int) -> Optional[float]:
            data = self._post(KPL_HOSTS["realtime"], {
                "a": "GetPlate_Info_QJ", "apiv": "w42", "c": "ZhiShuRanking",
                "VerSion": "5.21.0.2", "PlateID": plate_id, "Date": "",
            })
            if data and isinstance(data.get("List"), list) and len(data["List"]) > 4:
                return _to_float(data["List"][4])
            return None
        return {
            "strong": _to_int(row.get("strong")),
            "limit_height": _to_int(row.get("lbgd")),
            "big_loss_count": _to_int(row.get("df_num")),
            "day": str(row.get("Day") or ""),
            "yesterday_zt_pcp": _plate_pcp(801900),
            "yesterday_lb_pcp": _plate_pcp(801902),
            "yesterday_pb_pcp": _plate_pcp(801903),
        }

    def get_limit_up_reasons(self, stock_ids: List[str]) -> Dict[str, str]:
        """批量取涨停原因（GetKLineZhangTing），只保留首个原因，清洗换行。"""
        result: Dict[str, str] = {}
        for sid in stock_ids[:5]:
            data = self._post(KPL_HOSTS["realtime2"], {
                "a": "GetKLineZhangTing", "apiv": "w24", "c": "StockLineData",
                "StockID": sid, "VerSion": "5.21.0.2",
            })
            if data and data.get("List"):
                reason = str(data["List"][0].get("Reason") or "").strip()
                if reason:
                    result[sid] = " ".join(reason.split())  # 压平 \r\n
        return result

    def get_new_high_sectors(self) -> Optional[List[Dict[str, Any]]]:
        """百日新高板块：GroupCount_w28，name/count/ratio/code。"""
        data = self._post(KPL_HOSTS["realtime"], {
            "a": "GroupCount_w28", "c": "StockNewHigh", "VerSion": "5.20.0.8",
            "apiv": "w41", "Type": "0_0_0_0_0",
        })
        if data is None:
            return None
        result = []
        for r in data.get("List") or []:
            if not isinstance(r, list) or len(r) < 3:
                continue
            count, ratio = 0, 0.0
            if "," in str(r[1]):
                parts = str(r[1]).split(",")
                count = _to_int(parts[0])
                ratio = _to_float(parts[1]) or 0.0
            result.append({"name": str(r[0] or ""), "count": count, "ratio": ratio, "code": str(r[2])})
        return result[:5]

    def get_big_loss_stocks(self, date: str) -> Optional[List[Dict[str, Any]]]:
        """大面股 GetPMSL_KQXY（Date=YYYY-MM-DD）。"""
        data = self._post(KPL_HOSTS["history"], {
            "Date": date, "Index": 0, "PhoneOSNew": 2, "VerSion": "5.13.0.3",
            "a": "GetPMSL_KQXY", "apiv": "w35", "c": "FuPanLa", "st": 20,
        })
        if data is None:
            return None
        result = []
        for r in data.get("List") or []:
            if not isinstance(r, list) or len(r) < 7:
                continue
            result.append({
                "code": str(r[0] or ""), "name": str(r[1] or ""),
                "change_pct": str(r[2] or ""), "concept": str(r[6] or ""),
            })
        return result[:5]
```

- [ ] **Step 4: 新增渲染方法（情绪段 + 新高/大面股段）**

在 `src/market_analyzer.py` 新增三个方法：

```python
    def _build_emotion_section(self, overview: MarketOverview) -> str:
        """KPL 市场情绪段：复用大盘统计已取的自然涨停/炸板/破板率 + 情绪值/连板高度/昨表现。

        任一部分失败则该部分省略；两部分都不可用时返回空串。
        """
        kpl = self._get_kpl_client()
        if kpl is None:
            return ""
        try:
            lines = ["\n### 市场情绪\n"]
            # 涨停/跌停/自然涨停/炸板/破板率（复用 _get_market_statistics 已取 KPL 数据）
            if overview.natural_zt is not None:
                zbl = f"{overview.po_ban_rate:.2f}%" if overview.po_ban_rate is not None else "--"
                lines.append(f"- 涨停{overview.limit_up_count}只 跌停{overview.limit_down_count}只，"
                             f"自然涨停{overview.natural_zt}只，炸板{overview.zha_ban}只，破板率 {zbl}")
            # 情绪值 + 昨涨停/连板/破板今表现（ChangeStatistics + GetPlate_Info_QJ）
            emo = kpl.get_market_emotion()
            if emo:
                lines.append(f"- 情绪值 **{emo['strong']}**/100，连板高度 **{emo['limit_height']}** 板，"
                             f"大幅回撤 **{emo['big_loss_count']}** 只")
                tip = ("情绪值过高(>75)短期有释放亏钱效应风险；过低(<25)短线有反弹回暖需求。"
                       if emo["strong"] > 75 or emo["strong"] < 25 else "")
                if tip:
                    lines.append(f"- ⚠️ {tip}")
                pcp = [emo.get("yesterday_zt_pcp"), emo.get("yesterday_lb_pcp"), emo.get("yesterday_pb_pcp")]
                labels = ["昨涨停今表现", "昨连板今表现", "昨破板今表现"]
                parts = [f"{l}: {p:+.2f}%" if p is not None else f"{l}: --" for l, p in zip(labels, pcp)]
                lines.append("- " + " ｜ ".join(parts))
            if len(lines) == 1:  # 两部分都无数据
                return ""
            return "\n".join(lines) + "\n"
        except Exception as e:
            logger.warning(f"[大盘] 情绪段失败: {e}")
            return ""

    def _build_risk_section(self) -> str:
        """百日新高板块 + 大面股风险段（失败整段省略）。"""
        kpl = self._get_kpl_client()
        if kpl is None:
            return ""
        try:
            from datetime import datetime as _dt
            today = _dt.now().strftime("%Y-%m-%d")
            lines: List[str] = []
            highs = kpl.get_new_high_sectors()
            if highs:
                lines.append("\n### 百日新高板块\n")
                for h in highs:
                    lines.append(f"- **{h['name']}**：新高 {h['count']} 家（{h['ratio']}%）")
            losses = kpl.get_big_loss_stocks(today)
            if losses:
                lines.append("\n### 大面股风险\n")
                for lo in losses:
                    concept = f"（{lo['concept']}）" if lo["concept"] else ""
                    lines.append(f"- {lo['name']}（{lo['code']}）{lo['change_pct']}{concept}")
            if not lines:
                return ""
            return "\n".join(lines) + "\n"
        except Exception as e:
            logger.warning(f"[大盘] 新高/大面股段失败: {e}")
            return ""
```

- [ ] **Step 5: 新增注入方法与 Prompt 扩展**

新增 `_inject_risk_section`（插入到「### 五、消息催化」段之后）：

```python
    def _inject_risk_section(self, review: str, risk_section: str) -> str:
        """将新高/大面股段注入到五、消息催化之后；空段直接返回原文。"""
        if not risk_section.strip():
            return review
        import re
        pattern = r"(### 五、消息催化.*?)(?=\n### \S|\Z)"
        match = re.search(pattern, review, re.DOTALL)
        if not match:
            return review + risk_section
        idx = match.end()
        return review[:idx] + "\n" + risk_section + review[idx:]
```

扩展 `_build_limit_up_prompt_block`，把概念天梯 + 涨停原因 + 情绪喂给 LLM：

```python
        # 概念天梯（KPL 题材）——注意：本方法参数名是 ladder（_format_limit_up_section 才是 ladeder）
        if ladder.concept_ladders:
            concept_parts = []
            for cl in ladder.concept_ladders[:3]:
                top_board = max(cl['stocks_by_board'].keys())
                concept_parts.append(f"{cl['sector']}({cl['total']}只/最高{top_board}板)")
            lines.append("- 涨停集中概念: " + "；".join(concept_parts))
            # 高度板涨停原因
            leader_ids = [s.code for s in (ladder.height_leaders or [])[:3]]
            if leader_ids:
                kpl = self._get_kpl_client()
                if kpl is not None:
                    try:
                        reasons = kpl.get_limit_up_reasons(leader_ids)
                        for s in ladder.height_leaders[:3]:
                            reason = reasons.get(s.code) or getattr(s, "limit_reason", "")
                            if reason:
                                lines.append(f"- {s.name}({s.consecutive}板)涨停原因: {reason[:80]}")
                    except Exception:
                        pass
```

在 `generate_market_review` 中接入：把情绪段并入 `limit_up_section`（**不依赖天梯成功**——情绪数据独立抓取，天梯整体失败时仍能注入），风险段单独注入：

```python
            limit_up_ladder = self._build_limit_up_ladder()
            limit_up_section = ""
            if limit_up_ladder is not None:
                limit_up_section = self._format_limit_up_section(limit_up_ladder)
            limit_up_section += self._build_emotion_section(overview)
```

在 `review = self._inject_limit_up_section(review, limit_up_section)` 之后追加：

```python
            risk_section = self._build_risk_section()
            if risk_section:
                review = self._inject_risk_section(review, risk_section)
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_kpl_client.py tests/test_market_review.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add data_provider/kpl_client.py src/market_analyzer.py tests/test_kpl_client.py
git commit -m "feat: add KPL emotion/new-high/big-loss sections and limit-up reasons"
```

---

### Task 6: 推送瘦身（近三日催化线索只留标题）

**Files:**
- Modify: `src/market_analyzer.py`（`_build_news_block` 约 1021-1040 行）
- Test: `tests/test_market_review.py`

**Interfaces:**
- Consumes: `news`（SearchResult 对象或 dict，`title`/`snippet`）
- Produces: `_build_news_block(news) -> str`（只含标题）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_market_review.py`（顶部已有 `ensure_litellm_stub()`、`MagicMock` 导入；`MarketAnalyzer.__init__` 无 `config` 参数，`self.config` 由 `get_config()` 内部注入）：

```python
class NewsBlockTrimTest(unittest.TestCase):
    def _make_analyzer(self):
        from unittest.mock import MagicMock
        from src.market_analyzer import MarketAnalyzer
        analyzer = MarketAnalyzer(analyzer=MagicMock())
        analyzer._get_review_language = lambda: "zh"
        return analyzer

    def test_news_block_title_only(self):
        analyzer = self._make_analyzer()
        news = [
            {"title": "半导体板块异动 个股大面积涨停", "snippet": "板块异动 | A股半导体板块早盘大涨 个股大面积涨停 凡本报注明来源...很长的摘要"},
            {"title": "收评:沪指涨1.17%", "snippet": "中国经济网北京5月6日讯 A股三大指数今日集体上涨..."},
        ]
        block = analyzer._build_news_block(news)
        self.assertIn("半导体板块异动 个股大面积涨停", block)
        self.assertNotIn("板块异动 | A股半导体板块早盘大涨", block)  # snippet 不再出现
        self.assertNotIn("中国经济网北京5月6日讯", block)
        self.assertLessEqual(len(block), len("#### 近三日催化线索") + 200)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_market_review.py::NewsBlockTrimTest -v`
Expected: FAIL（snippet 仍在 block 中）

- [ ] **Step 3: 实现只留标题**

修改 `_build_news_block`：

```python
    def _build_news_block(self, news: List) -> str:
        """Build a compact news catalyst list (title-only, no snippet)."""
        if not news:
            return ""
        if self._get_review_language() == "en":
            lines = ["#### News Catalysts"]
        else:
            lines = ["#### 近三日催化线索"]

        for idx, item in enumerate(news[:5], 1):
            if hasattr(item, "title"):
                title = getattr(item, "title", "") or "-"
            else:
                title = item.get("title", "-") or "-"
            title = str(title).strip().replace("\n", " ")[:60] or "-"
            lines.append(f"{idx}. **{title}**")
        return "\n".join(lines)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_market_review.py::NewsBlockTrimTest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/market_analyzer.py tests/test_market_review.py
git commit -m "perf: trim news catalyst block to title-only"
```

---

### Task 7: 钉钉消息分片修复（配置化分片 + 整体编号）

**Files:**
- Modify: `src/notification_sender/custom_webhook_sender.py`（`__init__`、`send_to_custom`、`_send_dingtalk_chunked`）
- Modify: `src/notification.py`（Stream 路径约 389 行）
- Modify: `main.py`（三处 4096 预切分：缠论约 933-936、飞龙约 1044-1047、一指约 1147-1150）
- Test: `tests/test_notification_sender.py`

**Interfaces:**
- Consumes: `Config.dingtalk_chunk_max_bytes`（Task 1 已加）
- Produces: `_send_dingtalk_chunked(url, content, max_bytes=2000)`，budget = `max(1000, max_bytes - 400)`；`CustomWebhookSender` 属性 `_dingtalk_chunk_max_bytes`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_notification_sender.py`（该文件顶部已有 `_config(**overrides)` helper 与 `CustomWebhookSender` 导入，类命名沿用 `Test` 前缀约定）：

```python
class TestDingTalkChunk(unittest.TestCase):
    def _sender(self, chunk_bytes=2000):
        cfg = _config(
            dingtalk_chunk_max_bytes=chunk_bytes,
            custom_webhook_urls=["https://oapi.dingtalk.com/robot/send?access_token=test"],
            custom_webhook_body_template=None,
        )
        return CustomWebhookSender(cfg)

    def test_send_to_custom_chunks_dingtalk_by_config(self):
        # 走 send_to_custom 钉钉路径：无 body 模板时落到 _send_dingtalk_chunked，
        # 分片大小必须取自 config 的 dingtalk_chunk_max_bytes(2000)，而非写死 20000。
        sender = self._sender(chunk_bytes=2000)
        content = "# 大盘复盘\n" + ("超长内容测试" * 400)  # ~2400B
        with mock.patch.object(sender, "_post_custom_webhook", return_value=True) as post:
            sender.send_to_custom(content)
        self.assertGreater(len(post.call_args_list), 1)  # 拆成多条
        for call in post.call_args_list:
            payload = call[0][1]
            text = payload["markdown"]["text"]
            self.assertLessEqual(len(text.encode("utf-8")), 2400)  # 单条 ≤ 2KB + 页标记/keyword 余量

    def test_single_message_when_small(self):
        sender = self._sender(chunk_bytes=2000)
        with mock.patch.object(sender, "_post_custom_webhook", return_value=True) as post:
            sender.send_to_custom("小消息")
        self.assertEqual(len(post.call_args_list), 1)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_notification_sender.py::TestDingTalkChunk -v`
Expected: FAIL（`test_send_to_custom_chunks_dingtalk_by_config` 的 `assertGreater(... > 1)`：旧代码 `send_to_custom` 写死 `max_bytes=20000`，整篇一次发送，`post.call_args_list` 只有 1 条且 text 超 2400B）

- [ ] **Step 3: 实现配置化分片**

`custom_webhook_sender.py::__init__` 追加：

```python
        self._dingtalk_chunk_max_bytes = getattr(config, 'dingtalk_chunk_max_bytes', 2000)
```

`send_to_custom` 中两处 `max_bytes=20000`（[custom_webhook_sender.py:78](src/notification_sender/custom_webhook_sender.py#L78) 与 [:83](src/notification_sender/custom_webhook_sender.py#L83)）改为 `max_bytes=self._dingtalk_chunk_max_bytes`（`notification.py:389` 那一处在 Step 4 处理）。

`_send_dingtalk_chunked` 修改默认值与预算公式：

```python
    def _send_dingtalk_chunked(self, url: str, content: str, max_bytes: int = 2000) -> bool:
        import time as _time

        # 为 keyword + 分页标记预留空间，控制单条 ≤ max_bytes
        budget = max(1000, max_bytes - 400)
        chunks = chunk_content_by_max_bytes(content, budget)
```

- [ ] **Step 4: Stream 路径与 main.py 三处**

`src/notification.py` 约 389 行：

```python
                if self._send_dingtalk_chunked(session_webhook, content,
                                               max_bytes=self._dingtalk_chunk_max_bytes):
```

`main.py` 缠论（约 933-936 行）改为整篇一次推送：

```python
            if chan_report:
                if not args.no_notify:
                    notifier.send(chan_report)
                    logger.info("缠论分析推送完成")
                else:
                    logger.info("缠论分析完成（已跳过推送）")
```

`main.py` 飞龙（约 1044-1047 行）：

```python
                if not args.no_notify:
                    notifier.send(report)
                    logger.info("飞龙选股推送完成")
                else:
                    logger.info("飞龙选股完成（已跳过推送）")
```

`main.py` 一指（约 1147-1150 行）：

```python
                if not args.no_notify:
                    notifier.send(report)
                    logger.info("一阳指选股推送完成")
                else:
                    logger.info("一阳指选股完成（已跳过推送）")
```

> 说明：`notifier.send()` 内部各渠道各自分片（Telegram 4096/飞书 20KB/企微 2KB/钉钉走 `_send_dingtalk_chunked` 2000B），水印由 `send()` 统一追加一次，不再重复。

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_notification_sender.py tests/test_notification.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/notification_sender/custom_webhook_sender.py src/notification.py main.py tests/test_notification_sender.py
git commit -m "fix: configurable DingTalk chunk size to avoid folding"
```

---

### Task 8: 集成回归 + 文档收尾

**Files:**
- Test: 全量 `tests/`
- Modify: `docs/CHANGELOG.md`（补块 B/C 条目）、相关 `docs/*.md`

- [ ] **Step 1: 运行 CI 门禁**

Run: `./scripts/ci_gate.sh`
Expected: PASS

- [ ] **Step 2: 运行非网络测试**

Run: `python -m pytest -m "not network" -q`
Expected: PASS（新增测试 + 全量回归）

- [ ] **Step 3: 验证编译与 import**

Run: `python -m py_compile data_provider/kpl_client.py src/market_analyzer.py src/config.py main.py src/notification_sender/custom_webhook_sender.py`
Expected: PASS

- [ ] **Step 4: 手动冒烟（需用户配合，网络环境）**

Run: `python main.py --market-review --dry-run`（KPL 走真实接口）
Expected: 报告包含 4 个新段、催化线索只留标题、无异常报错；日志出现 `[KPL]` 来源标记。
若 KPL 接口被封：日志 `[KPL]` 告警后自动回退 akshare，报告结构不受影响。

- [ ] **Step 5: 补 CHANGELOG**

`docs/CHANGELOG.md` `[Unreleased]` 追加：

```
- [改进] 大盘统计与概念榜改用 KPL 数据源，akshare 兜底
- [新功能] 推送新增概念天梯/市场情绪/百日新高/大面股风险段落
- [改进] 近三日催化线索只保留标题，缩减报告长度
- [修复] 钉钉消息分片改为可配置大小(默认2000B)，解决消息折叠
```

- [ ] **Step 6: 同步文档**

核对 `docs/*.md` 中提及推送段落/数据源/配置项的部分，按第 7 节配置表同步；中英双语文档若有英文版需评估同步，未同步需在交付说明写明原因。

- [ ] **Step 7: Commit**

```bash
git add docs/CHANGELOG.md docs
git commit -m "docs: update changelog and docs for KPL integration"
```

---

## 自检记录

- **设计覆盖**：设计第 3 节（KPLClient/接口映射/降级）→ Task 1-5；第 4 节（四个新段）→ Task 4-5；第 5 节（瘦身）→ Task 6；第 6 节（钉钉分片）→ Task 7；第 7 节（配置）→ Task 1；第 9 节（验证）→ Task 8。
- **细化声明**：`get_limit_up_pool` 用 `GetPlateInfo_w38` 单请求实现（复盘数据，含板块名），优于 `DailyLimitPerformance` 五连请求——设计第 3.2 表的意图（替换脆弱涨停池 + 补概念天梯）完整落实，在 Task 4 已注明。
- **类型一致性**：`KPLClient._post`/`get_*` 返回类型在 Task 1-5 间一致；`LimitUpStock.limit_reason` 默认值向后兼容；`_get_kpl_client` 在 Task 2 定义、Task 3-5 复用。
