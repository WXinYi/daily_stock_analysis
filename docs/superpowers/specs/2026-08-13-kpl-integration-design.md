# KPL 接口集成 + 推送优化 设计文档

- 日期：2026-08-13
- 状态：已获用户确认（2026-08-13）
- 范围：大盘复盘（`main.py --market-review` 定时推送）+ 三个 CLI 模式（缠论/飞龙/一指）的推送链路
- 配套 KPL 接口文档：`/tmp/kpl_api.md`（来源 https://github.com/LowellLee/kpl）

---

## 1. 背景与目标

当前钉钉推送存在三个问题：

1. **数据源脆弱**：板块/概念/连板数据全部依赖 akshare（同花顺/东财接口），容易被反爬封禁；`_fetch_concept_rankings()` 逐个拉取 375 个概念，单次耗时数分钟。
2. **内容缺项**：`concept_ladders` 字段永远为空（设计有、实现没有）；缺情绪定量指标、题材涨停原因、百日新高、大面股风险等短线客关注的高价值数据。
3. **消息折叠**：钉钉分片预算写死 18500B/4096B，单条消息过大，客户端折叠。

目标：接入 KPL（开盘啦）免费私有接口替换/补强 akshare 数据，同时给推送瘦身 + 修复钉钉折叠。

---

## 2. 总体设计（三块）

| 块 | 内容 | 触达 |
|---|---|---|
| A | KPL 数据源接入（KPL 主 + akshare 兜底），新增 4 类推送段落，**大盘统计替换** | 数据层 |
| B | 推送瘦身：近三日催化线索只留标题 | 报告生成 |
| C | 钉钉消息分片修复：配置化分片 + 整体编号 | 推送链路 |

> **范围确认（2026-08-13）**：用户确认本次纳入「大盘统计替换」；北向资金按板块复活、龙虎榜、短线精灵/人气热榜、全球指数等列为**后续阶段**，见第 11 节。

---

## 3. 块 A：KPL 数据源接入

### 3.1 KPLClient 新模块

新建 `data_provider/kpl_client.py`（跟随现有 `data_provider/` 目录边界），封装 KPL 私有接口：

- **Base URL 族**（按接口区分）：
  - `https://apphwshhq.longhuvip.com`（实时）
  - `https://apphq.longhuvip.com`（实时）
  - `https://apphis.longhuvip.com`（历史）
- **公共参数**：`DeviceID=d66474b3-fd78-3a95-a56d-76e29e765ea3`（文档公开演示 ID，无需鉴权）、`PhoneOSNew=1`、`VerSion` 沿用文档各接口值。
- **Headers**（所有接口一致，文档实测有效）：
  - `Content-Type: application/x-www-form-urlencoded; charset=UTF-8`
  - `User-Agent: Dalvik/2.1.0 (Linux; U; Android 14; V2178A Build/UP1A.231005.007)`
- **代理处理**：KPL 是大陆服务，必须直连。在 `src/config.py` 的 `domestic_domains`（NO_PROXY 列表）追加 `longhuvip.com`，防止 `HTTP_PROXY` 把请求送出国导致地域封锁（与东财 MX 教训一致，见记忆 `mx-skill-geoblock-abandoned`）。
- **超时/重试**：`timeout=10s`，失败走 akshare 兜底；连续失败 3 次后本次运行内熔断该接口，避免拖慢主流程。

### 3.2 接口映射表（现状 → KPL）

| 现状实现 | 现有数据源 | KPL 替换 | 说明 |
|---|---|---|---|
| `get_market_stats()`（[akshare_fetcher.py:1623](data_provider/akshare_fetcher.py#L1623)，东财全市场快照→新浪兜底） | akshare 全市场行情 | `MoodNumCount`（`c=MarketMood`：上涨/下跌/涨停/跌停家数 + 全市场量能 qscln）+ `RiseFallAnalysis`（`c=HomeDingPan`：涨停/跌停/自然涨停/破板率/炸板数） | **大盘统计替换（本次纳入）**。一次请求替代全市场快照，顺带获得自然涨停数/破板率新指标。`flat_count` 无直接字段（置 0 或省略）；`qscln` 单位需实测换算为亿元 |
| `_fetch_concept_rankings()`（[market_analyzer.py:429](src/market_analyzer.py#L429)，375 概念逐只拉取，慢） | akshare THS | `RealRankingInfo`（精选板块列表-实时，`a=RealRankingInfo&st=60&Type=1&ZSType=7`） | 一次返回全量板块：名称/强度/涨幅/涨速/成交额/主力净额/量比，直接可排序取 Top5。替换概念涨跌榜，顺带获得**板块强度**字段 |
| `get_limit_up_pool()`（[akshare_fetcher.py:1824](data_provider/akshare_fetcher.py#L1824)） | akshare EM 涨停池 | `DailyLimitPerformance`（实时连板涨停，`a=DailyLimitPerformance&PidType=1..5`） | 按 PidType=1/2/3/4/5 分别拉首板~更高板，每只含：代码/名称/涨停时间/涨停原因/封单/实际流通/换手/所属板块代码 |
| 概念天梯（`concept_ladders` 恒空） | 无 | `GetPlateInfo_w38`（涨停复盘-复盘啦，`a=GetPlateInfo_w38&c=DailyLimitResumption`） | 返回按板块分组的涨停列表（板块名 + 每只涨停的原因/连板数/封单），直接可构建**概念天梯** |
| 情绪段缺失 | 无 | `RiseFallAnalysis` / `MoodNumCount` / `GetPlate_Info_QJ`(PlateID 801900/801902/801903) / `ChangeStatistics` | 涨停跌停数、破板率、上涨下跌家数、昨日涨停/连板/破板今日表现、情绪值+连板高度 |
| 题材原因缺失 | 无 | `GetKLineZhangTing`（涨停原因，`c=StockLineData`） | 个股开盘啦涨停原因（Reason 字段，含催化事件），喂给 LLM 提升概念主线分析 |
| 百日新高/大面股缺失 | 无 | `GroupCount_w28`（百日新高，`c=StockNewHigh`）/ `GetPMSL_KQXY`（大面股，`c=FuPanLa`） | 板块百日新高家数+比率；当日大面股（-10%左右）名单及所属概念 |

### 3.3 降级策略（KPL 主 + akshare 兜底）

- 每个 KPL 抓取函数独立 try/except，失败即调用现有 akshare 实现（保持不变）。
- KPL 成功时用 KPL 数据（更全更稳）；失败时静默回退，日志记录 `[KPL]` 来源。
- 概念涨跌榜在 KPL `RealRankingInfo` 失败时回退到现有 `_fetch_concept_rankings()`（akshare THS）。
- 连板天梯在 KPL `DailyLimitPerformance` + `GetPlateInfo_w38` 失败时回退到现有 `get_limit_up_pool()`。
- **大盘统计**在 KPL `MoodNumCount`/`RiseFallAnalysis` 失败时回退到现有 `get_market_stats()`（akshare 东财→新浪）。`flat_count` 取 KPL 无来源时置 0，不影响涨跌占比计算。
- 新增段落（情绪指标/百日新高/大面股）无 akshare 等价物，失败则整段省略并记日志（报告不报错）。

---

## 4. 块 A：新增推送段落

### 4.1 概念天梯（补齐空缺字段）

`LimitUpLadder.concept_ladders` 从恒空改为填充：

- 数据源：`GetPlateInfo_w38` 涨停复盘，按 `ZSName`（板块）分组，统计每板块涨停数与各连板数分布。
- 结构复用 `industry_ladders` 现有 schema：`{sector, total, stocks_by_board}`，取涨停数 Top5。
- 展示：在现有「连板天梯 · 行业板块」之后新增「连板天梯 · 概念板块」，格式一致。
- 行业天梯（akshare 所属行业）与概念天梯（题材）信息不同，两者并存，用户已确认。

### 4.2 情绪定量指标（新段「市场情绪」）

汇总 KPL 情绪接口，输出紧凑一段：

- 涨停/跌停/自然涨停/炸板数/破板率（`RiseFallAnalysis`）
- 情绪值 strong + 连板高度 lbgd + 大幅回撤 df_num（`ChangeStatistics`），并附文档给出的情绪阈值提示（>75 亏钱效应风险 / <25 反弹需求）
- 昨日涨停今表现 / 昨日连板今表现 / 昨日破板今日表现（`GetPlate_Info_QJ` PlateID=801900/801902/801903）
- 目标体积 ≤ 500B，单段表格/列表，注入到「四、资金与情绪」段附近。

### 4.3 题材热点解读（强化「三、概念主线」）

- `GetKLineZhangTing` 取高度板/领涨涨停股的开盘啦涨停原因，替换现有个股零原因罗列，喂给 `_build_review_prompt`（[market_analyzer.py:1115](src/market_analyzer.py#L1115)）的概念主线 prompt 块。
- 推送侧不新增长段，只让 LLM 结论更有依据；LLM 输出「三、概念主线」段会因输入更实而略长（长度预算按 +300B 估）。

### 4.4 百日新高板块 + 大面股风险（新段）

- 百日新高（`GroupCount_w28`）：板块名 + 「新高家数,家数占比%」，取 Top5，展示强势方向扩散。
- 大面股（`GetPMSL_KQXY`）：当日跌幅居前且接近跌停的名单（代码/名称/跌幅/概念），Top5，配合现有风险提示。
- 两段合并为一个「新高 / 大面股」紧凑段，目标体积 ≤ 400B，注入「五、消息催化」之后。

---

## 5. 块 B：推送瘦身

`_build_news_block()`（[market_analyzer.py:1021](src/market_analyzer.py#L1021)）：「近三日催化线索」由 `标题 — 120字摘要` 改为**只留标题**（用户已确认）。snippet 截断逻辑移除，每条一行。

- 现占用 1785B（报告 8431B 的 21%），瘦身后 ≈ 450B，净省 **~1.3KB**。
- LLM prompt 侧的 `news_text` **保持原文摘要**（LLM 需要全文信息，不影响分析质量），只改推送展示。

---

## 6. 块 C：钉钉消息分片修复

### 根因

`_send_dingtalk_chunked`（[custom_webhook_sender.py:263](src/notification_sender/custom_webhook_sender.py#L263)）预算写死 `max_bytes=20000 → budget=18500`；三个 CLI 模式又在 [main.py:934](main.py#L934)/[1045](main.py#L1045)/[1148](main.py#L1148) 预切成 4096B 后再逐段 `notifier.send()`。单条钉钉消息过大 → 客户端折叠；且每段重复水印、局部编号。

### 修复（4 点）

| # | 改动 | 文件 |
|---|---|---|
| 1 | 新增配置 `dingtalk_chunk_max_bytes`（默认 **2000**，`DINGTALK_CHUNK_MAX_BYTES`） | `src/config.py`（字段 + `_load_from_env`） |
| 2 | `_send_dingtalk_chunked` 改用该配置；预算公式 `budget = max(1000, max_bytes - 400)`（预留 keyword + 分页标记），`max_bytes` 默认值同步改为 2000 | `src/notification_sender/custom_webhook_sender.py`（`send_to_custom` 内 3 处写死 20000 → `self._dingtalk_chunk_max_bytes`；`_send_dingtalk_chunked` 默认参数） |
| 3 | Stream 路径同样传配置 | `src/notification.py:389` 的 `_send_dingtalk_chunked(session_webhook, content, ...)` |
| 4 | 去掉 main.py 三个 CLI 模式（缠论/飞龙/一指）的 4096 预切分，整篇报告一次 `notifier.send()`（对齐定时主流程 [main.py:537](main.py#L537) 现有行为） | `main.py` 三处 |

### 效果

- 8.4KB 复盘 → 钉钉约 **5 条 ~2KB 消息**，带 `📄 (1/5)~(5/5)` 全局分页标记、1s 间隔顺序发送，不折叠。
- 水印 `Strategy by Wxy · date` 每条报告只出现一次（`notifier.send()` 内部统一追加）。
- 其他渠道（Telegram 4096 / 飞书 ~20KB / 企微 ~2KB）本来就自管分片，**不受影响**，反而消除了 main.py 预切分的重复水印。

---

## 7. 配置变更汇总

| 配置项 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `kpl_enabled` | `KPL_ENABLED` | `true` | KPL 总开关，关闭则全走 akshare |
| `kpl_timeout_seconds` | `KPL_TIMEOUT_SECONDS` | `10` | 单接口超时 |
| `dingtalk_chunk_max_bytes` | `DINGTALK_CHUNK_MAX_BYTES` | `2000` | 钉钉单条消息字节上限（防折叠） |
| `kpl_device_id` | `KPL_DEVICE_ID` | 文档公开演示值 | KPL 请求 DeviceID，可覆盖 |
| `dingtalk_webhook_keyword` | `DINGTALK_WEBHOOK_KEYWORD` | 现有 | 未变，沿用 |

同步更新：`.env.example`、`docs/CHANGELOG.md`（`[Unreleased]` 扁平条目）、相关 `docs/*.md`。`src/config.py` 的 `domestic_domains` 追加 `longhuvip.com`。

---

## 8. 长度预算

| 项 | 变更 | 体积 |
|---|---|---|
| 现状复盘报告 | — | 8431B |
| 近三日催化线索瘦身 | -1335B | 7096B |
| 情绪定量指标 | +500B | ~7596B |
| 概念天梯 | +500B | ~8096B |
| 百日新高+大面股 | +400B | ~8496B |
| 题材热点（推送侧） | +300B | ~8796B |
| **合计** | | **~8.8KB ≈ 5 条钉钉消息** |

---

## 9. 验证方案

1. **接口验证**（`pytest -m network` 或本地脚本）：逐个 KPL 接口返回结构符合文档字段（设计期已用文档 DeviceID 实测 7 个接口可用）。
2. **降级验证**：设 `KPL_ENABLED=false` 跑 `--market-review --dry-run`，确认报告与现状一致；KPL 单接口人为断连，确认自动回退 akshare。
3. **大盘统计验证**：对比 KPL `MoodNumCount`/`RiseFallAnalysis` 与 `get_market_stats()` 的涨跌家数/涨停跌停/成交额，确认口径一致（量能 qscln 单位换算为亿元后误差 ≤ 5%）。
4. **报告结构验证**：跑 `--market-review` 生成报告，检查 4 个新段与瘦身后的催化线索段落。
5. **分片验证**：`_send_dingtalk_chunked` 单测 + 实际推送到测试群，确认 ~5 条 2KB 消息、`📄 (i/n)` 编号正确、无折叠。
6. **回归**：`./scripts/ci_gate.sh` + `python -m pytest -m "not network"`；确认 Telegram/飞书/企微路径不受影响。

---

## 10. 风险与回滚

| 风险 | 应对 |
|---|---|
| KPL 为私有接口，字段/域名随时可能变更 | 全 try/except + akshare 兜底；`kpl_enabled=false` 一键回退 |
| KPL 被地域封锁（非大陆 IP 403，如东财 MX 前例） | `longhuvip.com` 强制直连（NO_PROXY），不走海外代理；仍失败则走兜底 |
| DeviceID 无鉴权，属文档公开演示值 | 提供 `KPL_DEVICE_ID` 可配置覆盖 |
| 分片过小导致消息条数多 | 默认 2000B 可调（`DINGTALK_CHUNK_MAX_BYTES`） |
| 新增段落无 akshare 等价物 | 失败整段省略，报告不报错 |

回滚方式：`KPL_ENABLED=false`（数据层全部回退 akshare）+ 删除 4 个新段渲染分支；分片改动集中在配置项与 `_send_dingtalk_chunked`，回退 `dingtalk_chunk_max_bytes=20000` 即恢复原行为。

---

## 11. 后续阶段（本次不实施）

用户确认以下 KPL 能力列为后续阶段，不在本设计实施，避免范围膨胀：

| 能力 | KPL 接口 | 价值 | 前置验证 |
|---|---|---|---|
| 北向资金按板块复活（季度持股排名，非单日净流入） | `GGList_BXZJ` | 恢复被废弃的 `_get_north_flow()`，展示北向重仓板块 | 确认季度语义与现有字段兼容 |
| 龙虎榜（含买入/卖出营业部） | `龙虎榜接口` | 游资/机构席位追踪，现有推送无此数据 | 需实测返回结构与鉴权 |
| 短线精灵 / 盘中人气热榜 | `Radar` / `GetHotPHB` | 盘中异动快讯，可做盘中推送 | 需接入盘中定时链路 |
| 全球指数（港股/美股上下文） | `GlobalCommon` | 补消息催化的指数行情数据点 | 文档示例带 `UserID`+`Token`，**需验证是否强制鉴权** |
