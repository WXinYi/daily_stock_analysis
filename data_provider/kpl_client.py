# -*- coding: utf-8 -*-
"""KPL(开盘啦 longhuvip) 私有接口客户端。

无需鉴权，使用文档公开 DeviceID。所有接口大陆直连（NO_PROXY 已含 longhuvip.com）。
单一 HTTP 入口 `_post`，各业务方法负责解析；连续失败 3 次对该动作熔断。
"""
import logging
from typing import Any, Dict, List, Optional

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
