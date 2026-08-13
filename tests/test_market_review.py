# -*- coding: utf-8 -*-
"""Tests for localized market review wrappers."""

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

def _build_optional_module_stubs() -> dict[str, ModuleType]:
    stubs: dict[str, ModuleType] = {}
    google_module: ModuleType | None = None

    for module_name in ("google.generativeai", "google.genai", "anthropic"):
        try:
            importlib.import_module(module_name)
            continue
        except ImportError:
            stub = ModuleType(module_name)
            stubs[module_name] = stub
            if not module_name.startswith("google."):
                continue
            if google_module is None:
                try:
                    google_module = importlib.import_module("google")
                except ImportError:
                    google_module = ModuleType("google")
                    stubs["google"] = google_module
            setattr(google_module, module_name.split(".", 1)[1], stub)

    return stubs


sys.modules.update(_build_optional_module_stubs())
import src.core.market_review as market_review_module

run_market_review = market_review_module.run_market_review


class MarketReviewLocalizationTestCase(unittest.TestCase):
    def _make_notifier(self) -> MagicMock:
        notifier = MagicMock()
        notifier.save_report_to_file.return_value = "/tmp/market_review.md"
        notifier.is_available.return_value = True
        notifier.send.return_value = True
        return notifier

    def test_run_market_review_uses_english_notification_title(self) -> None:
        notifier = self._make_notifier()
        market_analyzer = MagicMock()
        market_analyzer.run_daily_review.return_value = "## 2026-04-10 A-share Market Recap\n\nBody"

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="en", market_review_region="cn"),
        ), patch.object(market_review_module, "MarketAnalyzer", return_value=market_analyzer):
            result = run_market_review(notifier, send_notification=True)

        self.assertEqual(result, "## 2026-04-10 A-share Market Recap\n\nBody")
        saved_content = notifier.save_report_to_file.call_args.args[0]
        self.assertTrue(saved_content.startswith("# 🎯 Market Review\n\n"))
        sent_content = notifier.send.call_args.args[0]
        self.assertTrue(sent_content.startswith("🎯 Market Review\n\n"))
        self.assertTrue(notifier.send.call_args.kwargs["email_send_to_all"])

    def test_run_market_review_merges_both_regions_with_english_wrappers(self) -> None:
        notifier = self._make_notifier()
        cn_analyzer = MagicMock()
        cn_analyzer.run_daily_review.return_value = "CN body"
        hk_analyzer = MagicMock()
        hk_analyzer.run_daily_review.return_value = "HK body"
        us_analyzer = MagicMock()
        us_analyzer.run_daily_review.return_value = "US body"

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="en", market_review_region="both"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            side_effect=[cn_analyzer, hk_analyzer, us_analyzer],
        ):
            result = run_market_review(notifier, send_notification=False)

        self.assertIn("# A-share Market Recap\n\nCN body", result)
        self.assertIn("# HK Market Recap\n\nHK body", result)
        self.assertIn("> Next market recap follows", result)
        self.assertIn("# US Market Recap\n\nUS body", result)
        saved_content = notifier.save_report_to_file.call_args.args[0]
        self.assertTrue(saved_content.startswith("# 🎯 Market Review\n\n"))
        notifier.send.assert_not_called()

    def test_run_market_review_comma_joined_subset_cn_us(self) -> None:
        """Regression: compute_effective_region("both", {"cn","us"}) -> "cn,us"
        must produce A-share + US report without HK."""
        notifier = self._make_notifier()
        cn_analyzer = MagicMock()
        cn_analyzer.run_daily_review.return_value = "CN body"
        us_analyzer = MagicMock()
        us_analyzer.run_daily_review.return_value = "US body"

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="zh", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            side_effect=[cn_analyzer, us_analyzer],
        ):
            result = run_market_review(
                notifier, send_notification=False, override_region="cn,us"
            )

        self.assertIn("# A股大盘复盘\n\nCN body", result)
        self.assertIn("# 美股大盘复盘\n\nUS body", result)
        self.assertNotIn("港股", result)
        self.assertNotIn("HK", result)

    def test_run_market_review_comma_joined_subset_cn_hk(self) -> None:
        """Regression: compute_effective_region("both", {"cn","hk"}) -> "cn,hk"
        must produce A-share + HK report without US."""
        notifier = self._make_notifier()
        cn_analyzer = MagicMock()
        cn_analyzer.run_daily_review.return_value = "CN body"
        hk_analyzer = MagicMock()
        hk_analyzer.run_daily_review.return_value = "HK body"

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="zh", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            side_effect=[cn_analyzer, hk_analyzer],
        ):
            result = run_market_review(
                notifier, send_notification=False, override_region="cn,hk"
            )

        self.assertIn("# A股大盘复盘\n\nCN body", result)
        self.assertIn("# 港股大盘复盘\n\nHK body", result)
        self.assertNotIn("美股", result)
        self.assertNotIn("US Market", result)


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


if __name__ == "__main__":
    unittest.main()
