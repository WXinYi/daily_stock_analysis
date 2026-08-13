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
