#!/usr/bin/env python3
"""将每日选股结果合并到历史文件，按 (code, date) 去重。"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=["dragon", "one-finger"])
    args = parser.parse_args()

    pick_type = args.type
    today_file = os.path.join(ROOT, "data", f"{'dragon' if pick_type == 'dragon' else 'one_finger'}_picks_today.json")
    history_file = os.path.join(ROOT, "data", f"{'dragon' if pick_type == 'dragon' else 'one_finger'}_picks_history.json")

    if not os.path.exists(today_file):
        print(f"今日文件不存在: {today_file}, 跳过存档")
        return

    with open(today_file, "r", encoding="utf-8") as f:
        today_picks = json.load(f)

    if not today_picks:
        print("今日无选股结果，跳过存档")
        return

    history_picks = []
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            history_picks = json.load(f)

    existing = {(p["code"], p["date"]) for p in history_picks}
    added = 0
    for pick in today_picks:
        key = (pick["code"], pick["date"])
        if key not in existing:
            history_picks.append(pick)
            existing.add(key)
            added += 1

    # 按日期倒序
    history_picks.sort(key=lambda p: p.get("date", ""), reverse=True)

    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_picks, f, ensure_ascii=False, indent=2)

    print(f"存档完成: 新增 {added} 条, 历史共 {len(history_picks)} 条")


if __name__ == "__main__":
    main()
