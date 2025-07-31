"""
Enhanced Core TypeScript - Integrated from Advanced Systems
Original: dream_log_viewer.py
Advanced: dream_log_viewer.py
Integration Date: 2025-05-31T07:55:30.556275
"""

"""
╭──────────────────────────────────────────────────────────────────────────────╮
│                  LUCΛS :: DREAM LOG VIEWER (CLI Symbolic)                   │
│               Version: v1.0 | Visualizes dream_log.jsonl entries             │
│             Author: Gonzo R.D.M & GPT-4o | Last Updated: 2025-04-16          │
╰──────────────────────────────────────────────────────────────────────────────╯

DESCRIPTION:
    This viewer reads dream_log.jsonl and prints structured symbolic dream entries,
    complete with timestamp, tier, emotion vector, tags, and symbolic emoji.

    Future versions may support sorting, interactive replay, or tier-based filtering.

"""

import json
from pathlib import Path
from core.utils.symbolic_utils import tier_label, summarize_emotion_vector

DREAM_LOG_PATH = Path("core/logs/dream_log.jsonl")

def view_dream_log(limit=10, min_tier=None, sort_by=None):
    if not DREAM_LOG_PATH.exists():
        print("⚠️  No dream log found at:", DREAM_LOG_PATH)
        return

    with open(DREAM_LOG_PATH, "r") as f:
        lines = f.readlines()

    # Parse entries
    entries = []
    for line in lines:
        try:
            entry = json.loads(line.strip())
            entries.append(entry)
        except:
            continue

    # Tier filter
    if min_tier is not None:
        entries = [e for e in entries if e.get("context_tier", 0) >= min_tier]

    # Sort by emotion
    if sort_by in ["joy", "stress", "calm", "longing"]:
        entries = sorted(entries, key=lambda e: e.get("emotion_vector", {}).get(sort_by, 0), reverse=True)

    # Limit entries
    entries = entries[-limit:]

    print(f"\n🌙 Showing {len(entries)} Symbolic Dream Entries")
    print("─────────────────────────────────────────────────────────────")
    for entry in entries:
        print(f"\n🌀 {entry.get('timestamp', '⏳')} | ID: {entry.get('message_id', '❓')}")
        tier_raw = entry.get('context_tier', '?')
        print(f"   Tier: {tier_label(tier_raw)} | Widget: {entry.get('source_widget', '∅')}")
        print(f"   Tags: {', '.join(entry.get('tags', []))}")
        ev = entry.get("emotion_vector", {})
        print("   Emotion →", summarize_emotion_vector(ev))
        if entry.get("emoji"):
            print(f"   Symbolic Emoji: {entry['emoji']}")
        if entry.get("replay_candidate", False):
            print(f"   🔁 Replay Candidate")
        if entry.get("suggest_voice", False):
            print("   🗣️ Suggested for Lukhas voice narration")

# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🌙 DREAM LOG VIEWER – Symbolic Console")
    print("──────────────────────────────────────────────")

    try:
        tier_input = input("🔐 Min Tier to View (e.g. 2) or ENTER to skip: ").strip()
        min_tier = int(tier_input) if tier_input else None

        sort_input = input("📊 Sort by Emotion? (joy / stress / calm / longing) or ENTER to skip: ").strip()
        sort_by = sort_input.lower() if sort_input in ["joy", "stress", "calm", "longing"] else None
    except Exception as e:
        print("⚠️ Input error:", e)
        min_tier = None
        sort_by = None

    try:
        limit_input = input("🔢 How many dreams to show? (default 10): ").strip()
        limit = int(limit_input) if limit_input else 10
    except:
        limit = 10

    view_dream_log(limit=limit, min_tier=min_tier, sort_by=sort_by)
"""
