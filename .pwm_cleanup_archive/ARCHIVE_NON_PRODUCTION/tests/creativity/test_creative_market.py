import json
from pathlib import Path

from creativity.creative_market import CreativeMarket


def test_export_and_reputation(tmp_path: Path):
    export_file = tmp_path / "market.jsonl"
    market = CreativeMarket(export_file)

    item = market.export_item("sunset over the sea", "poetry")

    assert export_file.exists()
    with export_file.open() as f:
        line = f.readline()
    data = json.loads(line)

    assert data["item_id"] == item.item_id
    assert "glyph" in data
    assert item.tag.id
    assert market.reputation_store[item.item_id] == item.reputation
