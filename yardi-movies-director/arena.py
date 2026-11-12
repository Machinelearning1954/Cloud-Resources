"""Arena: blind battles between your own models, scored by your own votes.

Two different models get the same prompt and seed. Their names stay hidden
until you pick a winner. Ratings are Elo, replayed from the stored votes each
time, so there is no separate rating state to drift out of sync.
"""
from __future__ import annotations

START = 1000.0
K = 32.0
SCORES = {"a": 1.0, "b": 0.0, "tie": 0.5}


def leaderboard(battles: list[dict], names: dict[str, str]) -> dict[str, list[dict]]:
    """Per kind, models ranked by rating. Only voted battles made from real model output count."""
    tables: dict[str, dict[str, dict]] = {}
    for battle in sorted((b for b in battles if b.get("winner") and b.get("counts")), key=lambda b: b["voted_at"]):
        table = tables.setdefault(battle["kind"], {})
        a, b = (
            table.setdefault(model, {"model": model, "name": names.get(model, model), "rating": START,
                                     "wins": 0, "losses": 0, "ties": 0})
            for model in (battle["a_model"], battle["b_model"])
        )
        score = SCORES[battle["winner"]]
        expected = 1.0 / (1.0 + 10 ** ((b["rating"] - a["rating"]) / 400.0))
        a["rating"] += K * (score - expected)
        b["rating"] -= K * (score - expected)
        if score == 0.5:
            a["ties"] += 1
            b["ties"] += 1
        else:
            winner, loser = (a, b) if score == 1.0 else (b, a)
            winner["wins"] += 1
            loser["losses"] += 1
    return {
        kind: [dict(row, rating=round(row["rating"])) for row in sorted(table.values(), key=lambda r: -r["rating"])]
        for kind, table in tables.items()
    }
