from __future__ import annotations

from app.tools.places_tool import _normalize_category


def test_food_market_query_classifies_market_as_food():
    category = _normalize_category(
        types=["market", "food", "point_of_interest", "establishment"],
        query="Madrid local food markets",
        name="Mercado de San Miguel",
    )

    assert category == "food"


def test_shopping_query_classifies_store_as_shopping():
    category = _normalize_category(
        types=["store", "book_store", "point_of_interest", "establishment"],
        query="Tokyo manga shops",
        name="Manga Store",
    )

    assert category == "shopping"


def test_plain_market_with_shopping_query_stays_shopping():
    category = _normalize_category(
        types=["market", "store", "point_of_interest", "establishment"],
        query="Madrid local markets and specialty shops",
        name="El Rastro",
    )

    assert category == "shopping"
