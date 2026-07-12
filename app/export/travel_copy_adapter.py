from __future__ import annotations

from typing import Any


def highlight_text(value: str) -> str:
    text = _clean(value).lower()
    rules = [
        (("typical spanish cuisine", "local spanish food", "spanish food", "restaurants"), "authentische lokale Küche"),
        (("football", "stadium", "bernabeu", "bernabéu"), "Fußballgeschichte und Stadionkultur"),
        (("architecture", "architektur", "viewpoint", "aussicht"), "besondere Architektur und Aussichtspunkte"),
        (("nature", "park", "garden", "natur"), "Parks, Grünflächen und ruhige Pausen"),
        (("museum", "museen"), "ausgewählte Museen und Kulturorte"),
        (("anime", "manga"), "Anime-, Manga- und Popkultur-Orte"),
        (("gaming", "game"), "Gaming-Orte und interaktive Erlebnisse"),
        (("market", "street food", "markt"), "Märkte, Streetfood und lokale Spezialitäten"),
    ]
    for keywords, label in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return _sentence_case(_clean(value))


def avoid_text(value: str) -> str:
    text = _clean(value).lower()
    if text in {"clubs", "club", "nightclubs", "nachtclubs"}:
        return "Nachtclubs und lautes Partyprogramm"
    if "sport" in text:
        return "sportliche Aktivitäten"
    if "tourist" in text:
        return "klassische Touristenfallen"
    return _sentence_case(_clean(value))


def category_label(activity: dict[str, Any], details: dict[str, str]) -> str:
    raw = _clean(activity.get("category")).lower()
    name = _clean(activity.get("name")).lower()
    address = _clean(details.get("address")).lower()
    haystack = " ".join([raw, name, address, _clean(details.get("matched_must_have")).lower()])

    if any(word in haystack for word in ["market", "mercado", "markt"]):
        return "Tapas & Markt"
    if any(word in haystack for word in ["restaurant", "food", "cuisine", "taberna", "casa", "venta"]):
        return "Restaurant"
    if any(word in haystack for word in ["football", "stadium", "bernabeu", "bernabéu", "sport"]):
        return "Fußballerlebnis"
    if any(word in haystack for word in ["park", "garden", "nature", "retiro", "natur"]):
        return "Park & Erholung"
    if any(word in haystack for word in ["architecture", "architektur", "templo", "rooftop", "view", "aussicht"]):
        return "Architektur"
    if any(word in haystack for word in ["museum", "culture", "kultur"]):
        return "Kultur"
    if any(word in haystack for word in ["shop", "shopping", "store"]):
        return "Shopping"
    return "Besonderer Stopp"


def category_key(label: str) -> str:
    text = label.lower()
    if "restaurant" in text or "tapas" in text:
        return "food"
    if "park" in text:
        return "nature"
    if "fußball" in text:
        return "sport"
    if "kultur" in text or "architektur" in text:
        return "culture"
    if "shopping" in text:
        return "shopping"
    return "activity"


def activity_description(activity: dict[str, Any], details: dict[str, str], destination: str) -> str:
    name = _clean(activity.get("name"))
    lower_name = name.lower()
    label = category_label(activity, details)

    if "casa alberto" in lower_name:
        return "Starte in einem traditionsreichen Lokal, das seit Generationen für klassische Madrider Küche steht."
    if "mercado de san miguel" in lower_name:
        return "Ideal für eine lockere Pause mit Tapas, regionalen Spezialitäten und viel Atmosphäre unter einem Dach."
    if "legends" in lower_name and "football" in lower_name:
        return "Hier tauchst du mitten in die Geschichte des Fußballs ein, mit Erinnerungsstücken und großen Momenten."
    if "bernab" in lower_name:
        return "Ein Pflichtstopp für Fußballfans: Stadiongefühl, Vereinsgeschichte und ein Blick hinter die Kulissen."
    if "retiro" in lower_name:
        return "Eine ruhige Pause zwischen Alleen, Grünflächen und den bekannten Ecken des Retiro-Parks."
    if "templo de debod" in lower_name:
        return "Ein ungewöhnlicher Ort für den Tagesausklang, besonders schön mit Blick über die Stadt."
    if "rooftop" in lower_name or "azotea" in lower_name:
        return "Ein guter Abschluss mit weitem Blick über die Dächer und einem entspannten Wechsel der Perspektive."

    if label == "Restaurant":
        return f"Dieser Stopp bringt lokale Küche in deinen Tag und gibt dir einen genussvollen Eindruck von {destination}."
    if label == "Tapas & Markt":
        return "Hier kannst du unkompliziert probieren, vergleichen und dich durch mehrere kleine Spezialitäten treiben lassen."
    if label == "Fußballerlebnis":
        return "Dieser Stopp ergänzt die Reise mit Stadiongefühl, Fußballkultur und einem klaren lokalen Bezug."
    if label == "Park & Erholung":
        return "Dieser Ort schafft eine entspannte Pause im Tagesablauf und bringt etwas Grün in die Route."
    if label == "Architektur":
        return "Dieser Stopp setzt einen architektonischen Akzent und eignet sich gut für Fotos und Orientierung in der Stadt."
    if label == "Kultur":
        return "Dieser Ort ergänzt den Tag mit kulturellem Kontext und einem ruhigeren Programmpunkt."
    return f"Dieser Stopp ergänzt den Tagesplan sinnvoll und bietet dir einen abwechslungsreichen Eindruck von {destination}."


def day_title(day_number: int, activities: list[dict[str, Any]], destination: str) -> str:
    labels = [activity.get("category", "") for activity in activities]
    joined = " ".join(labels).lower()
    if "fußball" in joined and ("restaurant" in joined or "tapas" in joined):
        return "Stadionkultur und lokale Küche"
    if "fußball" in joined and "park" in joined:
        return "Stadionluft, Stadtgrün und Ausblicke"
    if "restaurant" in joined and "kultur" in joined:
        return f"Kulinarischer Auftakt in {destination}"
    if "architektur" in joined and "restaurant" in joined:
        return "Architektur, Aussicht und gutes Essen"
    if "park" in joined:
        return "Ruhige Orte und entspannte Wege"
    return f"Tag {day_number} in {destination}"


def planning_text(ok: bool, total_cost: float | None, maximum_budget: float | None, currency: str) -> str:
    if ok and total_cost is not None and maximum_budget:
        reserve = max(0, maximum_budget - total_cost)
        return (
            f"Der Plan passt gut zu deinen Wünschen und bleibt im Budgetrahmen. "
            f"Es bleiben rund {_format_money(reserve, currency)} für spontane Stopps, Fahrten oder kleine Extras."
        )
    if ok:
        return "Der Plan ist sinnvoll über die Reisetage verteilt und passt zu deinen ausgewählten Interessen."
    return "Einige Punkte sollten vor der Reise noch geprüft werden, damit der Ablauf wirklich rund bleibt."


def plan_stats(days: list[dict[str, Any]], total_cost: float | None, maximum_budget: float | None, currency: str) -> list[dict[str, str]]:
    activity_count = sum(len(day.get("activities") or []) for day in days)
    total_hours = sum(day.get("total_duration_hours") or 0 for day in days)
    stats = [
        {"label": "Ausgewählte Stopps", "value": str(activity_count)},
        {"label": "Reisetage", "value": f"{len(days)} abwechslungsreiche Tage"},
        {"label": "Programmzeit", "value": _format_duration(total_hours)},
    ]
    if total_cost is not None:
        stats.append({"label": "Geplante Ausgaben", "value": _format_money(total_cost, currency)})
    if total_cost is not None and maximum_budget:
        reserve = max(0, maximum_budget - total_cost)
        stats.append({"label": "Freier Spielraum", "value": _format_money(reserve, currency)})
    return stats


def final_notes(total_cost: float | None, maximum_budget: float | None, currency: str) -> list[str]:
    notes = [
        "Prüfe Öffnungszeiten, Eintrittspreise und Reservierungen am besten am Vortag noch einmal.",
        "Plane zwischen den Stopps etwas Puffer ein, besonders bei Restaurants, Museen und Stadiontouren.",
    ]
    if total_cost is not None and maximum_budget:
        reserve = max(0, maximum_budget - total_cost)
        notes.append(f"Mit diesem Plan bleiben rund {_format_money(reserve, currency)} als Reserve für spontane Wünsche.")
    return notes


def clean_public_text(value: Any) -> str:
    text = _clean(value)
    replacements = {
        "Budget utilization is meaningful and within target range": "Das Budget bleibt angenehm im Rahmen",
        "validation is clean": "der Ablauf wirkt stimmig",
        "quality review": "Reiseeinschätzung",
        "nature experiences": "Naturerlebnisse",
        "typical Spanish cuisine": "authentische spanische Küche",
        "watch a football match in Madrid": "Fußballkultur in Madrid",
        "architecture experiences": "besondere Architektur",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _sentence_case(text: str) -> str:
    cleaned = _clean(text)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else ""


def _clean(value: Any) -> str:
    text = " ".join(str(value or "").replace("\x00", "").split())
    return "" if text.lower() in {"none", "null", "nan"} else text


def _format_money(value: float | None, currency: str) -> str:
    if value is None:
        return ""
    return f"{value:,.0f} {currency}".replace(",", ".")


def _format_duration(hours: float | None) -> str:
    if hours is None:
        return ""
    if abs(hours - 1) < 0.01:
        return "1 Stunde"
    formatted = f"{hours:.1f}".replace(".", ",").rstrip("0").rstrip(",")
    return f"{formatted} Stunden"
