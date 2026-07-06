def clean_text(text: str) -> str:
    """
    Clean a text by removing leading/trailing spaces and converting to lowercase.

    Args:
        text: The raw text to clean.

    Returns:
        The cleaned text.
    """
    text = text.strip()
    text = text.lower()

    return text


# if __name__ == "__main__":
#     print(clean_text("  FLOOD IN GERMANY  "))
#     print(clean_text("  ALERT: Heavy Rain  "))


def clean_alert(alert: dict) -> dict:
    """
    Clean an alert dictionary by removing spaces and converting to lowercase.

    Args:
        alert: The raw alert dictionary to clean.

    Returns:
        The cleaned alert dictionary.
    """
    cleaned_alert = {}
    cleaned_alert["titre"] = clean_text(alert.get("titre", ""))
    cleaned_alert["pays"] = clean_text(alert.get("pays", ""))
    cleaned_alert["severite"] = clean_text(alert.get("severite", ""))

    return cleaned_alert


if __name__ == "__main__":
    alerte = {
        "titre": "  FLOOD IN GERMANY  ",
        "pays": "  germany  ",
        "severite": "  ORANGE  ",
    }
    print(clean_alert(alerte))
