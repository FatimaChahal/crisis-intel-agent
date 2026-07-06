from data.silver.models import Alert


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


def clean_alert(alert: dict) -> Alert:
    """
    Clean an alert dictionary and return a validated Alert model.

    Args:
        alert: The raw alert dictionary to clean.

    Returns:
        A validated Alert instance with cleaned fields.
    """
    return Alert(
        titre=clean_text(alert.get("titre", "")),
        pays=clean_text(alert.get("pays", "")),
        severite=clean_text(alert.get("severite", "")),
    )


if __name__ == "__main__":
    alerte_brute = {
        "titre": "  FLOOD IN GERMANY  ",
        "pays": "  germany  ",
        "severite": "  ORANGE  ",
    }
    print(clean_alert(alerte_brute))