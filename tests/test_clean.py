from data.silver.clean import clean_text, clean_alert


def test_clean_text_strips_spaces():
    """Test that clean_text removes leading and trailing spaces."""
    assert clean_text("  hello  ") == "hello"


def test_clean_text_lowercase():
    """Test that clean_text converts to lowercase."""
    assert clean_text("FLOOD") == "flood"


def test_clean_alert():
    """Test that clean_alert cleans all fields."""
    raw = {
        "titre": "  FLOOD IN GERMANY  ",
        "pays": "  GERMANY  ",
        "severite": "  ORANGE  ",
    }
    result = clean_alert(raw)
    assert result.titre == "flood in germany"
    assert result.pays == "germany"
    assert result.severite == "orange"
