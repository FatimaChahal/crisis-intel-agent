import pandas as pd
import json
from data.silver.clean import clean_alert


def load_bronze(path: str = "data/bronze/alerts.csv") -> pd.DataFrame:
    """
    Load raw alerts data from Bronze layer.

    Args:
        path: Path to the CSV file.

    Returns:
        A raw DataFrame with alerts.
    """
    return pd.read_csv(path)

def load_bronze_json(path: str = "data/bronze/alerts.json") -> list:
    """
    Load raw alerts data from Bronze layer (JSON format).

    Args:
        path: Path to the JSON file.

    Returns:
        A list of raw alert dictionaries.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data

def clean_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw alerts DataFrame by stripping spaces and lowercasing.

    Args:
        df: The raw DataFrame from Bronze layer.

    Returns:
        The cleaned DataFrame for Silver layer.
    """
    for col in df.columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.lower()
    return df

def clean_bronze_json(alerts: list) -> list:
    """
    Clean a list of raw alert dictionaries and validate with Pydantic.

    Args:
        alerts: List of raw alert dictionaries from Bronze layer.

    Returns:
        List of validated Alert objects.
    """
    cleaned = []
    for alert in alerts:
        cleaned.append(clean_alert(alert))
    return cleaned

def save_silver(df: pd.DataFrame, path: str = "data/silver/alerts_clean.csv") -> None:
    """
    Save cleaned alerts DataFrame to Silver layer.

    Args:
        df: The cleaned DataFrame to save.
        path: Path to save the CSV file.

    Returns:
        None
    """
    df.to_csv(path, index=False)
    print(f"✅ Saved to {path}")
    

if __name__ == "__main__":
    # CSV pipeline
    df = load_bronze()
    df_clean = clean_bronze(df)
    save_silver(df_clean)

    # JSON pipeline
    alerts = load_bronze_json()
    alerts_clean = clean_bronze_json(alerts)
    for alert in alerts_clean:
        print(alert)