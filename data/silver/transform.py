import pandas as pd


def load_bronze(path: str = "data/bronze/alerts.csv") -> pd.DataFrame:
    """
    Load raw alerts data from Bronze layer.

    Args:
        path: Path to the CSV file.

    Returns:
        A raw DataFrame with alerts.
    """
    return pd.read_csv(path)


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


# if __name__ == "__main__":
#     df = load_bronze()
#     df_clean = clean_bronze(df)
#     print(df_clean)

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
    df = load_bronze()
    df_clean = clean_bronze(df)
    print(df_clean)
    save_silver(df_clean)