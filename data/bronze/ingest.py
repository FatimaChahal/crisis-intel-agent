import pandas as pd


def load_bronze() -> dict:
    """
    Load raw EFFIS data from Excel file into bronze layer.

    Returns:
        Dictionary with two raw DataFrames: burnt_area and nr_fires.
    """
    path = "data/bronze/effis_report_2024.xlsx"

    df_area = pd.read_excel(path, sheet_name=0)
    df_fires = pd.read_excel(path, sheet_name=1)

    print(f"✅ Bronze loaded: {len(df_area)} years, {len(df_area.columns)-1} countries")
    print(f"✅ Burnt area shape: {df_area.shape}")
    print(f"✅ Nr fires shape: {df_fires.shape}")

    # Save to CSV (Bronze layer)
    df_area.to_csv("data/bronze/burnt_area_raw.csv", index=False)
    df_fires.to_csv("data/bronze/nr_fires_raw.csv", index=False)

    print("✅ Bronze saved to CSV")
    return {"burnt_area": df_area, "nr_fires": df_fires}


if __name__ == "__main__":
    data = load_bronze()
    