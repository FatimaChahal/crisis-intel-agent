import pandas as pd
import numpy as np


COUNTRY_NAMES = {
    "PRT": "Portugal", "ESP": "Spain", "FRA": "France",
    "ITA": "Italy", "GRC": "Greece", "DZA": "Algeria",
    "AUT": "Austria", "BGR": "Bulgaria", "HRV": "Croatia",
    "CYP": "Cyprus", "CZE": "Czech Republic", "EST": "Estonia",
    "FIN": "Finland", "DEU": "Germany", "HUN": "Hungary",
    "LVA": "Latvia", "LBN": "Lebanon", "LTU": "Lithuania",
    "MAR": "Morocco", "NLD": "Netherlands", "MKD": "North Macedonia",
    "NOR": "Norway", "POL": "Poland", "ROU": "Romania",
    "SRB": "Serbia", "SVK": "Slovakia", "SVN": "Slovenia",
    "SWE": "Sweden", "CHE": "Switzerland", "TUR": "Turkey",
    "UKR": "Ukraine"
}


def clean_value(val) -> float:
    """
    Clean a raw value from Excel (remove spaces, convert to float).

    Args:
        val: Raw value from Excel cell.

    Returns:
        Cleaned float value or 0.0 if invalid.
    """
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        val = val.replace(" ", "").replace(",", ".")
        try:
            return float(val)
        except ValueError:
            return 0.0
    return float(val)


def compute_severity(burnt_area_ha: float, nr_fires: int) -> str:
    """
    Compute severity level based on burnt area and number of fires.

    Args:
        burnt_area_ha: Total burnt area in hectares.
        nr_fires: Number of fires recorded.

    Returns:
        Severity level string: LOW, MEDIUM, HIGH, or EXTREME.
    """
    if burnt_area_ha >= 100000:
        return "EXTREME"
    elif burnt_area_ha >= 20000:
        return "HIGH"
    elif burnt_area_ha >= 5000:
        return "MEDIUM"
    else:
        return "LOW"


def build_silver() -> pd.DataFrame:
    """
    Transform raw Bronze CSVs into a clean Silver DataFrame.

    Returns:
        Clean DataFrame with one row per country per year.
    """
    df_area = pd.read_csv("data/bronze/burnt_area_raw.csv")
    df_fires = pd.read_csv("data/bronze/nr_fires_raw.csv")

    countries = [c for c in df_area.columns if c != "Year"]
    records = []

    for _, row_area in df_area.iterrows():
        year = int(row_area["Year"])
        row_fires = df_fires[df_fires["Year"] == year]

        for code in countries:
            burnt_area = clean_value(row_area[code])
            nr_fires = 0
            if not row_fires.empty:
                nr_fires = int(clean_value(row_fires.iloc[0][code]))

            # Skip rows with no data
            if burnt_area == 0 and nr_fires == 0:
                continue

            country_name = COUNTRY_NAMES.get(code, code)
            severity = compute_severity(burnt_area, nr_fires)

            records.append({
                "year": year,
                "country_code": code,
                "country": country_name,
                "burnt_area_ha": burnt_area,
                "nr_fires": nr_fires,
                "severity": severity,
                "avg_fire_size_ha": round(burnt_area / nr_fires, 2) if nr_fires > 0 else 0,
                "decade": (year // 10) * 10,
            })

    df_silver = pd.DataFrame(records)
    df_silver = df_silver.sort_values(["year", "burnt_area_ha"], ascending=[True, False])
    df_silver = df_silver.reset_index(drop=True)

    df_silver.to_csv("data/silver/fires_clean.csv", index=False)
    print(f"✅ Silver: {len(df_silver)} rows, {df_silver['country'].nunique()} countries")
    print(f"✅ Years: {df_silver['year'].min()} → {df_silver['year'].max()}")
    print(f"✅ Severity distribution:\n{df_silver['severity'].value_counts()}")
    print(df_silver.head(10))
    return df_silver


if __name__ == "__main__":
    build_silver()
    