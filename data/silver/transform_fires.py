import pandas as pd

COUNTRY_NAMES = {
    "PRT": "Portugal",
    "ESP": "Spain",
    "FRA": "France",
    "ITA": "Italy",
    "GRC": "Greece",
    "DZA": "Algeria",
    "AUT": "Austria",
    "BGR": "Bulgaria",
    "HRV": "Croatia",
    "CYP": "Cyprus",
    "CZE": "Czech Republic",
    "EST": "Estonia",
    "FIN": "Finland",
    "DEU": "Germany",
    "HUN": "Hungary",
    "LVA": "Latvia",
    "LBN": "Lebanon",
    "LTU": "Lithuania",
    "MAR": "Morocco",
    "NLD": "Netherlands",
    "MKD": "North Macedonia",
    "NOR": "Norway",
    "POL": "Poland",
    "ROU": "Romania",
    "SRB": "Serbia",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "SWE": "Sweden",
    "CHE": "Switzerland",
    "TUR": "Turkey",
    "UKR": "Ukraine",
}


def clean_value(val) -> float:
    """Clean a raw value from Excel."""
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
    """Compute severity level based on burnt area and number of fires."""
    if burnt_area_ha >= 100000:
        return "EXTREME"
    elif burnt_area_ha >= 20000:
        return "HIGH"
    elif burnt_area_ha >= 5000:
        return "MEDIUM"
    else:
        return "LOW"


def build_silver() -> pd.DataFrame:
    """Transform raw EFFIS Bronze CSVs into a clean Silver DataFrame."""
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

            if burnt_area == 0 and nr_fires == 0:
                continue

            country_name = COUNTRY_NAMES.get(code, code)
            severity = compute_severity(burnt_area, nr_fires)

            records.append(
                {
                    "year": year,
                    "country_code": code,
                    "country": country_name,
                    "burnt_area_ha": burnt_area,
                    "nr_fires": nr_fires,
                    "severity": severity,
                    "avg_fire_size_ha": (
                        round(burnt_area / nr_fires, 2) if nr_fires > 0 else 0
                    ),
                    "decade": (year // 10) * 10,
                }
            )

    df_silver = pd.DataFrame(records)
    df_silver = df_silver.sort_values(
        ["year", "burnt_area_ha"], ascending=[True, False]
    )
    df_silver = df_silver.reset_index(drop=True)
    df_silver.to_csv("data/silver/fires_clean.csv", index=False)

    print(f"✅ EFFIS Silver: {len(df_silver)} rows")
    return df_silver


def get_season(month: int) -> str:
    """Get season from month number."""
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    return "autumn"


def build_silver_modis() -> pd.DataFrame:
    """Transform raw MODIS data into clean Silver DataFrame."""
    df = pd.read_csv("data/bronze/modis_fires_raw.csv")

    df["FIREDATE"] = pd.to_datetime(df["FIREDATE"], errors="coerce")
    df["FINALDATE"] = pd.to_datetime(df["FINALDATE"], errors="coerce")
    df["duration_days"] = (df["FINALDATE"] - df["FIREDATE"]).dt.days
    df["duration_days"] = df["duration_days"].fillna(1).clip(lower=1).astype(int)
    df["AREA_HA"] = pd.to_numeric(df["AREA_HA"], errors="coerce").fillna(0)
    df = df[df["AREA_HA"] > 10].copy()
    df["year"] = df["FIREDATE"].dt.year
    df["month"] = df["FIREDATE"].dt.month
    df["season"] = df["month"].apply(get_season)

    veg_cols = ["BROADLEA", "CONIFER", "MIXED", "SCLEROPH", "TRANSIT"]
    for col in veg_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["dominant_vegetation"] = df[veg_cols].idxmax(axis=1)

    df["severity"] = df["AREA_HA"].apply(
        lambda x: (
            "EXTREME"
            if x >= 100000
            else "HIGH" if x >= 20000 else "MEDIUM" if x >= 5000 else "LOW"
        )
    )

    df_silver = df[
        [
            "COUNTRY",
            "PROVINCE",
            "COMMUNE",
            "year",
            "month",
            "season",
            "AREA_HA",
            "duration_days",
            "severity",
            "dominant_vegetation",
            "PERCNA2K",
        ]
    ].rename(
        columns={
            "COUNTRY": "country_code",
            "PROVINCE": "province",
            "COMMUNE": "commune",
            "AREA_HA": "burnt_area_ha",
            "PERCNA2K": "protected_area_pct",
        }
    )

    df_silver = df_silver.sort_values("burnt_area_ha", ascending=False)
    df_silver = df_silver.reset_index(drop=True)
    df_silver.to_csv("data/silver/modis_fires_clean.csv", index=False)

    print(f"✅ MODIS Silver: {len(df_silver)} fires")
    print(f"✅ Severity:\n{df_silver['severity'].value_counts()}")
    print(
        df_silver[
            ["country_code", "year", "burnt_area_ha", "severity", "province"]
        ].head(5)
    )
    return df_silver


if __name__ == "__main__":
    build_silver()
    build_silver_modis()
