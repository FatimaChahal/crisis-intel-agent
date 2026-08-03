import pandas as pd


def compute_risk_score(
    burnt_area_ha: float, duration_days: int, protected_area_pct: float
) -> float:
    """
    Compute normalized risk score between 0.0 and 1.0.

    Args:
        burnt_area_ha: Total burnt area in hectares.
        duration_days: Fire duration in days.
        protected_area_pct: Percentage of protected area affected.

    Returns:
        Normalized risk score between 0.0 and 1.0.
    """
    score_area = min(burnt_area_ha / 100000, 1.0)
    score_duration = min(duration_days / 30, 1.0)
    score_protected = min(protected_area_pct / 100, 1.0)
    return round(0.6 * score_area + 0.3 * score_duration + 0.1 * score_protected, 3)


def build_summary_modis(row: pd.Series) -> str:
    """
    Build human-readable summary for RAG indexing.

    Args:
        row: A Silver MODIS DataFrame row.

    Returns:
        Natural language summary of the fire event.
    """
    province = row["province"]
    try:
        province.encode("utf-8")
        if "?" in province or province == "nan":
            province = "unknown province"
    except Exception:
        province = "unknown province"

    commune = row["commune"]
    try:
        commune.encode("utf-8")
        if "?" in commune or commune == "nan":
            commune = ""
    except Exception:
        commune = ""
    commune = row["commune"] if str(row["commune"]) not in ["?????", "nan"] else ""
    location = f"{province}, {commune}".strip(", ") if commune else province

    return (
        f"Wildfire in {row['country_code']} ({location}) "
        f"in {row['season']} {int(row['year'])}. "
        f"Burnt area: {int(row['burnt_area_ha'])} hectares. "
        f"Duration: {int(row['duration_days'])} days. "
        f"Severity: {row['severity']}. "
        f"Dominant vegetation: {row['dominant_vegetation']}. "
        f"Protected area affected: {round(float(row['protected_area_pct']), 1)}%. "
        f"Risk score: {row['risk_score']}."
    )


def build_gold_modis() -> pd.DataFrame:
    """
    Enrich MODIS Silver data for Gold layer — RAG ready.

    Returns:
        Enriched DataFrame with summaries and risk scores.
    """
    df = pd.read_csv("data/silver/modis_fires_clean.csv")
    # Fix Greek character encoding issues
    for col in ["province", "commune"]:
        df[col] = df[col].apply(
            lambda x: (
                "unknown"
                if any(c == "?" for c in str(x)) or str(x) == "nan"
                else str(x)
            )
        )

    df["protected_area_pct"] = pd.to_numeric(
        df["protected_area_pct"], errors="coerce"
    ).fillna(0)

    df["risk_score"] = df.apply(
        lambda r: compute_risk_score(
            r["burnt_area_ha"], r["duration_days"], r["protected_area_pct"]
        ),
        axis=1,
    )

    df["crisis_type"] = "wildfire"

    # Clean province/commune encoding issues
    df["province"] = df["province"].apply(
        lambda x: "unknown" if str(x) in ["?????", "nan"] else str(x)
    )
    df["commune"] = df["commune"].apply(
        lambda x: "unknown" if str(x) in ["?????", "nan"] else str(x)
    )

    # Drop rows with missing year or season before building summary
    df = df.dropna(subset=["year", "season", "burnt_area_ha"])
    df["year"] = df["year"].astype(int)

    df["summary"] = df.apply(build_summary_modis, axis=1)

    # Keep significant fires only
    df_gold = df[df["burnt_area_ha"] >= 100].copy()
    df_gold = df_gold.sort_values("risk_score", ascending=False)
    df_gold = df_gold.reset_index(drop=True)

    df_gold.to_csv("data/gold/modis_fires_gold.csv", index=False)

    print(f"✅ MODIS Gold: {len(df_gold)} fires")
    print(
        f"✅ Risk score: {df_gold['risk_score'].min()} → {df_gold['risk_score'].max()}"
    )
    print(f"\n📄 Example summary (top fire):")
    print(df_gold.iloc[0]["summary"])
    print(f"\n📄 Example summary (fire #5):")
    print(df_gold.iloc[4]["summary"])
    return df_gold


if __name__ == "__main__":
    build_gold_modis()
