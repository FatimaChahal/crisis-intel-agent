import pandas as pd


def compute_risk_score(burnt_area_ha: float, nr_fires: int,
                       avg_fire_size_ha: float) -> float:
    """
    Compute a normalized risk score between 0.0 and 1.0.

    Args:
        burnt_area_ha: Total burnt area in hectares.
        nr_fires: Number of fires recorded.
        avg_fire_size_ha: Average fire size in hectares.

    Returns:
        Normalized risk score between 0.0 and 1.0.
    """
    max_area = 500000
    max_fires = 20000
    max_avg = 500

    score_area = min(burnt_area_ha / max_area, 1.0)
    score_fires = min(nr_fires / max_fires, 1.0)
    score_avg = min(avg_fire_size_ha / max_avg, 1.0)

    return round(0.6 * score_area + 0.3 * score_fires + 0.1 * score_avg, 3)


def build_summary(row: pd.Series) -> str:
    """
    Build a human-readable summary for RAG indexing.

    Args:
        row: A Silver DataFrame row.

    Returns:
        A natural language summary of the fire event.
    """
    return (
        f"In {row['year']}, {row['country']} recorded {int(row['nr_fires'])} forest fires "
        f"burning {int(row['burnt_area_ha'])} hectares. "
        f"Severity: {row['severity']}. "
        f"Average fire size: {row['avg_fire_size_ha']} ha. "
        f"Risk score: {row['risk_score']}. "
        f"This was a {row['severity'].lower()} severity wildfire event "
        f"in {row['decade']}s Europe."
    )


def build_gold() -> pd.DataFrame:
    """
    Enrich Silver data with risk score, summary and crisis type for Gold layer.

    Returns:
        Enriched DataFrame ready for RAG indexing.
    """
    df = pd.read_csv("data/silver/fires_clean.csv")

    # Compute risk score
    df["risk_score"] = df.apply(
        lambda r: compute_risk_score(
            r["burnt_area_ha"], r["nr_fires"], r["avg_fire_size_ha"]
        ), axis=1
    )

    # Add crisis type
    df["crisis_type"] = "wildfire"

    # Add region
    southern = ["PRT", "ESP", "FRA", "ITA", "GRC", "DZA", "MAR", "TUR", "CYP"]
    df["region"] = df["country_code"].apply(
        lambda c: "Southern Europe" if c in southern else "Northern/Eastern Europe"
    )

    # Build text summary for RAG
    df["summary"] = df.apply(build_summary, axis=1)

    # Keep only meaningful rows
    df_gold = df[df["burnt_area_ha"] > 100].copy()
    df_gold = df_gold.sort_values("risk_score", ascending=False)
    df_gold = df_gold.reset_index(drop=True)

    df_gold.to_csv("data/gold/fires_gold.csv", index=False)

    print(f"✅ Gold: {len(df_gold)} rows")
    print(f"✅ Risk score range: {df_gold['risk_score'].min()} → {df_gold['risk_score'].max()}")
    print(f"\n📄 Example summary:")
    print(df_gold.iloc[0]["summary"])
    return df_gold


if __name__ == "__main__":
    build_gold()