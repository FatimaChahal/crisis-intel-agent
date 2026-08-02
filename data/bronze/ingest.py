import pandas as pd
import geopandas as gpd


def load_bronze_effis() -> pd.DataFrame:
    """
    Load raw EFFIS country totals from Excel file.

    Returns:
        Dictionary with two raw DataFrames: burnt_area and nr_fires.
    """
    path = "data/bronze/effis_report_2024.xlsx"
    df_area = pd.read_excel(path, sheet_name=0)
    df_fires = pd.read_excel(path, sheet_name=1)

    df_area.to_csv("data/bronze/burnt_area_raw.csv", index=False)
    df_fires.to_csv("data/bronze/nr_fires_raw.csv", index=False)

    print(f"✅ EFFIS Bronze: {len(df_area)} years, {len(df_area.columns)-1} countries")
    return {"burnt_area": df_area, "nr_fires": df_fires}


def load_bronze_modis() -> gpd.GeoDataFrame:
    """
    Load raw MODIS burnt area polygons from Shapefile.

    Returns:
        GeoDataFrame with 102561 individual fire records.
    """
    gdf = gpd.read_file("data/bronze/modis.ba.poly.shp")

    # Save as CSV (without geometry for now)
    df = pd.DataFrame(gdf.drop(columns=["geometry"]))
    df.to_csv("data/bronze/modis_fires_raw.csv", index=False)

    print(f"✅ MODIS Bronze: {len(gdf)} fires, {gdf['COUNTRY'].nunique()} countries")
    print(f"✅ Date range: {gdf['FIREDATE'].min()} → {gdf['FIREDATE'].max()}")
    return gdf


if __name__ == "__main__":
    load_bronze_effis()
    load_bronze_modis()