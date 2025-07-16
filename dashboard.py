import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st
from shapely.geometry import Point
import numpy as np
import matplotlib.ticker as ticker


@st.cache_data
def load_roads():
    return gpd.read_file("data/tl_2019_41_prisecroads").to_crs("EPSG:4326")


@st.cache_data
def load_counties():
    gdf = gpd.read_file("data/BLM_OR_County_Boundaries_Polygon_Hub_-3504410327477223647")
    gdf = gdf.to_crs("EPSG:4326")

    gdf["COUNTY_NAM"] = gdf["COUNTY_NAM"].str.upper()

    mask = (gdf["ORIG_FID"] > 41) | (
        (gdf["ORIG_FID"] > 37) & (gdf["COUNTY_NAM"].isin(["CLATSOP", "COLUMBIA"]))
    )

    return gdf[mask]


@st.cache_data
def load_fire_data():
    df = pd.read_csv("data/ODF_Fire_3681693300358663469.csv", dtype={"Longitude": str, "Latitude": str}, low_memory=False)
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df = df.dropna(subset=["Longitude", "Latitude"])
    return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["Longitude"], df["Latitude"])], crs="EPSG:4326")


@st.cache_data
def load_landslide_data():
    gdf = gpd.read_file("data/SLIDO_Release_4p5_wMetadata.gdb", layer="Historic_Landslide_Points")
    return gdf.to_crs("EPSG:4326").dropna(subset=["geometry"])


@st.cache_data
def load_flood_data():
    gdb_path = "data/Oregon_Statewide_Flood_Hazards.gdb"  
    layer_name = "FEMA_BFE"  
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    return gdf.to_crs("EPSG:4326").dropna(subset=["geometry"])


def plot_bubble_map(fire_gdf, landslide_gdf, show_fire=True, show_landslide=True):
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 1))

    combined_bounds = None

    if show_fire:
        fire_bubble_gdf = fire_gdf.dropna(subset=["FinalFireSizeAcres"])
        fire_bubble_gdf = fire_bubble_gdf[fire_bubble_gdf["FinalFireSizeAcres"] > 10]
        fire_bubble_gdf["FinalFireSizeAcres"] = fire_bubble_gdf["FinalFireSizeAcres"].clip(upper=5000)
        fire_sizes = (np.log1p(fire_bubble_gdf["FinalFireSizeAcres"])) ** 2 * 2.5
        fire_bubble_gdf.plot(ax=ax, markersize=fire_sizes, color='firebrick', alpha=0.6, label="Fires")
        combined_bounds = fire_bubble_gdf.total_bounds

    if show_landslide:
        landslide_bubble_gdf = landslide_gdf.dropna(subset=["VOLUME_ft3"])
        landslide_bubble_gdf = landslide_bubble_gdf[landslide_bubble_gdf["VOLUME_ft3"] > 1000]
        landslide_bubble_gdf["VOLUME_ft3"] = landslide_bubble_gdf["VOLUME_ft3"].clip(upper=1_000_000)
        landslide_sizes = (np.log1p(landslide_bubble_gdf["VOLUME_ft3"])) ** 2 * 0.8
        landslide_bubble_gdf.plot(ax=ax, markersize=landslide_sizes, color='saddlebrown', alpha=0.6, label="Landslides")
        bounds = landslide_bubble_gdf.total_bounds
        if combined_bounds is not None:
            combined_bounds = [
                min(combined_bounds[0], bounds[0]),
                min(combined_bounds[1], bounds[1]),
                max(combined_bounds[2], bounds[2]),
                max(combined_bounds[3], bounds[3]),
            ]
        else:
            combined_bounds = bounds


    ax.set_title("Bubble Map of Oregon Fires and Landslides", color='white')
    ax.set_axis_off()
    plt.legend(loc='upper right')
    plt.tight_layout()
    return fig


def plot_dot_map(fire_gdf, landslide_gdf, flood_gdf=None, roads_gdf=None, counties_gdf=None,
                 show_fires=True, show_landslides=True, show_floods=True,
                 show_roads=True, show_counties=True):
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 1))

    if show_counties and counties_gdf is not None:
        counties_gdf.boundary.plot(ax=ax, edgecolor="white", linewidth=0.8, alpha=0.6)
    if show_roads and roads_gdf is not None:
        roads_gdf.plot(ax=ax, color="gray", linewidth=0.4, alpha=0.6)

    if show_landslides:
        landslide_gdf.plot(ax=ax, color="saddlebrown", markersize=1, alpha=0.5, label="Landslides")
    if show_fires:
        fire_gdf.plot(ax=ax, color="firebrick", markersize=0.5, alpha=0.6, label="Fires")
    if show_floods and flood_gdf is not None and not flood_gdf.empty:
        flood_gdf.plot(ax=ax, color="royalblue", markersize=0.5, alpha=0.6, label="Flood Zones")


    ax.set_title("Oregon Fires, Landslides, and Floods", color='white')
    ax.set_axis_off()
    plt.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_fires_by_year(fire_df):
    fire_df["PreparedDate"] = pd.to_datetime(fire_df["PreparedDate"], errors="coerce")
    fire_df["Year"] = fire_df["PreparedDate"].dt.year
    counts = fire_df["Year"].value_counts().sort_index()

    counts = counts[counts.index >= 1970]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(counts.index, counts.values, marker='o', color='darkred')
    ax.set_title("Number of Fires per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fires")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_fire_by_county(data):
    county_number_to_name = {
        1: "Baker", 2: "Benton", 3: "Clackamas", 4: "Clatsop", 5: "Columbia",
        6: "Coos", 7: "Crook", 8: "Curry", 9: "Deschutes", 10: "Douglas",
        11: "Gilliam", 12: "Grant", 13: "Harney", 14: "Hood River", 15: "Jackson",
        16: "Jefferson", 17: "Josephine", 18: "Klamath", 19: "Lake", 20: "Lane",
        21: "Lincoln", 22: "Linn", 23: "Malheur", 24: "Marion", 25: "Morrow",
        26: "Multnomah", 27: "Polk", 28: "Sherman", 29: "Tillamook", 30: "Umatilla",
        31: "Union", 32: "Wallowa", 33: "Wasco", 34: "Washington", 35: "Wheeler", 36: "Yamhill"
    }

    data = data.copy()
    data["County"] = pd.to_numeric(data["County"], errors="coerce")
    data["CountyName"] = data["County"].map(county_number_to_name)
    county_counts = data['CountyName'].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    ax = county_counts.plot(kind='bar', color='darkred')
    plt.title('Top 10 Counties Most Affected by Fires', fontsize=16)
    plt.xlabel('County', fontsize=15)
    plt.ylabel('Number of Fires', fontsize=15)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()


def plot_least_fire_by_county(data):
    county_number_to_name = {
        1: "Baker", 2: "Benton", 3: "Clackamas", 4: "Clatsop", 5: "Columbia",
        6: "Coos", 7: "Crook", 8: "Curry", 9: "Deschutes", 10: "Douglas",
        11: "Gilliam", 12: "Grant", 13: "Harney", 14: "Hood River", 15: "Jackson",
        16: "Jefferson", 17: "Josephine", 18: "Klamath", 19: "Lake", 20: "Lane",
        21: "Lincoln", 22: "Linn", 23: "Malheur", 24: "Marion", 25: "Morrow",
        26: "Multnomah", 27: "Polk", 28: "Sherman", 29: "Tillamook", 30: "Umatilla",
        31: "Union", 32: "Wallowa", 33: "Wasco", 34: "Washington", 35: "Wheeler", 36: "Yamhill"
    }

    data = data.copy()
    data["County"] = pd.to_numeric(data["County"], errors="coerce")
    data["CountyName"] = data["County"].map(county_number_to_name)
    county_counts = data['CountyName'].value_counts().sort_values().head(10)

    plt.figure(figsize=(10, 6))
    ax = county_counts.plot(kind='bar', color='darkred')
    plt.title('Top 10 Counties Least Affected by Fires', fontsize=16)
    plt.xlabel('County', fontsize=15)
    plt.ylabel('Number of Fires', fontsize=15)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', integer=True))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()


def plot_landslides_by_year(landslide_df):
    landslide_df['YEAR'] = pd.to_numeric(landslide_df['YEAR'], errors='coerce')
    landslide_df = landslide_df[(landslide_df['YEAR'] >= 1965) & (landslide_df['YEAR'] != 1996)]
    counts = landslide_df['YEAR'].value_counts().sort_index()
    year_range = pd.Series(index=range(1965, int(landslide_df['YEAR'].max()) + 1), dtype=int).fillna(0)
    counts = year_range.add(counts, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(counts.index, counts.values, marker='o', color='brown')
    ax.set_title("Number of Landslides per Year (excluding 1996)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Landslides")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_floods_by_year(flood_df):
    flood_df = flood_df.copy()
    flood_df["EFF_DATE"] = pd.to_datetime(flood_df["EFF_DATE"], errors="coerce")
    flood_df["Year"] = flood_df["EFF_DATE"].dt.year

    counts = flood_df["Year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(counts.index, counts.values, marker='o', color='blue')
    ax.set_title("Number of Flood Features per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_flood_elevation_distribution(flood_df):
    flood_df = flood_df.copy()
    flood_df = flood_df[pd.to_numeric(flood_df["ELEV"], errors="coerce").notnull()]
    flood_df["ELEV"] = flood_df["ELEV"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(flood_df["ELEV"], bins=30, color='skyblue', edgecolor='black')
    ax.set_title("Base Flood Elevation Distribution")
    ax.set_xlabel("Elevation (ft)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


fire_uncertainty = {
    "HARNEY": "high",
    "MALHEUR": "high",
    "GILLIAM": "high",
    "MORROW": "medium",
    "JOSEPHINE": "high",
    "JACKSON": "high",
    "CURRY": "high",
    "COOS": "high",
    "DOUGLAS": "high",
    "LANE": "high",
    "LINN": "high",
    "BENTON": "high",
    "LINCOLN": "high",
    "CLATSOP": "high",
    "TILLAMOOK": "high",
    "CLACKAMAS": "high",
    "COLUMBIA": "high",

    "WHEELER": "medium",
    "BAKER": "medium",
    "LAKE": "medium",
    "JEFFERSON": "medium",
    "WASCO": "medium",
    "UNION": "medium",
    "MARION": "medium",
    "YAMHILL": "medium",
    "HOOD RIVER": "medium",
    "POLK": "medium",
    "KLAMATH": "medium",
    "WASHINGTON": "medium",

    "DESCHUTES": "low",
    "CROOK": "low",
    "GRANT": "low",
    "WALLOWA": "low",
    "SHERMAN": "high",
    "UMATILLA": "low",
    "MULTNOMAH": "low"
}

landslide_uncertainty = {
    "BAKER": "high",
    "BENTON": "high",
    "CLACKAMAS": "high",
    "CLATSOP": "medium",
    "COLUMBIA": "medium",
    "COOS": "medium",
    "CROOK": "high",
    "CURRY": "medium",
    "DESCHUTES": "high",
    "DOUGLAS": "high",
    "GILLIAM": "high",
    "GRANT": "high",
    "HARNEY": "medium",
    "HOOD RIVER": "medium",
    "JACKSON": "high",
    "JEFFERSON": "high",
    "JOSEPHINE": "high",
    "KLAMATH": "high",
    "LAKE": "high",
    "LANE": "high",
    "LINCOLN": "high",
    "LINN": "high",
    "MALHEUR": "high",
    "MARION": "high",
    "MORROW": "medium",
    "MULTNOMAH": "high",
    "POLK": "high",
    "SHERMAN": "high",
    "TILLAMOOK": "high",
    "UMATILLA": "medium",
    "UNION": "high",
    "WALLOWA": "medium",
    "WASCO": "high",
    "WASHINGTON": "high",
    "WHEELER": "high",
    "YAMHILL": "high"
}


flood_uncertainty = {
    "HARNEY": "high",
    "MALHEUR": "high",
    "GRANT": "high",
    "BAKER": "high",
    "LAKE": "high",
    "BENTON": "high",
    "YAMHILL": "high",
    "POLK": "high",
    "HOOD RIVER": "high",
    "MULTNOMAH": "high",
    "KLAMATH": "high",


    "CLACKAMAS": "medium",
    "MARION": "medium",
    "SHERMAN": "medium",
    "GILLIAM": "medium",
    "WASCO": "medium",
    "CLATSOP": "medium",
    "COLUMBIA": "medium",
    "WALLOWA": "medium",

    "CROOK": "high",
    "DESCHUTES": "high",
    "JEFFERSON": "high",
    "WHEELER": "high",


    "LINN": "low",
    "LANE": "low",
    "DOUGLAS": "low",
    "JOSEPHINE": "low",
    "JACKSON": "low",
    "LINCOLN": "low",
    "TILLAMOOK": "low",
    "COOS": "low",
    "CURRY": "low",
    "UMATILLA": "low",
    "UNION": "low",
    "MORROW": "low",
    "WASHINGTON": "low"

}


def classify(county_name, mapping):
    return mapping.get(county_name.upper(), "no data")


def plot_uncertainty_map_layers(counties_gdf, roads_gdf=None,
                                show_roads=True,
                                show_counties=True,
                                show_fire=True,
                                show_landslide=True,
                                show_flood=False): 
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 1))

    color_map = {
        "high": "#006837",
        "medium": "#fee08b",
        "low": "#d73027",
    }

    if show_fire:
        counties_gdf.plot(
            ax=ax,
            color=counties_gdf["Fire_Uncertainty"].map(color_map),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.6,
            label="Fire Uncertainty"
        )

    if show_landslide:
        counties_gdf.plot(
            ax=ax,
            color=counties_gdf["Landslide_Uncertainty"].map(color_map),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.6,
            label="Landslide Uncertainty"
        )

    if show_flood:
        counties_gdf.plot(
            ax=ax,
            color=counties_gdf["Flood_Uncertainty"].map(color_map),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.6,
            label="Flood Uncertainty"
        )

    if show_roads and roads_gdf is not None:
        roads_gdf.plot(ax=ax, color="gray", linewidth=0.4, alpha=0.6)

    if show_counties:
        counties_gdf.boundary.plot(ax=ax, edgecolor="white", linewidth=1, alpha=0.6)


    for level, color in color_map.items():
        ax.scatter([], [], color=color, label=f"{level.capitalize()} Confidence")

    ax.set_title("Uncertainty Map by County", color='white')
    ax.legend(loc="lower left")
    ax.set_axis_off()
    plt.tight_layout()
    return fig



tab1, tab2, tab3, tab4, tab5 = st.tabs(["Maps", "Uncertainty Map", "Fire Data", "Landslide Data", "Flood Data"])


fire_gdf = load_fire_data()
landslide_gdf = load_landslide_data()
flood_gdf = load_flood_data()

with tab1:

    fire_gdf["PreparedDate"] = pd.to_datetime(
        fire_gdf["PreparedDate"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )

    fire_years = fire_gdf["PreparedDate"].dt.year.dropna().astype(int)


    st.header("Oregon Fire, Landslide, & Flood Map")
    st.markdown("""
    My safe spot is **Harney** and **Malheur** counties.
    """)

    # --- Map view toggle ---
    col1, col2 = st.columns([0.05, 1])
    with col1:
        view_toggle = st.toggle("", label_visibility="collapsed", key="map_toggle")
    with col2:
        st.markdown(f"#### {'Bubble Map, click to toggle to Dot Map' if view_toggle else 'Dot Map, click to toggle to Bubble Map'}")


    fire_gdf["Year"] = fire_gdf["PreparedDate"].dt.year
    landslide_years = pd.to_numeric(landslide_gdf["YEAR"], errors="coerce").dropna().astype(int)
    flood_years = pd.to_datetime(flood_gdf["EFF_DATE"], errors="coerce").dt.year.dropna().astype(int)

    combined_years = sorted(set(fire_years.tolist() + landslide_years.tolist() + flood_years.tolist()))

    filtered_years = [year for year in combined_years if year >= 1960]

    min_year = min(filtered_years)
    max_year = max(filtered_years)
    selected_range = st.slider("Select a year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    start_year, end_year = selected_range


    fire_gdf_year = fire_gdf[
        (fire_gdf["Year"] >= start_year) & (fire_gdf["Year"] <= end_year)
    ]



    landslide_gdf["YEAR"] = pd.to_numeric(landslide_gdf["YEAR"], errors="coerce")
    landslide_gdf_year = landslide_gdf[(landslide_gdf["YEAR"] >= start_year) & (landslide_gdf["YEAR"] <= end_year)]


    flood_gdf["EFF_DATE"] = pd.to_datetime(flood_gdf["EFF_DATE"], errors="coerce")
    flood_gdf["Year"] = flood_gdf["EFF_DATE"].dt.year
    flood_gdf_year = flood_gdf[(flood_gdf["Year"] >= start_year) & (flood_gdf["Year"] <= end_year)]


    if view_toggle:
        show_fire_bubbles = st.checkbox("Fires", value=True)
        show_landslide_bubbles = st.checkbox("Landslides", value=True)
        st.pyplot(plot_bubble_map(fire_gdf_year, landslide_gdf_year, show_fire_bubbles, show_landslide_bubbles))
    else:
        show_roads = st.checkbox("Roads", value=True)
        show_counties = st.checkbox("Counties", value=True)
        show_fires = st.checkbox("Fires", value=True)
        show_landslides = st.checkbox("Landslides", value=True)
        show_floods = st.checkbox("Flood Zones", value=True)

        roads_gdf = load_roads() if show_roads else None
        counties_gdf = load_counties() if show_counties else None
        flood_data = flood_gdf_year if show_floods else None

        st.pyplot(plot_dot_map(
            fire_gdf_year,
            landslide_gdf_year,
            flood_gdf=flood_data,
            roads_gdf=roads_gdf,
            counties_gdf=counties_gdf,
            show_fires=show_fires,
            show_landslides=show_landslides,
            show_floods=show_floods,
            show_roads=show_roads,
            show_counties=show_counties
        ))


with tab2:

    st.header("Uncertainty Map")
    st.markdown("""
    My safe spot is **Harney** and **Malheur** counties.\n
    This map visualizes **uncertainty levels** for three factors:  
    - **Fire Uncertainty**: How certain we are that sparse trees results in fewer fires.  
    - **Landslide Uncertainty**: How certain we are that low-slope terrain results in fewer landslides.  
    - **Flood Uncertainty**: How certain we are that high elevation results in fewer floods.
    """)


    show_roads = True
    show_counties = True

    show_fire = st.checkbox("Fire Uncertainty", value=False)
    show_landslide = st.checkbox("Landslide Uncertainty", value=False)
    show_flood = st.checkbox("Flood Uncertainty", value=False)

    roads_gdf = load_roads() if show_roads else None
    counties_gdf = load_counties()
    counties_gdf["Fire_Uncertainty"] = counties_gdf["COUNTY_NAM"].apply(lambda x: classify(x, fire_uncertainty))
    counties_gdf["Landslide_Uncertainty"] = counties_gdf["COUNTY_NAM"].apply(lambda x: classify(x, landslide_uncertainty))
    counties_gdf["Flood_Uncertainty"] = counties_gdf["COUNTY_NAM"].apply(lambda x: classify(x, flood_uncertainty))

    st.pyplot(plot_uncertainty_map_layers(
        counties_gdf,
        roads_gdf=roads_gdf,
        show_roads=show_roads,
        show_counties=show_counties,
        show_fire=show_fire,
        show_landslide=show_landslide,
        show_flood=show_flood
    ))

    st.subheader("Causal Factor Network")
    st.image("photo/reasonnet.png")


with tab3:

    st.header("Fire Data Visualizations")

    st.subheader("Top 10 Counties Most Affected by Fires")
    plot_fire_by_county(fire_gdf)

    st.subheader("Top 10 Counties Least Affected by Fires")
    plot_least_fire_by_county(fire_gdf)

    st.subheader("Fires Per Year")
    st.pyplot(plot_fires_by_year(fire_gdf))

with tab4:

    st.header("Landslide Data Visualizations")
    st.subheader("Landslides Per Year")
    st.pyplot(plot_landslides_by_year(landslide_gdf))


with tab5:
    st.header("Flood Data Visualizations")

    st.subheader("Flood Elevation Distribution")
    st.pyplot(plot_flood_elevation_distribution(flood_gdf))
    st.subheader("Floods per Year")
    st.pyplot(plot_floods_by_year(flood_gdf))


