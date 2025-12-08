import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st
from shapely.geometry import Point
import numpy as np
import matplotlib.ticker as ticker

st.set_page_config(
    page_title="Oregon Fire, Flood & Landslide Dashboard",
    page_icon="🔥",
)

# Display main title and subtitle
st.title("Oregon Fire, Flood & Landslide Dashboard")
st.markdown("*An interactive visualization of natural disasters in Oregon*")


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


@st.cache_data
def load_roads():
    """
        Load Oregon road data from shapefile.

        Returns:
            GeoDataFrame: Road geometries in EPSG:4326 projection, or None if error
        """
    try:
        return gpd.read_file("data/tl_2019_41_prisecroads").to_crs("EPSG:4326")
    except FileNotFoundError:
        st.error("Road data file not found. Check that data/tl_2019_41_prisecroads exists.")
    except Exception as e:
        st.error(f"Error loading roads data: {e}")


@st.cache_data
def load_counties():
    """
        Load Oregon county boundary data and filter to valid counties.

        Applies filtering to remove duplicate/invalid county entries based on ORIG_FID.

        Returns:
            GeoDataFrame: Filtered county boundaries in EPSG:4326 projection, or None if error
        """
    try:
        gdf = gpd.read_file("data/BLM_OR_County_Boundaries_Polygon_Hub_-3504410327477223647")
        gdf = gdf.to_crs("EPSG:4326")

        gdf["COUNTY_NAM"] = gdf["COUNTY_NAM"].str.upper()

        # Filtering out Washington counties from the dataset
        mask = (gdf["ORIG_FID"] > 41) | (
            (gdf["ORIG_FID"] > 37) & (gdf["COUNTY_NAM"].isin(["CLATSOP", "COLUMBIA"]))
        )

        return gdf[mask]
    except FileNotFoundError:
        st.error("County boundaries data file not found. Please check the data directory.")
        return None
    except Exception as e:
        st.error(f"Error loading county data: {e}")
        return None

@st.cache_data
def load_fire_data():
    """
       Load Oregon fire incident data from CSV and convert to GeoDataFrame.

       Processes longitude/latitude coordinates and creates Point geometries.

       Returns:
           GeoDataFrame: Fire incidents with Point geometries in EPSG:4326, or None if error
       """
    try:
        df = pd.read_csv("data/ODF_Fire_3681693300358663469.csv", dtype={"Longitude": str, "Latitude": str}, low_memory=False)
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df = df.dropna(subset=["Longitude", "Latitude"])
        return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["Longitude"], df["Latitude"])], crs="EPSG:4326")
    except FileNotFoundError:
        st.error("Fire data file not found. Please check that 'data/ODF_Fire_3681693300358663469.csv' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading fire data: {e}")
        return None


@st.cache_data
def load_landslide_data():
    """
        Load Oregon landslide data from geodatabase.

        Returns:
            GeoDataFrame: Historic landslide points in EPSG:4326, or None if error
        """
    try:
        gdf = gpd.read_file("data/SLIDO_Release_4p5_wMetadata.gdb", layer="Historic_Landslide_Points")
        return gdf.to_crs("EPSG:4326").dropna(subset=["geometry"])
    except FileNotFoundError:
        st.error("Landslide data file not found. Please check that 'data/SLIDO_Release_4p5_wMetadata.gdb' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading landslide data: {e}")
        return None


@st.cache_data
def load_flood_data():
    """
        Load Oregon flood hazard data from geodatabase.

        Returns:
            GeoDataFrame: FEMA Base Flood Elevation features in EPSG:4326, or None if error
        """
    try:
        gdb_path = "data/Oregon_Statewide_Flood_Hazards.gdb"
        layer_name = "FEMA_BFE"
        gdf = gpd.read_file(gdb_path, layer=layer_name)
        return gdf.to_crs("EPSG:4326").dropna(subset=["geometry"])
    except FileNotFoundError:
        st.error("Flood data file not found. Please check that 'data/Oregon_Statewide_Flood_Hazards.gdb' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading flood data: {e}")
        return None


# =============================================================================
# MAPPING FUNCTIONS
# =============================================================================

def plot_bubble_map(fire_gdf, landslide_gdf, show_fire=True, show_landslide=True):
    """
       Create bubble map where bubble size represents fire size (acres) or landslide volume (cubic feet).

       Args:
           fire_gdf (GeoDataFrame): Fire incident data
           landslide_gdf (GeoDataFrame): Landslide incident data
           show_fire (bool): Whether to display fire bubbles
           show_landslide (bool): Whether to display landslide bubbles

       Returns:
           Figure: Matplotlib figure object
       """
    try:
        fig, ax = plt.subplots(figsize=(14, 14))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((1, 1, 1, 1))

        combined_bounds = None

        if show_fire:
            # Filter fires with valid size data and minimum 10 acres
            fire_bubble_gdf = fire_gdf.dropna(subset=["FinalFireSizeAcres"])
            fire_bubble_gdf = fire_bubble_gdf[fire_bubble_gdf["FinalFireSizeAcres"] > 10]
            fire_bubble_gdf["FinalFireSizeAcres"] = fire_bubble_gdf["FinalFireSizeAcres"].clip(upper=5000)
            fire_sizes = (np.log1p(fire_bubble_gdf["FinalFireSizeAcres"])) ** 2 * 2.5
            fire_bubble_gdf.plot(ax=ax, markersize=fire_sizes, color='firebrick', alpha=0.6, label="Fires")
            combined_bounds = fire_bubble_gdf.total_bounds

        if show_landslide:
            # Filter landslides with valid volume data and minimum 1000 cubic feet
            landslide_bubble_gdf = landslide_gdf.dropna(subset=["VOLUME_ft3"])
            landslide_bubble_gdf = landslide_bubble_gdf[landslide_bubble_gdf["VOLUME_ft3"] > 1000]
            landslide_bubble_gdf["VOLUME_ft3"] = landslide_bubble_gdf["VOLUME_ft3"].clip(upper=1_000_000)
            landslide_sizes = (np.log1p(landslide_bubble_gdf["VOLUME_ft3"])) ** 2 * 0.8
            landslide_bubble_gdf.plot(ax=ax, markersize=landslide_sizes, color='saddlebrown', alpha=0.6, label="Landslides")
            # Update combined bounds to include both layers
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

    except Exception as e:
            st.error(f"Error creating bubble map: {e}")
            return plt.figure()


def plot_dot_map(fire_gdf, landslide_gdf, flood_gdf=None, roads_gdf=None, counties_gdf=None,
                 show_fires=True, show_landslides=True, show_floods=True,
                 show_roads=True, show_counties=True):
    """
        Create dot map showing individual incident locations with optional base layers.

        Args:
            fire_gdf (GeoDataFrame): Fire incident data
            landslide_gdf (GeoDataFrame): Landslide incident data
            flood_gdf (GeoDataFrame, optional): Flood zone data
            roads_gdf (GeoDataFrame, optional): Road network data
            counties_gdf (GeoDataFrame, optional): County boundary data
            show_fires (bool): Whether to display fire points
            show_landslides (bool): Whether to display landslide points
            show_floods (bool): Whether to display flood zones
            show_roads (bool): Whether to display roads
            show_counties (bool): Whether to display county boundaries

        Returns:
            Figure: Matplotlib figure object
        """
    try:
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

    except Exception as e:
        st.error(f" Error creating dot map: {e}")
        return plt.figure()


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_fires_by_year(fire_df):
    """
       Plot time series of fire incidents per year (1970 onwards).

       Args:
           fire_df (DataFrame): Fire incident data with PreparedDate column

       Returns:
           Figure: Matplotlib figure object
       """
    try:
        fire_df["PreparedDate"] = pd.to_datetime(fire_df["PreparedDate"], errors="coerce")
        fire_df["Year"] = fire_df["PreparedDate"].dt.year
        counts = fire_df["Year"].value_counts().sort_index()

        # Filter to 1970 onwards for cleaner visualization
        counts = counts[counts.index >= 1970]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(counts.index, counts.values, marker='o', color='darkred')
        ax.set_title("Number of Fires per Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Fires")
        ax.grid(True)
        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Error plotting fires by year: {e}")
        return plt.figure()


def plot_fire_by_county(data):
    """
        Plot bar chart of top 10 counties most affected by fires.

        Args:
            data (DataFrame): Fire incident data with County column (numeric codes)
        """
    try:
        if data is None or data.empty:
            st.warning(" No fire data available to plot.")
            return

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

        # Convert county codes to numeric and map to names
        data["County"] = pd.to_numeric(data["County"], errors="coerce")
        data["CountyName"] = data["County"].map(county_number_to_name)
        # Get top 10 counties by fire count
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

    except Exception as e:
        st.error(f"Error plotting fire by county: {e}")


def plot_least_fire_by_county(data):
    """
        Plot bar chart of top 10 counties least affected by fires.

        Args:
            data (DataFrame): Fire incident data with County column (numeric codes)
        """
    try:
        if data is None or data.empty:
            st.warning("No fire data available to plot.")
            return

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

    except Exception as e:
        st.error(f"Error plotting least affected counties: {e}")


def plot_landslides_by_year(landslide_df):
    """
       Plot time series of landslide incidents per year (1965 onwards, excluding 1996).

       Note: 1996 is excluded due to data quality issues in that year.

       Args:
           landslide_df (DataFrame): Landslide incident data with YEAR column

       Returns:
           Figure: Matplotlib figure object
       """
    try:
        landslide_df['YEAR'] = pd.to_numeric(landslide_df['YEAR'], errors='coerce')
        # Filter to 1965 onwards and exclude problematic 1996 data
        landslide_df = landslide_df[(landslide_df['YEAR'] >= 1965) & (landslide_df['YEAR'] != 1996)]
        counts = landslide_df['YEAR'].value_counts().sort_index()
        # Create continuous year range to show years with zero incidents
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

    except Exception as e:
        st.error(f"Error plotting landslides by year: {e}")
        return plt.figure()


def plot_floods_by_year(flood_df):
    """
        Plot time series of flood zone features per year.

        Args:
            flood_df (DataFrame): Flood data with EFF_DATE column

        Returns:
            Figure: Matplotlib figure object
        """
    try:
        if flood_df is None or flood_df.empty:
            st.warning("No flood data available to plot.")
            return plt.figure()

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

    except Exception as e:
        st.error(f"Error plotting floods by year: {e}")
        return plt.figure()


def plot_flood_elevation_distribution(flood_df):
    """
        Plot histogram of base flood elevations.

        Args:
            flood_df (DataFrame): Flood data with ELEV column

        Returns:
            Figure: Matplotlib figure object
        """
    try:
        if flood_df is None or flood_df.empty:
            st.warning("No flood data available to plot.")
            return plt.figure()

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

    except Exception as e:
        st.error(f"Error plotting flood elevation distribution: {e}")
        return plt.figure()


# =============================================================================
# UNCERTAINTY MAPPING
# =============================================================================

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
    """
       Create choropleth map showing uncertainty levels by county.

       Args:
           counties_gdf (GeoDataFrame): County boundaries with uncertainty columns
           roads_gdf (GeoDataFrame, optional): Road network data
           show_roads (bool): Whether to display roads
           show_counties (bool): Whether to display county boundaries
           show_fire (bool): Whether to show fire uncertainty
           show_landslide (bool): Whether to show landslide uncertainty
           show_flood (bool): Whether to show flood uncertainty

       Returns:
           Figure: Matplotlib figure object
       """

    try:
        if counties_gdf is None or counties_gdf.empty:
            st.warning("No county data available to plot uncertainty map.")
            return plt.figure()

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

    except Exception as e:
        st.error(f"Error creating uncertainty map: {e}")
        return plt.figure()


with st.spinner("Loading data..."):
    fire_gdf = load_fire_data()
    landslide_gdf = load_landslide_data()
    flood_gdf = load_flood_data()

# Check if critical data loaded successfully
data_loaded = fire_gdf is not None or landslide_gdf is not None or flood_gdf is not None

if not data_loaded:
    st.error("Failed to load any disaster data. Please check data files and try again.")
    st.stop()


# =============================================================================
# Dashboard Display
# =============================================================================

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Maps", "Uncertainty Map", "Fire Data", "Landslide Data", "Flood Data"])


fire_gdf = load_fire_data()
landslide_gdf = load_landslide_data()
flood_gdf = load_flood_data()

with tab0:
    st.header("📊 Project Overview")

    st.markdown("""
        ### Welcome to the Oregon Natural Disaster Dashboard

        This interactive dashboard visualizes historical data on **fires**, **landslides**, and **floods** across Oregon counties. 
        """)

    # Key Features
    st.subheader("Key Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            **Interactive Maps**
            - Toggle between bubble and dot map views
            - Filter by year range (1960-present)
            - Visualize fires, landslides, and flood zones
            - Overlay roads and county boundaries
            """)

        st.markdown("""
            **Uncertainty Analysis**
            - County-level confidence ratings
            - Causal factor relationships
            - Risk assessment visualization
            """)

    with col2:
        st.markdown("""
            **Statistical Analysis**
            - Fires per year and by county
            - Landslide temporal patterns
            - Flood elevation distributions
            - Top affected regions
            """)

        st.markdown("""
            **Data-Driven Insights**
            - Identify high-risk areas
            - Track temporal trends
            - Compare disaster types
            """)

    # How to Use
    st.subheader("How to Use This Dashboard")
    st.markdown("""
        1. **Maps Tab**: Explore spatial distribution of disasters. Use the year slider to filter data and toggle between visualization styles.
        2. **Uncertainty Map Tab**: View confidence levels for causal relationships between environmental factors and disasters.
        3. **Fire/Landslide/Flood Tabs**: Dive into detailed statistics and temporal trends for each disaster type.
        """)

    # Data Sources
    st.subheader("Data Sources")
    st.markdown("""
        - **Fire Data**: Oregon Department of Forestry (ODF)
        - **Landslide Data**: Oregon SLIDO (Statewide Landslide Information Database)
        - **Flood Data**: FEMA Base Flood Elevation (BFE) database
        - **Geographic Data**: U.S. Census Bureau (roads), BLM (county boundaries)
        """)

    # Quick Stats
    st.subheader("Dataset Summary")

    # Calculate some quick stats
    total_fires = len(fire_gdf)
    total_landslides = len(landslide_gdf)
    total_flood_zones = len(flood_gdf)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Fire Records", f"{total_fires:,}")
    with col2:
        st.metric("Total Landslide Records", f"{total_landslides:,}")
    with col3:
        st.metric("Total Flood Zone Features", f"{total_flood_zones:,}")

    # Methodology
    st.subheader("Methodology")
    st.markdown("""
        **Uncertainty Ratings** represent our confidence in causal relationships:
        - **High Confidence (Green)**: Strong evidence supports the relationship
        - **Medium Confidence (Yellow)**: Moderate evidence, some uncertainty remains
        - **Low Confidence (Red)**: Limited evidence, high uncertainty

        **Map Visualizations**:
        - Bubble maps scale markers by fire size (acres) and landslide volume (cubic feet)
        - Dot maps show individual incident locations
        - All data is projected to EPSG:4326 (WGS84) for consistent mapping
        """)

    # Footer
    st.markdown("---")
    st.caption(
        "**Tip**: Use the tabs above to explore different aspects of the data. Start with the Maps tab for a visual overview!")

with tab1:

    fire_gdf["PreparedDate"] = pd.to_datetime(
        fire_gdf["PreparedDate"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )

    fire_years = fire_gdf["PreparedDate"].dt.year.dropna().astype(int)


    st.header("Oregon Fire, Landslide, & Flood Map")

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
    This map visualizes **uncertainty levels** by county for three factors:  
    - **Fire Uncertainty**: How certain we are that sparse trees results in fewer fires.  
    - **Landslide Uncertainty**: How certain we are that low-slope terrain results in fewer landslides.  
    - **Flood Uncertainty**: How certain we are that high elevation results in fewer floods.
    """)


    show_roads = True
    show_counties = True

    uncertainty_option = st.radio(
        "Select Uncertainty Factor to Display:",
        options=["None", "Fire Uncertainty", "Landslide Uncertainty", "Flood Uncertainty"],
        index=0  # Default to "None"
    )

    # Set flags based on selection
    show_fire = (uncertainty_option == "Fire Uncertainty")
    show_landslide = (uncertainty_option == "Landslide Uncertainty")
    show_flood = (uncertainty_option == "Flood Uncertainty")

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


