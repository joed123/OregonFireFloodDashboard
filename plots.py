"""
plots.py - Visualization Functions Module

This module contains all plotting and visualization functions for the Oregon
Natural Disasters Dashboard. It generates both geographic maps and statistical
charts using matplotlib and geopandas.

Key Components:
- Map Visualizations:
  * Bubble maps (sized by fire/landslide severity)
  * Dot maps (individual incident locations)
  * Uncertainty choropleth maps (county-level confidence ratings)

- Statistical Charts:
  * Time series plots (fires, landslides, floods by year)
  * Bar charts (counties most/least affected by fires)
  * Histograms (flood elevation distribution)

- Uncertainty Data:
  * Fire, landslide, and flood uncertainty dictionaries by county
  * Classification function for uncertainty mapping

All visualizations use consistent styling and color schemes defined in data.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import matplotlib.ticker as ticker
import data as cfg

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

        county_number_to_name = cfg.COUNTY_NUMBER_TO_NAME

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


        county_number_to_name = cfg.COUNTY_NUMBER_TO_NAME

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
