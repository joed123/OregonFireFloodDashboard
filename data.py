"""
data.py - Data Loading and Configuration Module

This module handles all data loading operations and configuration constants for
the Oregon Natural Disasters Dashboard. It loads geographic and disaster data
from various sources (shapefiles, CSVs, geodatabases) and provides caching to
improve performance.

Key Components:
- Configuration constants (file paths, colors, thresholds)
- County number-to-name mappings
- Cached data loading functions for:
  * Roads (shapefile)
  * County boundaries (shapefile with filtering)
  * Fire incidents (CSV to GeoDataFrame)
  * Landslide data (geodatabase)
  * Flood hazard zones (geodatabase)

All spatial data is reprojected to EPSG:4326 (WGS84) for consistency.
"""


import pandas as pd
import geopandas as gpd
import streamlit as st
from shapely.geometry import Point


PAGE_TITLE = "Oregon Fire, Flood & Landslide Dashboard"
PAGE_ICON = "🔥"

# Data filtering
FIRE_MIN_YEAR = 1970
FIRE_MIN_SIZE_ACRES = 10
FIRE_MAX_SIZE_ACRES = 5000
LANDSLIDE_MIN_YEAR = 1965
LANDSLIDE_EXCLUDE_YEAR = 1996
LANDSLIDE_MIN_VOLUME_FT3 = 1000
LANDSLIDE_MAX_VOLUME_FT3 = 1_000_000


FIRE_COLOR = "firebrick"
LANDSLIDE_COLOR = "saddlebrown"
FLOOD_COLOR = "royalblue"
ROAD_COLOR = "gray"
COUNTY_BORDER_COLOR = "white"

UNCERTAINTY_COLORS = {
    "high": "#006837",
    "medium": "#fee08b",
    "low": "#d73027",
}

# County mapping
COUNTY_NUMBER_TO_NAME = {
            1: "Baker", 2: "Benton", 3: "Clackamas", 4: "Clatsop", 5: "Columbia",
            6: "Coos", 7: "Crook", 8: "Curry", 9: "Deschutes", 10: "Douglas",
            11: "Gilliam", 12: "Grant", 13: "Harney", 14: "Hood River", 15: "Jackson",
            16: "Jefferson", 17: "Josephine", 18: "Klamath", 19: "Lake", 20: "Lane",
            21: "Lincoln", 22: "Linn", 23: "Malheur", 24: "Marion", 25: "Morrow",
            26: "Multnomah", 27: "Polk", 28: "Sherman", 29: "Tillamook", 30: "Umatilla",
            31: "Union", 32: "Wallowa", 33: "Wasco", 34: "Washington", 35: "Wheeler", 36: "Yamhill"
        }


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

