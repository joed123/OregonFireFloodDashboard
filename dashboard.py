"""
dashboard.py - Main Streamlit Application

This is the main entry point for the Oregon Fire, Flood & Landslide Dashboard.
It orchestrates the user interface, data loading, and visualization display
using Streamlit's multi-tab layout.

Structure:
- Page configuration and title
- Data loading with error handling
- Six interactive tabs:
  1. Overview - Project information and dataset summary
  2. Maps - Interactive geographic visualizations with year filtering
  3. Uncertainty Map - County-level confidence ratings for causal factors
  4. Fire Data - Statistical analysis and charts for fire incidents
  5. Landslide Data - Temporal analysis of landslide occurrences
  6. Flood Data - Flood elevation and temporal pattern analysis

Dependencies:
- data.py: Data loading functions and configuration constants
- plots.py: All visualization and plotting functions

Run with: streamlit run dashboard.py
"""


import pandas as pd
import streamlit as st

import data
from data import load_fire_data, load_landslide_data, load_flood_data, load_roads, load_counties
from plots import (
    plot_bubble_map,
    plot_dot_map,
    plot_uncertainty_map_layers,
    plot_fire_by_county,
    plot_least_fire_by_county,
    plot_fires_by_year,
    plot_landslides_by_year,
    plot_floods_by_year,
    plot_flood_elevation_distribution,
    classify,
    fire_uncertainty,
    landslide_uncertainty,
    flood_uncertainty
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title=data.PAGE_TITLE,
    page_icon=data.PAGE_ICON,
)

st.title(data.PAGE_TITLE)
st.markdown("*An interactive visualization of natural disasters in Oregon*")

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

# =============================================================================
# TAB 0: OVERVIEW
# =============================================================================
# Displays project information, key features, usage instructions, data sources,
# dataset summary statistics, and methodology explanation. This is the landing
# page that introduces users to the dashboard's capabilities.
# =============================================================================

with tab0:
    st.header("Project Overview")

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

    st.markdown("""
        **Key Finding**: 
            Harney and Malheur counties emerge as the safest counties.
        """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Fire Records", f"{total_fires:,}")
    with col2:
        st.metric("Total Landslide Records", f"{total_landslides:,}")
    with col3:
        st.metric("Total Flood Zone Features", f"{total_flood_zones:,}")

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

    st.markdown("---")
    st.caption(
        "**Tip**: Use the tabs above to explore different aspects of the data. Start with the Maps tab for a visual overview!")


# =============================================================================
# TAB 1: INTERACTIVE MAPS
# =============================================================================
# Provides interactive geographic visualization of fire, landslide, and flood
# incidents across Oregon. Users can toggle between bubble maps (sized by
# incident severity) and dot maps (showing individual locations), filter by
# year range, and overlay roads and county boundaries for spatial context.
# =============================================================================

with tab1:

    fire_gdf["PreparedDate"] = pd.to_datetime(
        fire_gdf["PreparedDate"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )

    fire_years = fire_gdf["PreparedDate"].dt.year.dropna().astype(int)


    st.header("Oregon Fire, Landslide, & Flood Map")

    st.markdown("### Legend")
    
    legend_col1, legend_col2 = st.columns(2)
    
    with legend_col1:
        st.markdown("""
        **Dot Map**
        - 🔴 **Fires**: Individual fire locations
        - 🟤 **Landslides**: Individual landslide locations
        - 🔵 **Flood Zones**: FEMA flood zone areas
        - ⚫ **Roads**: Road network (optional overlay)
        - ⬜ **Counties**: County boundaries (optional overlay)
        """)
    
    with legend_col2:
      st.markdown("""
        **Bubble Map**
        - 🔴 **Fires**: Bubble size = fire size (acres)
        - 🟤 **Landslides**: Bubble size = volume (cubic feet)
        """)
    
    st.markdown("---")


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

# =============================================================================
# TAB 2: UNCERTAINTY MAP
# =============================================================================
# Displays county-level uncertainty/confidence ratings for causal relationships
# between environmental factors and disasters. Shows color-coded maps where:
# - Fire uncertainty: relationship between sparse trees and fire occurrence
# - Landslide uncertainty: relationship between slope and landslide occurrence
# - Flood uncertainty: relationship between elevation and flood occurrence
# Also includes a causal factor network diagram.
# =============================================================================

with tab2:

    st.header("Uncertainty Map")
    st.markdown("""
    This map visualizes **uncertainty levels** by county for three factors:  
    - **Fire Uncertainty**: How certain we are that sparse trees results in fewer fires.  
    - **Landslide Uncertainty**: How certain we are that low-slope terrain results in fewer landslides.  
    - **Flood Uncertainty**: How certain we are that high elevation results in fewer floods.
    """)

  # Static Legend
    st.markdown("### Legend")
    st.markdown("""
    **Confidence Levels** (based on evidence strength for causal relationships):
    - 🟢 **High Confidence (Green)**: Strong evidence supports the relationship
    - 🟡 **Medium Confidence (Yellow)**: Moderate evidence, some uncertainty remains
    - 🔴 **Low Confidence (Red)**: Limited evidence, high uncertainty
    """)
    st.markdown("---")


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


# =============================================================================
# TAB 3: FIRE DATA ANALYSIS
# =============================================================================
# Presents detailed statistical analysis and visualizations of fire incidents
# across Oregon, including:
# - Bar charts of top 10 most and least affected counties
# - Time series showing fire frequency trends from 1970 to present
# Helps identify high-risk areas and temporal patterns in fire occurrence.
# =============================================================================

with tab3:

    st.header("Fire Data Visualizations")

    st.subheader("Top 10 Counties Most Affected by Fires")
    plot_fire_by_county(fire_gdf)

    st.subheader("Top 10 Counties Least Affected by Fires")
    plot_least_fire_by_county(fire_gdf)

    st.subheader("Fires Per Year")
    st.pyplot(plot_fires_by_year(fire_gdf))

# =============================================================================
# TAB 4: LANDSLIDE DATA ANALYSIS
# =============================================================================
# Displays temporal analysis of landslide incidents across Oregon from 1965
# to present (excluding 1996 due to data quality issues). Shows year-over-year
# trends in landslide occurrence to identify patterns and high-risk periods.
# ============================================================================

with tab4:

    st.header("Landslide Data Visualizations")
    st.subheader("Landslides Per Year")
    st.pyplot(plot_landslides_by_year(landslide_gdf))

# =============================================================================
# TAB 5: FLOOD DATA ANALYSIS
# =============================================================================
# Provides analysis of FEMA Base Flood Elevation data including:
# - Histogram showing distribution of flood elevations across Oregon
# - Time series of flood zone features over time
# Helps understand flood risk patterns and elevation-related vulnerabilities.
# =============================================================================

with tab5:
    st.header("Flood Data Visualizations")

    st.subheader("Flood Elevation Distribution")
    st.pyplot(plot_flood_elevation_distribution(flood_gdf))
    st.subheader("Floods per Year")
    st.pyplot(plot_floods_by_year(flood_gdf))


