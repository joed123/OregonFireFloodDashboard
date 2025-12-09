# Oregon Natural Hazards Dashboard

A comprehensive data visualization dashboard for analyzing fire, flood, and landslide hazards across Oregon counties to identify the safest areas for living. 

## View it here: https://oregonfireflooddashboard.streamlit.app/

Check out my infographic-style map, "Oregon_map.png," too!

## Project Overview

This project aims to identify the least fire, flood, and landslide-prone areas in Oregon through advanced geospatial analysis and interactive data visualization. Using multiple visualization techniques, including dot maps, bubble maps, bar charts, and uncertainty maps, the dashboard provides insights into natural hazard patterns across Oregon's counties.

**Key Finding:** Harney and Malheur counties emerge as the safest areas based on the comprehensive analysis of historical hazard data.

## Features

### Interactive Maps
- **Dot Maps**: Visualize the spatial distribution of hazards with point data
- **Bubble Maps**: Show hazard severity with size-scaled markers
- **Uncertainty Maps**: Display confidence levels in hazard predictions by county

### Data Visualizations
- Time series analysis of hazards by year
- County-level hazard frequency rankings
- Flood elevation distribution analysis
- Interactive year range filtering

### File Descriptions
- dashboard.py - Main application entry point that creates the Streamlit interface with six interactive tabs (Overview, Maps, Uncertainty Map, Fire Data, Landslide Data, Flood Data).
- data.py - Handles all data loading operations with caching, contains configuration constants (colors, thresholds, file paths), and county mappings. Loads data from shapefiles, CSVs, and geodatabases.
- plots.py - Contains all visualization functions including map generators (bubble, dot, uncertainty maps) and statistical charts (time series, bar charts, histograms).

## Getting Started


1. Download the required datasets from the sources listed below and place them in a folder called "data"

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run dashboard.py
```


## Data Sources

**Fire Data:**
- Oregon Department of Forestry Fire Statistics
- https://www.oregon.gov/odf/fire/pages/firestats.aspx

**Landslide Data:**
- Oregon Department of Geology and Mineral Industries (DOGAMI) SLIDO Database
- https://www.oregon.gov/dogami/slido/pages/index.aspx

**Flood Data:**
- Oregon Statewide Flood Hazards Database
- https://www.arcgis.com/home/item.html?id=b1249fe6c1d743f7b9fd2474055293e1

**Geographic Boundaries:**
- Oregon County Boundaries: https://geohub.oregon.gov/datasets/51d396ec311d449fa2f6cb1141d496f2_1/explore
- Oregon Roads and Highways: https://catalog.data.gov/dataset/tiger-line-shapefile-2019-state-oregon-primary-and-secondary-roads-state-based-shapefile

## Technical Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **GeoPandas**: Geospatial data manipulation
- **Matplotlib**: Static visualizations
- **Pandas**: Data analysis and manipulation
- **Shapely**: Geometric operations
- **NumPy**: Numerical computing

## Dashboard Sections

1. **Maps Tab**: Interactive dot and bubble maps with year filtering
2. **Uncertainty Map Tab**: County-level confidence visualization
3. **Fire Data Tab**: Fire frequency analysis and county rankings
4. **Landslide Data Tab**: Temporal landslide pattern analysis
5. **Flood Data Tab**: Flood elevation and frequency distributions
