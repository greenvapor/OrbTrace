import ee
import streamlit as st
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# Initialize GEE
service_account = 'orbtrace@forest-in-rochester-hills.iam.gserviceaccount.com'
script_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(script_dir, '../../GoogleCloudKeys/forest-in-rochester-hills-084f70827372.json')
credentials = ee.ServiceAccountCredentials(service_account, key_path)
ee.Initialize(credentials)

st.title("Yearly Satellite Image NDVI Viewer")

# Location (Rochester Hills)
location = ee.Geometry.Point(-83.141, 42.658)

# UI: Multi-year selection, from 2015 to the current year
current_year = datetime.datetime.now().year
years = st.multiselect("Select years to compare NDVI", options=list(range(2015, current_year + 1)), default=[2020, 2021])

if not years:
    st.warning("Please select at least one year.")
    st.stop()

# Function to calculate NDVI (select bands first, then median; returns None if no data)
def get_yearly_ndvi(year):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate(start_date, end_date)
                  .filterBounds(location)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .select(['B4', 'B8']))  # Explicitly select bands
    
    count = collection.size().getInfo()
    if count == 0:
        return None
    
    image = collection.median()  # Bands are unified, so no error
    
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ndvi

# Calculate mean NDVI per year (returns NaN if None)
def get_mean_ndvi(ndvi_image, geometry):
    if ndvi_image is None:
        return np.nan
    stats = ndvi_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    )
    mean_val = stats.get('NDVI').getInfo()
    return mean_val if mean_val is not None else np.nan

# List to store mean NDVI values
mean_ndvi_values = []
ndvi_images = {}

for y in years:
    ndvi_img = get_yearly_ndvi(y)
    mean_val = get_mean_ndvi(ndvi_img, location)
    mean_ndvi_values.append({'year': y, 'mean_ndvi': mean_val})
    ndvi_images[y] = ndvi_img

df_ndvi = pd.DataFrame(mean_ndvi_values).sort_values('year')

# Plot NDVI mean values graph
st.subheader("NDVI Mean Values Over Selected Years")
fig, ax = plt.subplots()
ax.plot(df_ndvi['year'], df_ndvi['mean_ndvi'], marker='o', linestyle='-', color='green')
ax.set_xlabel("Year")
ax.set_ylabel("Mean NDVI")
ax.set_title("Yearly Mean NDVI Trend")
ax.grid(True)
st.pyplot(fig)

# Select year for NDVI map visualization
selected_map_year = st.selectbox("Select year for NDVI map visualization", years)
selected_ndvi_img = ndvi_images.get(selected_map_year)

if selected_ndvi_img is not None:
    ndvi_params = {
        'min': 0.0,
        'max': 1.0,
        'palette': ['white', 'yellowgreen', 'green']
    }
    map_id_dict = selected_ndvi_img.getMapId(ndvi_params)
    tile_url = map_id_dict['tile_fetcher'].url_format

    m = folium.Map(location=[42.658, -83.141], zoom_start=11)
    folium.TileLayer(
        tiles=tile_url,
        attr='Google Earth Engine',
        name=f'NDVI {selected_map_year}',
        overlay=True,
        control=True
    ).add_to(m)

    st.subheader(f"NDVI Map for {selected_map_year}")
    st_folium(m, width=700, height=500)
else:
    st.warning(f"No valid Sentinel-2 data available for year {selected_map_year}.")
