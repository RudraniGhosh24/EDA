# Crimes Against Women in India - EDA & Dashboard

This repository contains an extensive Exploratory Data Analysis (EDA) and an interactive Streamlit dashboard analyzing crimes against women in India using historical crime and census data (2001-2020).

## Project Overview

The primary objective of this project is to explore and visualize the trends, distribution, and sociological correlations of crimes against women across different states and districts in India.

We combined two major datasets:
1. **Crime Dataset**: Statistics of various crimes against women (Rape, Dowry Deaths, Kidnapping, etc.) from 2001-2020.
2. **Census Dataset**: Demographics, literacy rates, urban/rural populations, and basic infrastructure stats.

## What's Included

* **Jupyter Notebooks**: Step-by-step data cleaning, merging, PCA, and static visualizations using Matplotlib and Seaborn.
* **Streamlit Dashboard (`app.py`)**: An interactive web application built with Streamlit and Plotly to make the data exploration more intuitive.
* **Datasets**: Processed datasets used for the final analysis (`all_done_2.csv`, `cleaned_dataset.csv`).

## Key Features of the Dashboard

1. **Normalized Crime Rates**: Instead of looking at absolute crime numbers (which heavily skew towards highly populated districts), the dashboard normalizes the data to show the **Crime Rate per 100,000 women**.
2. **Geospatial Mapping**: A Choropleth map of India showing the intensity of crime rates across states.
3. **Interactive Crime Trends**: Line charts to track how specific crimes (e.g., Dowry Deaths vs. Kidnapping) have trended over the selected timeframe.
4. **Sociological Correlations & Feature Engineering**: We engineered new features like:
   - *Gender Ratio* (Females per 1000 Males)
   - *Literacy Gap* (Male Literacy Rate - Female Literacy Rate)
   - *Urbanization Rate* (%)
   These features can be interactively plotted against crime rates using the Scatter Analysis and Correlation Heatmap tabs.


## Repository Structure

* `app.py`: The main Streamlit application script.
* `EDA_Review3.ipynb`, `EDA_final1.ipynb`: Core notebooks showcasing data preprocessing, PCA, and static plots.
* `all_done_2.csv`, `cleaned_dataset.csv`: Cleaned and merged data files.
* `extra/`: Additional notebooks and unmerged CSV files from early EDA stages.

## Citations

- Patel, Anjum (2022), “Crime Against Women Dataset for Dark Spot identification in India”, Mendeley Data, V1, doi: [10.17632/whrdh8c5zb.1](https://doi.org/10.17632/whrdh8c5zb.1)
