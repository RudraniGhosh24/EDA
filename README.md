# Crimes Against Women in India: A Causal and Spatiotemporal Dashboard

🚀 **Live Dashboard:** [https://crime-against-women-india.streamlit.app/](https://crime-against-women-india.streamlit.app/)

This repository contains an advanced interactive Streamlit dashboard and quantitative research framework analyzing crimes against women in India. By integrating National Crime Records Bureau (NCRB) administrative data with Census demographics (2001-2022), this project moves beyond standard descriptive Exploratory Data Analysis (EDA) to deploy rigorous **causal inference, hierarchical modeling, and Bayesian machine learning**.

## Project Overview

The primary objective of this project is to unearth the true socio-demographic drivers of violence against women in India. Traditional analyses often rely on raw crime counts, which suffer from severe population skews and administrative reporting biases. To correct this, we built a multi-tiered computational pipeline to normalize variables, test for spatial clustering, and isolate causal mechanisms.

## Key Features of the Dashboard

1. **Normalized Geospatial Mapping**: A state-level Choropleth map mapping the intensity of crime dynamically normalized to **Crime Rate per 100,000 women**, correcting for raw population density distortions.
2. **Hierarchical Mixed-Effects Modeling**: The dashboard moves beyond standard Ordinary Least Squares (OLS) panel regression by deploying a two-level mixed-effects hierarchical model. It calculates the Intraclass Correlation Coefficient (ICC) to mathematically partition the variance between state-level policy impacts and local, district-level demographic drivers.
3. **Structural Causal Modeling (DoWhy)**: Utilizing Directed Acyclic Graphs (DAGs) and Pearl's do-calculus, the model mathematically closes backdoor paths to isolate the exact causal Average Treatment Effect (ATE) of variables like the Male-Female Literacy Gap on crime rates, complete with Random Common Cause refutation stress-testing.
4. **Bayesian Structural Time Series (BSTS)**: Acts as a synthetic control method to isolate and quantify the exact statistical shock caused by the 2013 Criminal Law (Amendment) Act, proving that major legal reforms often trigger administrative reporting surges rather than immediate shifts in underlying criminal behavior.
5. **Bayesian Belief Networks (Risk Assessment)**: A probabilistic risk assessment matrix that maps highly nonlinear sociological interactions, querying the conditional probability of severe crime given overlapping demographic conditions (e.g., Urbanization vs. Literacy Gap).
6. **Inter-Crime Causal Pipelines**: Explicitly calculates whether specific offenses (such as Kidnapping & Abduction) act as direct, causal precursors to extreme sexual violence within localized districts.

## Repository Structure

* `app.py`: The core Streamlit application script containing all UI rendering and advanced ML modeling code.
* `all_done_2.csv`, `cleaned_dataset.csv`: Normalized and merged panel datasets mapping crime to census demographics.
* `requirements.txt`: Python dependencies required to run the causal and bayesian frameworks (e.g., `dowhy`, `pgmpy`, `ruptures`, `statsmodels`).

## How to Run Locally

1. Clone this repository: `git clone https://github.com/RudraniGhosh24/EDA.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the dashboard: `streamlit run app.py`

## Citations & Data Sources

- Patel, Anjum (2022), “Crime Against Women Dataset for Dark Spot identification in India”, Mendeley Data, V1, doi: [10.17632/whrdh8c5zb.1](https://doi.org/10.17632/whrdh8c5zb.1)
- National Crime Records Bureau (NCRB) Annual Publications.
- Census of India (2011).
