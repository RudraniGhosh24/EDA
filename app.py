import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="Crimes Against Women EDA", layout="wide")

st.title("Crimes Against Women in India - Enhanced EDA Dashboard")
st.markdown("Interactive Exploration of Crime and Census Data with Normalized Rates and Geographic Mapping")

@st.cache_data
def load_and_preprocess_data():
    # 1. Load Data
    df_crimes = pd.read_csv("women-crimedataset-India.csv")
    
    # Rename for consistency
    df_crimes.rename(columns={'STATE/UT': 'state', 'DISTRICT': 'District'}, inplace=True)
    
    df_census = pd.read_csv("cleaned_dataset.csv")

    # 2. Standardize Names for Merging
    df_crimes['state'] = df_crimes['state'].astype(str).str.strip().str.upper()
    df_crimes['state'] = df_crimes['state'].str.replace('STATE : ', '', regex=False)
    df_crimes['state'] = df_crimes['state'].replace({
        'A& N ISLANDS': 'A & N ISLANDS',
        'A&N ISLANDS': 'A & N ISLANDS',
        'CHATTISGARH': 'CHHATTISGARH',
        'D&N HAVELI': 'D & N HAVELI',
        'DAMAN': 'DAMAN & DIU',
        'DIU': 'DAMAN & DIU',
        'DELHI': 'DELHI UT',
        'MADHYA PRADESH  ': 'MADHYA PRADESH', 
        'MADHYAPRADESH': 'MADHYA PRADESH',
        'TAMILNADU': 'TAMIL NADU'
    })
    
    df_crimes['District'] = df_crimes['District'].astype(str).str.strip().str.upper()
    
    # Remove aggregate totals to prevent double-counting
    df_crimes = df_crimes[~df_crimes['state'].str.contains('TOTAL', na=False)]
    df_crimes = df_crimes[~df_crimes['District'].str.contains('TOTAL', na=False)]

    df_census['state'] = df_census['state'].astype(str).str.strip().str.upper()
    state_mapping = {
        'ANDAMAN': 'A & N ISLANDS', 'ANDHRA': 'ANDHRA PRADESH', 'ARUNACHAL PRADESH': 'ARUNACHAL PRADESH',
        'ASSAM': 'ASSAM', 'BIHAR': 'BIHAR', 'CHANDIGARH': 'CHANDIGARH', 'CHHATTISGARH': 'CHHATTISGARH',
        'D & N HAWELI': 'D & N HAVELI', 'DELHI': 'DELHI UT', 'GOA': 'GOA', 'GUJARAT': 'GUJARAT',
        'HARYANA': 'HARYANA', 'HP': 'HIMACHAL PRADESH', 'JK': 'JAMMU & KASHMIR', 'JHARKHAND': 'JHARKHAND',
        'KARNATAKA': 'KARNATAKA', 'KERALA': 'KERALA', 'LAKSHDWEEP': 'LAKSHADWEEP', 'MADHYA PRADESH': 'MADHYA PRADESH',
        'MAHARASHTRA': 'MAHARASHTRA', 'MANIPUR': 'MANIPUR', 'MEGHALYA': 'MEGHALAYA', 'MIZORAM': 'MIZORAM',
        'NAGALAND': 'NAGALAND', 'ORRISA': 'ODISHA', 'PONDICHERRY': 'PUDUCHERRY', 'PUNJAB': 'PUNJAB',
        'RAJASTHAN': 'RAJASTHAN', 'SIKKIM': 'SIKKIM', 'TN': 'TAMIL NADU', 'TRIPURA': 'TRIPURA',
        'UP': 'UTTAR PRADESH', 'UTTRANCHAL': 'UTTARAKHAND', 'WEST BENGAL': 'WEST BENGAL'
    }
    df_census['state'] = df_census['state'].map(lambda x: state_mapping.get(x, x))
    df_census['District'] = df_census['District'].astype(str).str.strip().str.upper()

    # Drop duplicate districts in census to avoid exploding merge
    df_census = df_census.drop_duplicates(subset=['state', 'District'])

    # Merge Datasets
    df = pd.merge(df_crimes, df_census[['state', 'District', 'Female', 'persons', 'Female lit_Rate', 'Male', 'rural']], on=['state', 'District'], how='left')
    
    # Ensure demographic columns are numeric (some contain '-')
    for col in ['Female', 'persons', 'Male', 'rural']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Feature Engineering & Normalization
    df['Female Population'] = df['Female'].fillna(df['Male']) 
    df['Total Population'] = df['persons'].fillna(df['Male'] * 2) 

    crime_cols = ['MURDER', 'ATTEMPT TO MURDER', 'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE',
                  'KIDNAPPING & ABDUCTION', 'DOWRY DEATHS', 'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
                  'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES', 'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES']
    
    # Exclude columns that don't exist in the new dataset safely
    crime_cols = [c for c in crime_cols if c in df.columns]

    for c in crime_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['Total Crimes Against Women'] = df[crime_cols].sum(axis=1)

    # Normalized Rates (safely avoid div by zero)
    df['Crime Rate (per 100k women)'] = df.apply(
        lambda row: (row['Total Crimes Against Women'] / row['Female Population']) * 100000 if pd.notnull(row['Female Population']) and row['Female Population'] > 0 else 0,
        axis=1
    )
    
    df['Gender Ratio (F per 1000 M)'] = (df['Female'] / df['Male']) * 1000
    if 'Male lit_Rate' in df_census.columns: # fallback if not present
        # Ensure lit_rate columns are numeric
        df['Male lit_Rate'] = pd.to_numeric(df_census['Male lit_Rate'], errors='coerce')
        df['Female lit_Rate'] = pd.to_numeric(df['Female lit_Rate'], errors='coerce')
        df['Literacy Gap'] = df['Male lit_Rate'] - df['Female lit_Rate']
    else:
        df['Literacy Gap'] = None
    
    # If urban_population is missing, derive it from total persons - rural
    if 'rural' in df.columns and 'persons' in df.columns:
        df['Urbanization Rate (%)'] = ((df['persons'] - df['rural']) / df['Total Population']) * 100
    else:
        df['Urbanization Rate (%)'] = None

    return df, crime_cols

@st.cache_data
def load_national_data():
    df_nat = pd.read_csv("Crime against Women (2001-2022).csv")
    # Melt it so years are in a single column
    df_melt = df_nat.melt(id_vars=["CRIME HEAD"], var_name="YEAR", value_name="Total Incidents")
    df_melt["YEAR"] = pd.to_numeric(df_melt["YEAR"], errors="coerce")
    df_melt["Total Incidents"] = pd.to_numeric(df_melt["Total Incidents"].astype(str).str.replace(',', ''), errors="coerce").fillna(0)
    return df_melt

df, crime_cols = load_and_preprocess_data()
df_national = load_national_data()

# -----------------
# SIDEBAR FILTERS
# -----------------
st.sidebar.header("Filter Data")
states = sorted(df['state'].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("Select State", ["All"] + states)

if selected_state == "All":
    districts = sorted(df['District'].dropna().unique().tolist())
else:
    districts = sorted(df[df['state'] == selected_state]['District'].dropna().unique().tolist())
    
selected_district = st.sidebar.selectbox("Select District", ["All"] + districts)

min_year, max_year = int(df['YEAR'].min()), int(df['YEAR'].max())
selected_years = st.sidebar.slider("Select Year Range (District Level)", min_year, max_year, (min_year, max_year))

filtered_df = df[(df['YEAR'] >= selected_years[0]) & (df['YEAR'] <= selected_years[1])]
if selected_state != "All":
    filtered_df = filtered_df[filtered_df['state'] == selected_state]
if selected_district != "All":
    filtered_df = filtered_df[filtered_df['District'] == selected_district]

# -----------------
# MAIN DASHBOARD
# -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview & Rates", "Geospatial Map", "District Crime Trends", "Feature Correlation", "National Macro Trends (2001-2022)"])

with tab1:
    st.header("Overview: Absolute Numbers vs Normalized Rates")
    st.markdown("Comparing absolute total crimes against women vs the **Crime Rate per 100,000 women**.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Crimes in Selection", int(filtered_df['Total Crimes Against Women'].sum()))
    with col2:
        avg_rate = filtered_df['Crime Rate (per 100k women)'].mean()
        st.metric("Average Crime Rate (per 100k women)", f"{avg_rate:.2f}" if pd.notna(avg_rate) else "N/A")

    if selected_state != "All" and selected_district == "All":
        st.subheader(f"District-wise Breakdown in {selected_state}")
        dist_agg = filtered_df.groupby('District').agg({
            'Total Crimes Against Women': 'sum',
            'Crime Rate (per 100k women)': 'mean'
        }).reset_index()
        
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.bar(dist_agg.sort_values('Total Crimes Against Women', ascending=False).head(10), 
                          x='District', y='Total Crimes Against Women', title='Top 10 Districts by Absolute Crimes')
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.bar(dist_agg.sort_values('Crime Rate (per 100k women)', ascending=False).head(10), 
                          x='District', y='Crime Rate (per 100k women)', title='Top 10 Districts by Crime Rate (Normalized)')
            st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Geospatial Map (Choropleth)")
    st.markdown("Visualizing Crime Rates across Indian States.")
    
    try:
        geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
        response = requests.get(geojson_url)
        india_geojson = response.json()
        
        state_agg = df[(df['YEAR'] >= selected_years[0]) & (df['YEAR'] <= selected_years[1])].groupby('state').agg({
            'Crime Rate (per 100k women)': 'mean',
            'Total Crimes Against Women': 'sum'
        }).reset_index()
        
        state_agg['state_title'] = state_agg['state'].str.title()
        
        # Historical data for J&K includes Ladakh. The modern map splits them.
        # We duplicate the J&K row to color Ladakh with the same historical rate.
        jk_row = state_agg[state_agg['state_title'] == 'Jammu & Kashmir'].copy()
        if not jk_row.empty:
            jk_row['state_title'] = 'Ladakh'
            state_agg = pd.concat([state_agg, jk_row], ignore_index=True)
            
        state_agg['state_title'] = state_agg['state_title'].replace({
            'Andaman & Nicobar Island': 'Andaman & Nicobar Islands',
            'Arunachal Pradesh': 'Arunanchal Pradesh', 
            'Delhi Ut': 'NCT of Delhi',
            'Odisha': 'Orissa'
        })
        
        fig_map = px.choropleth(
            state_agg,
            geojson=india_geojson,
            featureidkey='properties.ST_NM',
            locations='state_title',
            color='Crime Rate (per 100k women)',
            color_continuous_scale="Reds",
            title=f"Average Crime Rate per 100k women by State ({selected_years[0]}-{selected_years[1]})",
            hover_data=['Total Crimes Against Women']
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load map: {e}")

with tab3:
    st.header("District/State Level Crime Trends")
    selected_crimes = st.multiselect("Select Crimes to view trend:", crime_cols, default=['RAPE', 'DOWRY DEATHS'])
    
    if selected_crimes:
        trend_df = filtered_df.groupby('YEAR')[selected_crimes].sum().reset_index()
        trend_melted = trend_df.melt(id_vars='YEAR', value_vars=selected_crimes, var_name='Crime', value_name='Total Incidents')
        fig_trend = px.line(trend_melted, x='YEAR', y='Total Incidents', color='Crime', markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

with tab4:
    st.header("Engineered Features & Sociological Correlations")
    st.markdown("Explore how new features like Urbanization, Literacy Gap, and Gender Ratio correlate with Crime.")
    
    features = ['Crime Rate (per 100k women)', 'Total Crimes Against Women', 'Urbanization Rate (%)', 'Literacy Gap', 'Gender Ratio (F per 1000 M)', 'Female lit_Rate']
    available_features = [f for f in features if f in filtered_df.columns and filtered_df[f].notna().sum() > 0]
    
    if available_features:
        corr = filtered_df[available_features].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("Scatter Analysis")
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox("X-Axis", [f for f in ['Urbanization Rate (%)', 'Literacy Gap', 'Gender Ratio (F per 1000 M)'] if f in available_features])
        with c2:
            y_axis = st.selectbox("Y-Axis", ['Crime Rate (per 100k women)', 'Total Crimes Against Women'])
            
        scatter_df = filtered_df.dropna(subset=[x_axis, y_axis]).groupby(['state', 'District'])[available_features].mean().reset_index()
        if not scatter_df.empty:
            fig_scatter = px.scatter(scatter_df, x=x_axis, y=y_axis, hover_data=['state', 'District'], trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)

with tab5:
    st.header("National Macro Trends (2001-2022)")
    st.markdown("Using the newly provided 22-year national dataset for a top-down view.")
    
    nat_crimes = sorted(df_national['CRIME HEAD'].dropna().unique().tolist())
    selected_nat_crimes = st.multiselect("Select National Crimes:", nat_crimes, default=["Total Crimes against Women", "Rape"])
    
    if selected_nat_crimes:
        nat_filtered = df_national[df_national['CRIME HEAD'].isin(selected_nat_crimes)]
        fig_nat_trend = px.line(nat_filtered, x='YEAR', y='Total Incidents', color='CRIME HEAD', markers=True, title="National Incidents Over Time")
        st.plotly_chart(fig_nat_trend, use_container_width=True)
