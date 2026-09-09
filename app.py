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
    
    # Strip police jurisdiction suffixes to fuse them into their geographic districts
    suffixes_to_remove = [
        r'\bCITY\b', r'\bCOMMR\.\b', r'\bRURAL\b', r'\bURBAN\b',
        r'\bRAILWAY\b', r'\bRLY\.\b', r'\bRLY\b', r'\bPOLICE\b', r'\bDISTRICT\b'
    ]
    for suffix in suffixes_to_remove:
        df_crimes['District'] = df_crimes['District'].str.replace(suffix, '', regex=True)
    
    df_crimes['District'] = df_crimes['District'].str.replace(r'[^A-Z ]', '', regex=True).str.strip()
    
    # Remove aggregate totals to prevent double-counting
    df_crimes = df_crimes[~df_crimes['state'].str.contains('TOTAL', na=False)]
    df_crimes = df_crimes[~df_crimes['District'].str.contains('TOTAL', na=False)]
    
    # Define crime columns safely
    crime_cols = ['MURDER', 'ATTEMPT TO MURDER', 'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE',
                  'KIDNAPPING & ABDUCTION', 'DOWRY DEATHS', 'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
                  'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES', 'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES']
    crime_cols = [c for c in crime_cols if c in df_crimes.columns]

    for c in crime_cols:
        df_crimes[c] = pd.to_numeric(df_crimes[c], errors='coerce').fillna(0)
        
    # FUSE subdivisions (e.g. Ahmedabad City + Rural) by summing crimes per year
    df_crimes = df_crimes.groupby(['state', 'District', 'YEAR'])[crime_cols].sum().reset_index()

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview & Rates", "Geospatial Map", "District Crime Trends", 
    "Feature Correlation", "National Macro Trends", "Advanced ML Analytics", "Causal & Probabilistic ML"
])

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

with tab6:
    st.header("Advanced ML Analytics")
    st.markdown("Using Machine Learning to study regional patterns and forecast future trends.")
    
    st.subheader("1. ARIMA Forecasting")
    st.markdown("Forecast crime trends for the next 5 years using the AutoRegressive Integrated Moving Average (ARIMA) model.")
    
    arima_level = st.radio("Forecast Level:", ["State", "District"], horizontal=True, key="arima_level")
    arima_crime = st.selectbox("Select Crime to Forecast:", crime_cols, index=2, key="arima_crime")
    
    if arima_level == "State":
        arima_region = st.selectbox("Select State:", sorted(df['state'].dropna().unique()), key="arima_state")
        ts_data = df[df['state'] == arima_region].groupby('YEAR')[arima_crime].sum().reset_index()
    else:
        arima_region = st.selectbox("Select District:", sorted(df['District'].dropna().unique()), index=0, key="arima_dist")
        ts_data = df[df['District'] == arima_region].groupby('YEAR')[arima_crime].sum().reset_index()
    
    ts_data = ts_data.sort_values('YEAR')
    
    # Fill any missing intermediate years with interpolation
    if len(ts_data) > 1:
        all_years = pd.DataFrame({'YEAR': range(int(ts_data['YEAR'].min()), int(ts_data['YEAR'].max()) + 1)})
        ts_data = all_years.merge(ts_data, on='YEAR', how='left')
        ts_data[arima_crime] = ts_data[arima_crime].interpolate(method='linear').fillna(0)
    
    if len(ts_data) >= 6:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            import plotly.graph_objects as go
            
            y = ts_data[arima_crime].values.astype(float)
            years = ts_data['YEAR'].values.astype(int)
            
            # Auto-select simpler order if series is short
            p = min(5, len(y) - 2)
            model = ARIMA(y, order=(p, 1, 0))
            model_fit = model.fit()
            forecast_result = model_fit.get_forecast(steps=5)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05)
            
            future_years = [int(years[-1]) + i for i in range(1, 6)]
            
            fig_arima = go.Figure()
            fig_arima.add_trace(go.Scatter(x=years.tolist(), y=y.tolist(), mode='lines+markers', name='Historical', line=dict(color='royalblue')))
            fig_arima.add_trace(go.Scatter(x=future_years, y=forecast.tolist(), mode='lines+markers', name='Forecast', line=dict(dash='dash', color='green')))
            
            # Confidence interval band
            fig_arima.add_trace(go.Scatter(
                x=future_years + future_years[::-1],
                y=conf_int.iloc[:, 1].tolist() + conf_int.iloc[:, 0].tolist()[::-1],
                fill='toself', fillcolor='rgba(0,200,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
            
            fig_arima.update_layout(
                title=f"ARIMA Forecast: {arima_crime} in {arima_region}",
                xaxis_title="Year", yaxis_title="Total Incidents"
            )
            st.plotly_chart(fig_arima, use_container_width=True)
            
            st.caption(f"Model: ARIMA({p},1,0) · AIC: {model_fit.aic:.1f} · {len(y)} data points")
        except Exception as e:
            st.warning(f"Could not fit ARIMA for this selection. Try a different region or crime. ({e})")
    else:
        st.warning(f"Only {len(ts_data)} year(s) of data available for {arima_region}. Need at least 6 for ARIMA.")
        
    st.markdown("---")
    st.subheader("2. Regional Clustering (PCA & DBSCAN)")
    st.markdown("Groups districts with similar socio-demographic and crime profiles using Principal Component Analysis and Density-Based Clustering.")
    
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import DBSCAN
        
        # Take the most recent year data for clustering
        latest_year = df['YEAR'].max()
        cluster_df = df[df['YEAR'] == latest_year].dropna(subset=crime_cols + ['Female Population'])
        
        if not cluster_df.empty:
            X = cluster_df[crime_cols]
            X_scaled = StandardScaler().fit_transform(X)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            dbscan = DBSCAN(eps=1.5, min_samples=3)
            clusters = dbscan.fit_predict(X_pca)
            
            cluster_df['PCA1'] = X_pca[:, 0]
            cluster_df['PCA2'] = X_pca[:, 1]
            cluster_df['Cluster'] = [str(c) if c != -1 else 'Outlier' for c in clusters]
            
            fig_cluster = px.scatter(
                cluster_df, x='PCA1', y='PCA2', color='Cluster',
                hover_data=['state', 'District'] + crime_cols[:2],
                title=f"PCA & DBSCAN Regional Clusters (Year {int(latest_year)})"
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
    except Exception as e:
        st.error(f"Clustering failed: {e}")

    st.markdown("---")
    st.subheader("3. Feature Selection (Lasso & RFE)")
    st.markdown("Identifying the most critical crimes that predict the overall normalized **Crime Rate** using Embedded and Wrapper methods.")
    
    try:
        from sklearn.linear_model import Lasso, LinearRegression
        from sklearn.feature_selection import RFE
        from sklearn.preprocessing import StandardScaler
        
        clean_ml = df.dropna(subset=crime_cols + ['Crime Rate (per 100k women)'])
        if len(clean_ml) > 100:
            X_fs = clean_ml[crime_cols]
            y_fs = clean_ml['Crime Rate (per 100k women)']
            X_fs_scaled = StandardScaler().fit_transform(X_fs)
            
            lasso = Lasso(alpha=0.1)
            lasso.fit(X_fs_scaled, y_fs)
            lasso_feats = X_fs.columns[lasso.coef_ != 0].tolist()
            
            lr = LinearRegression()
            rfe = RFE(estimator=lr, n_features_to_select=3)
            rfe.fit(X_fs_scaled, y_fs)
            rfe_feats = X_fs.columns[rfe.support_].tolist()
            
            c1, c2 = st.columns(2)
            with c1:
                st.info("**Lasso Regression (Embedded)** Selected Features:")
                st.write(lasso_feats if lasso_feats else "None strictly selected by Lasso")
            with c2:
                st.info("**Recursive Feature Elimination (RFE)** Top 3 Features:")
                st.write(rfe_feats)
    except Exception as e:
        st.error(f"Feature selection failed: {e}")
with tab7:
    st.header("Causal & Probabilistic Machine Learning")
    st.markdown("Moving beyond correlation and point-predictions to understand **why** crimes occur and quantify **uncertainty**.")
    
    st.subheader("1. Causal Impact Analysis (Interrupted Time Series)")
    st.markdown("Did the **2013 Criminal Law (Amendment) Act** (post-Nirbhaya) cause a statistically significant increase in the reporting of crimes against women? We use a Bayesian Structural Time Series (BSTS) approach to create a synthetic control.")
    
    try:
        import statsmodels.api as sm
        # National aggregate data for 'Total Crimes against Women'
        nat_total = df_national[df_national['CRIME HEAD'] == 'Total Crimes against Women'].sort_values('YEAR')
        if not nat_total.empty:
            pre_period = nat_total[nat_total['YEAR'] < 2013]
            post_period = nat_total[nat_total['YEAR'] >= 2013]
            
            # Simple Local Level model trained on pre-intervention
            y_pre = pre_period['Total Incidents'].values
            model_bsts = sm.tsa.UnobservedComponents(y_pre, 'local level')
            res_bsts = model_bsts.fit(disp=False)
            
            # Forecast into post-intervention period to get the "counterfactual"
            forecast_steps = len(post_period)
            pred = res_bsts.get_forecast(steps=forecast_steps)
            counterfactual = pred.predicted_mean
            
            # UnobservedComponents pred.conf_int() returns a numpy array or dataframe depending on version
            conf_int = pred.conf_int(alpha=0.05)
            if hasattr(conf_int, 'iloc'):
                lower_ci = conf_int.iloc[:, 0].tolist()
                upper_ci = conf_int.iloc[:, 1].tolist()
            else:
                lower_ci = conf_int[:, 0].tolist()
                upper_ci = conf_int[:, 1].tolist()
            
            import plotly.graph_objects as go
            fig_causal = go.Figure()
            # Actual Data
            fig_causal.add_trace(go.Scatter(x=nat_total['YEAR'], y=nat_total['Total Incidents'], mode='lines+markers', name='Actual Reported Crimes', line=dict(color='red')))
            # Counterfactual Forecast
            fig_causal.add_trace(go.Scatter(x=post_period['YEAR'], y=counterfactual, mode='lines', name='Expected (Counterfactual) without 2013 Act', line=dict(dash='dash', color='blue')))
            # Confidence Interval
            fig_causal.add_trace(go.Scatter(
                x=post_period['YEAR'].tolist() + post_period['YEAR'].tolist()[::-1],
                y=upper_ci + lower_ci[::-1],
                fill='toself', fillcolor='rgba(0,0,255,0.1)', line=dict(color='rgba(255,255,255,0)'),
                name='95% CI (Counterfactual)'
            ))
            
            fig_causal.add_vline(x=2013, line_width=2, line_dash="dash", line_color="black")
            fig_causal.add_annotation(x=2013, y=nat_total['Total Incidents'].max(), text="2013 Amendment Act", showarrow=True, arrowhead=1)
            
            fig_causal.update_layout(title="Causal Impact of 2013 Amendment on Crime Reporting", xaxis_title="Year", yaxis_title="Total Incidents")
            st.plotly_chart(fig_causal, use_container_width=True)
            
            # Calculate causal effect
            actual_post = post_period['Total Incidents'].values
            effect = actual_post - counterfactual
            st.info(f"**Causal Insight:** The model estimates that the 2013 amendment *caused* an absolute increase in reporting of approximately **{int(effect.mean())}** extra cases per year compared to what would have been expected if the previous trend had continued.")
    except Exception as e:
        st.error(f"Causal Impact failed: {e}")

    st.markdown("---")
    st.subheader("2. Causal Graph Inference (DoWhy)")
    st.markdown("Does the Literacy Gap *cause* higher crime rates, or are they just correlated? Using Microsoft's `DoWhy` library, we build a Directed Acyclic Graph (DAG) to estimate the true causal effect, controlling for Urbanization and Gender Ratio.")
    
    try:
        import dowhy
        from dowhy import CausalModel
        
        # Prepare data for causal model - renaming columns to remove spaces for DAG parser
        causal_df = df.dropna(subset=['Crime Rate (per 100k women)', 'Literacy Gap', 'Urbanization Rate (%)', 'Gender Ratio (F per 1000 M)']).copy()
        causal_df = causal_df.rename(columns={
            'Crime Rate (per 100k women)': 'Crime_Rate',
            'Urbanization Rate (%)': 'Urbanization',
            'Gender Ratio (F per 1000 M)': 'Gender_Ratio',
            'Literacy Gap': 'Literacy_Gap'
        })
        
        # Binarize treatment for simpler visualization/computation
        median_gap = causal_df['Literacy_Gap'].median()
        causal_df['High_Literacy_Gap'] = causal_df['Literacy_Gap'] > median_gap
        causal_df['High_Literacy_Gap'] = causal_df['High_Literacy_Gap'].astype(bool)
        
        # Define the Causal Graph (DAG) with safe names
        causal_graph = """
        digraph {
        Urbanization -> High_Literacy_Gap;
        Urbanization -> Crime_Rate;
        Gender_Ratio -> High_Literacy_Gap;
        Gender_Ratio -> Crime_Rate;
        High_Literacy_Gap -> Crime_Rate;
        }
        """
        
        model = CausalModel(
            data=causal_df,
            treatment='High_Literacy_Gap',
            outcome='Crime_Rate',
            graph=causal_graph
        )
        
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("**Assumed Causal DAG:**")
            st.code(causal_graph, language='dot')
        with c2:
            st.success(f"**Estimated Causal Effect:** {estimate.value:.2f}")
            st.markdown(f"**Interpretation:** Moving a district from a 'Low Literacy Gap' to a 'High Literacy Gap' causes the Crime Rate to change by **{estimate.value:.2f}** incidents per 100k women, holding Urbanization and Gender Ratio constant.")
            
    except Exception as e:
        st.error(f"DoWhy Causal Inference failed: {e}. Note: `dowhy` and `networkx` must be installed.")

    st.markdown("---")
    st.subheader("3. Probabilistic Forecasting (Gaussian Processes)")
    st.markdown("Predicting crime rates based on demographic profiles with quantified uncertainty bounds.")
    
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        gp_df = df.dropna(subset=['Urbanization Rate (%)', 'Crime Rate (per 100k women)']).copy()
        # Take a random sample to keep GP fast
        gp_df = gp_df.sample(n=min(500, len(gp_df)), random_state=42)
        
        X_gp = gp_df[['Urbanization Rate (%)']].values
        y_gp = gp_df['Crime Rate (per 100k women)'].values
        
        scaler_X = StandardScaler()
        X_gp_scaled = scaler_X.fit_transform(X_gp)
        
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=10.0)
        gp.fit(X_gp_scaled, y_gp)
        
        x_pred = np.linspace(X_gp.min(), X_gp.max(), 100).reshape(-1, 1)
        x_pred_scaled = scaler_X.transform(x_pred)
        y_pred, sigma = gp.predict(x_pred_scaled, return_std=True)
        
        import plotly.graph_objects as go
        fig_gp = go.Figure()
        fig_gp.add_trace(go.Scatter(x=X_gp.flatten(), y=y_gp, mode='markers', name='Observed Districts', marker=dict(opacity=0.5, color='gray')))
        fig_gp.add_trace(go.Scatter(x=x_pred.flatten(), y=y_pred, mode='lines', name='GP Mean Prediction', line=dict(color='red')))
        fig_gp.add_trace(go.Scatter(
            x=np.concatenate([x_pred.flatten(), x_pred.flatten()[::-1]]),
            y=np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
            fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        fig_gp.update_layout(title="Gaussian Process Regression: Crime Rate vs Urbanization", xaxis_title="Urbanization Rate (%)", yaxis_title="Crime Rate (per 100k women)")
        st.plotly_chart(fig_gp, use_container_width=True)
        
    except Exception as e:
        st.error(f"Gaussian Process failed: {e}")

    st.markdown("---")
    st.subheader("4. Bayesian Belief Networks (Risk Assessment)")
    st.markdown("Query the conditional probability of experiencing High Crime given specific sociological conditions.")
    
    try:
        try:
            from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
        except ImportError:
            from pgmpy.models import BayesianNetwork
            
        from pgmpy.estimators import MaximumLikelihoodEstimator
        from pgmpy.inference import VariableElimination
        
        bn_df = df.dropna(subset=['Crime Rate (per 100k women)', 'Literacy Gap', 'Urbanization Rate (%)']).copy()
        
        # Discretize for Bayesian Network
        bn_df['Crime_Level'] = pd.qcut(bn_df['Crime Rate (per 100k women)'], q=3, labels=['Low', 'Medium', 'High'])
        bn_df['Urban_Level'] = pd.qcut(bn_df['Urbanization Rate (%)'], q=3, labels=['Low', 'Medium', 'High'])
        bn_df['LitGap_Level'] = pd.qcut(bn_df['Literacy Gap'], q=3, labels=['Low', 'Medium', 'High'])
        
        train_data = bn_df[['Urban_Level', 'LitGap_Level', 'Crime_Level']]
        
        bn_model = BayesianNetwork([('Urban_Level', 'Crime_Level'), ('LitGap_Level', 'Crime_Level')])
        bn_model.fit(train_data, estimator=MaximumLikelihoodEstimator)
        
        infer = VariableElimination(bn_model)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            u_sel = st.selectbox("Urbanization Level:", ['Low', 'Medium', 'High'], key='u_sel')
        with c2:
            l_sel = st.selectbox("Literacy Gap Level:", ['Low', 'Medium', 'High'], key='l_sel')
            
        prob = infer.query(variables=['Crime_Level'], evidence={'Urban_Level': u_sel, 'LitGap_Level': l_sel})
        
        # Extract probabilities
        p_low = prob.values[0]
        p_med = prob.values[1]
        p_high = prob.values[2]
        
        with c3:
            st.metric("P(Crime = High)", f"{p_high:.1%}")
            
        # Plot probabilities
        import plotly.express as px
        fig_bn = px.bar(x=['Low', 'Medium', 'High'], y=[p_low, p_med, p_high], labels={'x': 'Crime Level', 'y': 'Probability'}, title=f"Conditional Probability of Crime given Urban={u_sel} & LitGap={l_sel}")
        st.plotly_chart(fig_bn, use_container_width=True)
        
    except Exception as e:
        st.error(f"Bayesian Network failed: {e}. Note: `pgmpy` must be installed.")
