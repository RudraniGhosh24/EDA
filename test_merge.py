import pandas as pd

df_crimes = pd.read_csv("women-crimedataset-India.csv")
df_crimes.rename(columns={'STATE/UT': 'state', 'DISTRICT': 'District'}, inplace=True)
df_crimes['state'] = df_crimes['state'].str.strip().str.upper()
df_crimes['state'] = df_crimes['state'].replace({'MADHYA PRADESH  ': 'MADHYA PRADESH', 'A&N ISLANDS': 'A & N ISLANDS', 'D&N HAVELI': 'D & N HAVELI'})
df_crimes['District'] = df_crimes['District'].str.strip().str.upper()

df_census = pd.read_csv("cleaned_dataset.csv")
df_census['state'] = df_census['state'].str.strip().str.upper()
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
df_census['District'] = df_census['District'].str.strip().str.upper()
df_census = df_census.drop_duplicates(subset=['state', 'District'])

df = pd.merge(df_crimes, df_census[['state', 'District', 'Female', 'persons', 'Female lit_Rate', 'Male', 'rural']], on=['state', 'District'], how='left')

print(f"Crimes shape: {df_crimes.shape}")
print(f"Merged shape: {df.shape}")
print(f"Missing census data rows: {df['Female'].isna().sum()}")
