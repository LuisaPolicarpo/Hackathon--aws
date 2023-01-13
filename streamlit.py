### ImportS
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

### Page config
st.set_page_config(page_title="Data Readers_app", layout="wide", menu_items=None)
st.title('Hello')

### Load the datasets
df = pd.read_csv('G:\Mi unidad\Hackathon--aws\Data\df_final.csv', compression='zip')
df_dummies2 = pd.read_csv('G:\Mi unidad\Hackathon--aws\df_dummies2.csv', compression='zip')

#Preprocessing of df for Neighborhood
df_dummies1 = pd.concat([df , df['type_local'].str.get_dummies()], 
          axis = 1)
df_final = pd.concat([df_dummies1 , df_dummies1['nature_mutation'].str.get_dummies()], 
          axis = 1)
df_final['price_m2'] = df_final['valeur_fonciere']/df_final['surface_reelle_bati']
list_departments = [75, 77, 78, 91, 92, 93, 94, 95]
df_final = df_final[df_final['code_departement'].isin(list_departments)]


### NEIGHBORHOOD RECOMENDATION
# Drop NaN
dft =  df_final.dropna()
# Characteristics for the model 
characteristics = ['valeur_fonciere','surface_reelle_bati','nombre_pieces_principales','longitude','latitude',
'Appartement','Local industriel. commercial ou assimilé','Maison','Adjudication','Echange','Expropriation','Vente',"Vente en l'état futur d'achèvement",
'Vente terrain à bâtir','price_m2']

df_train, df_test = train_test_split(dft, test_size=0.2, random_state=7)

parameters = {'n_neighbors':3, 
                'algorithm':'ball_tree', 
                'metric': 'cityblock'
}
model_N = NearestNeighbors(n_neighbors = 3, algorithm = 'ball_tree', metric ='cityblock'). fit(df_train[characteristics])

# Function
def recommend_neighborhoods(current_neighborhood):
    distance, index1 = model_N.kneighbors(df_train[df_train['nom_commune'] == current_neighborhood][characteristics])
    similar_neighborhoods = df_train.iloc[index1.flat]['nom_commune']
    similar_neighborhoods = similar_neighborhoods[similar_neighborhoods != current_neighborhood].head(2)
    return similar_neighborhoods

user_district = st.selectbox('Enter where you live', df_final['nom_commune'].unique())
district = recommend_neighborhoods(user_district)
st.subheader(district.values[0])
st.subheader(district.values[1])

## PRICE  
#Splits 
xcolumns = ['price_m2', 'surface_reelle_bati', 'nombre_pieces_principales', 'Appartement',
'Local industriel. commercial ou assimilé', 'Maison', 'Adjudication',
'Echange', 'Expropriation', 'Vente','Vente en l\'état futur d\'achèvement', 'Vente terrain à bâtir','code_commune_fact']


#District 1
dft0 = df_dummies2[df_dummies2['nom_commune']== district.values[0]]
X0 = dft0[xcolumns]
y0 = dft0['valeur_fonciere']
X_train0, X_test0, y_train0, y_test0 = train_test_split(X0, y0, test_size=0.2, random_state=7)

#District 2
dft01 = df_dummies2[df_dummies2['nom_commune']== district.values[1]]
X1 = dft01[xcolumns]
y1 = dft01['valeur_fonciere']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=7)

# Functions
def predict_house_price0(sqm, rooms, model, X_train0):
    model.fit(X_train0, y_train0)
    data = {'surface_reelle_bati': sqm, 'nombre_pieces_principales': rooms}
    df = pd.DataFrame(data, index=[0])

    df = pd.get_dummies(df)
    missing_cols = set(X_train0.columns) - set(df.columns)
    
    for c in missing_cols:
        df[c] = X_train0[c].median()
    df = df[X_train0.columns]
   
    prediction = model.predict(df)
    return prediction[0]

def predict_house_price1(sqm, rooms, model, X_train1):
    model.fit(X_train1, y_train1)
    data = {'surface_reelle_bati': sqm, 'nombre_pieces_principales': rooms}
    df = pd.DataFrame(data, index=[0])

    df = pd.get_dummies(df)
    missing_cols = set(X_train1.columns) - set(df.columns)
    
    for c in missing_cols:
        df[c] = X_train1[c].median()
    df = df[X_train1.columns]
   
    prediction = model.predict(df)
    return prediction[0]

model_lr = LinearRegression()

### Streamlit Code
user_sqm = st.number_input('Insert your desired sqm', min_value = 50, max_value = 1000)
user_beds = st.number_input('Insert your desired number of rooms', min_value = 1, max_value = 7)
reco0 = predict_house_price0(user_sqm, user_beds, model_lr, X_train0)
reco1  = predict_house_price1(user_sqm, user_beds, model_lr, X_train1)
st.subheader(district.values[0])
st.write('This is the predicted price:', round(reco0,2))
st.title('')
st.subheader(district.values[1])
st.write('This is the predicted price:', round(reco1,2))
