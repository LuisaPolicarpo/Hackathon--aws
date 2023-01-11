### Imports 
import streamlit as st 
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

### Load Dataset
df = pd.read_csv('G:\Mi unidad\Hackathon--aws\Data\df_final.csv', compression='zip')
df.head()

### Preprocessing for ML
df_dummies1 = pd.concat([df , df['type_local'].str.get_dummies()], 
          axis = 1)
df_final = pd.concat([df_dummies1 , df_dummies1['nature_mutation'].str.get_dummies()], 
          axis = 1)

### Main Function
# Drop NaN
dft =  df_final.dropna()

# Characteristics for the model 
characteristics = ['valeur_fonciere','surface_reelle_bati','nombre_pieces_principales','longitude','latitude',
'Appartement','Local industriel. commercial ou assimilé','Maison','Adjudication','Echange','Expropriation','Vente',"Vente en l'état futur d'achèvement",
'Vente terrain à bâtir']

# Split the data into Train and test set
df_train, df_test = train_test_split(dft, test_size=0.2, random_state=7)

# Fit the model 
model = NearestNeighbors(n_neighbors = 3, algorithm = 'ball_tree', metric ='cityblock'). fit(df_train[characteristics])

def recommend_neighborhoods(current_neighborhood):
    distance, index1 = model.kneighbors(df_train[df_train['nom_commune'] == current_neighborhood][characteristics])
    similar_neighborhoods = df_train.iloc[index1.flat]['nom_commune']
    similar_neighborhoods = similar_neighborhoods[similar_neighborhoods != current_neighborhood].head(3)
    return similar_neighborhoods

st.title = 'Our app'
name = st.text_input('Type the title and press Enter')
st.table = recommend_neighborhoods(name)
