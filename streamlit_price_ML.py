#import libraries
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#page config
st.set_page_config(page_title="Data Readers_app", layout="wide", menu_items=None)
st.title('Hello')

df_dummies2 = pd.read_csv('df_dummies2.csv', compression = 'zip')

#rename column 'Local industriel. commercial ou assimilé'
df_dummies2.rename(columns={'Local industriel. commercial ou assimilé' : 'Local_industriel_commercial_ou_assimilé', 'Vente en l\'état futur d\'achèvement' : 'Vente_en_létat_futur_dachèvement', 'Vente terrain à bâtir' : 'Vente_terrain_à_bâtir'}, inplace = True)

#converto to float
pd.to_numeric(df_dummies2['surface_reelle_bati'], errors="ignore")

st.table(df_dummies2.head())

# Train-test-split
X = df_dummies2[['surface_reelle_bati', 'nombre_pieces_principales', 'Appartement',
       'Local_industriel_commercial_ou_assimilé', 'Maison', 'Adjudication',
       'Echange', 'Expropriation', 'Vente',
       'Vente_en_létat_futur_dachèvement', 'Vente_terrain_à_bâtir', 'code_commune_fact']]
y = df_dummies2['valeur_fonciere']

# We set the size of the train set to 75%. And the rest is for the test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.75)
print("The length of the initial dataset is :", len(X))
print("The length of the train dataset is   :", len(X_train))
print("The length of the test dataset is    :", len(X_test))


#LR without scalers
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
print(model_lr.score(X_train, y_train))
print(model_lr.score(X_test, y_test))

# #predict values
# y_pred = model_lr.predict(X_test)

# df_predict = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
# st.table(df_predict.head())

#user inputs
# island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson']) 
# sex = st.selectbox('Penguin Sex', options=['Female', 'Male'])  
# bill_length_mm = st.number_input('Bill Length (mm)', min_value=0) 
# bill_depth_mm = st.number_input('Bill Depth (mm)', min_value=0) 
# flipper_length = st.number_input('Flipper Length (mm)', min_value=0) 
# body_mass = st.number_input('Body Mass (mm)', min_value=0)

surface_reelle_bati = st.slider(
    'Select a range of values for the m2',
    0.0, 1000.0, (50.0, 100.0))
st.write('Values:', surface_reelle_bati)

nombre_pieces_principales = st.number_input('nombre_pieces_principales', min_value=0) 

type_local = st.selectbox('type_local', options=['Appartement', 'Local_industriel_commercial_ou_assimilé', 'Maison'])

nature_mutation = st.selectbox('nature_mutation', options=['Adjudication', 'Echange', 'Expropriation', 'Vente', 'Vente_en_létat_futur_dachèvement', 'Vente terrain à bâtir'])
    
def predict(surface_reelle_bati, nombre_pieces_principales, type_local, nature_mutation):
    #Predicting the price
    Appartement == int(type_local == 'Appartement')
    Local_industriel_commercial_ou_assimilé == int(type_local == 'Local_industriel_commercial_ou_assimilé')
    type_local == int(type_local == 'Maison')
    
    
    Adjudication = int(nature_mutation == 'Adjudication')     
    Echange = int(nature_mutation == 'Echange')    
    Expropriation == nature_mutation == 'Expropriation'    
    Vente == int(nature_mutation == 'Vente')     
    Vente_en_létat_futur_dachèvement == int(nature_mutation == 'Vente_en_létat_futur_dachèvement')    
    Vente_terrain_à_bâtir == int(nature_mutation == 'Vente_terrain_à_bâtir')     

    prediction = model_lr.predict(
        
        pd.DataFrame([['surface_reelle_bati', 'nombre_pieces_principales', 'Appartement',
       'Local_industriel_commercial_ou_assimilé', 'Maison', 'Adjudication',
       'Echange', 'Expropriation', 'Vente',
       'Vente_en_létat_futur_dachèvement', 'Vente_terrain_à_bâtir', 'code_commune_fact']], columns=['surface_reelle_bati', 'nombre_pieces_principales', 'Appartement',
       'Local_industriel_commercial_ou_assimilé', 'Maison', 'Adjudication',
       'Echange', 'Expropriation', 'Vente',
       'Vente_en_létat_futur_dachèvement', 'Vente_terrain_à_bâtir', 'code_commune_fact']))
    
    return st.table(prediction)

if st.button('Predict Price'):
    price = predict(surface_reelle_bati, nombre_pieces_principales, type_local, nature_mutation)
    st.success(f'The predicted price of the diamond is ${price[0]:.2f} €')
    
# new_prediction = model_lr.predict([[surface_reelle_bati, nombre_pieces_principales, type_local, nature_mutation]])