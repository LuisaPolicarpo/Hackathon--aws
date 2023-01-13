import streamlit as st
import pandas as pd
import numpy as np  
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from string import punctuation
import pickle
from PIL import Image
import requests
import streamlit_nested_layout
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import plotly.express as px
import io 
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Datathon", page_icon="🔎", layout="wide", menu_items=None)
### Load the datasets
df_dummies2 = pd.read_csv('df_dummies2.csv', compression='zip')
df = pd.read_csv('df_final.csv', compression = 'zip')
# df_final2 = pd.read_csv('LP_df_final.csv')
df_final4 = df.dropna(subset=['valeur_fonciere'])
df_final0= df_final4.fillna(0)
condition1 = df_final0['nombre_pieces_principales'] > 1 
condition11 =  df_final0['nombre_pieces_principales'] < 7 
condition111 =df_final0['surface_reelle_bati'] < 147.5
condition1111 =df_final0['surface_reelle_bati'] > 30.5

df_final2 = df_final0[condition1 & condition11 & condition111 & condition1111][['id_mutation','surface_reelle_bati','nombre_pieces_principales','type_local','nature_mutation','nom_commune','code_departement','valeur_fonciere','longitude', 'latitude','date_mutation']]
df_final2[['year','month','day']] = df_final2.date_mutation.str.split("-", expand=True)
df_final2['price_m2'] = df_final2['valeur_fonciere']/df_final2['surface_reelle_bati']
df_apt = df_final2[df_final2['type_local']=='Appartement']
df_mai = df_final2[df_final2['type_local']=='Maison']



#Preprocessing of df for Neighborhood
df_dummies1 = pd.concat([df , df['type_local'].str.get_dummies()], 
          axis = 1)
df_final = pd.concat([df_dummies1 , df_dummies1['nature_mutation'].str.get_dummies()], 
          axis = 1)
df_final['price_m2'] = df_final['valeur_fonciere']/df_final['surface_reelle_bati']
list_departments = [75, 77, 78, 91, 92, 93, 94, 95]
df_final = df_final[df_final['code_departement'].isin(list_departments)]
with st.sidebar:
    choose = option_menu(None, ["Appartement vs Maison", "Neighborhood recomendation"],
                         icons=['house', 'bi bi-calendar3', 'kanban', 'bi bi-geo-alt'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#E5E5EA"},#E5E5EA
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#F91A00"},
        "nav-link-selected": {"background-color": "darkblue"},
    }
    )

# if choose == "Île-de-France":   

#     image3 = Image.open("C:/Users/luisa/OneDrive/Ambiente de Trabalho/hackathon/Il de france.jpg")

#     st.header('**Île-de-France**')
#     st.subheader('_Where should you live?_')
#     st.image(image3, width=550)


if choose == "Appartement vs Maison":  
    col31, col32= st.columns(2)
    with col31:
 
        local = ['Appartement', 'Maison']
        data = [398420, 172941]
        colors = ['darkblue', 'red']
        explode = [0.1, 0]
        fig, ax = plt.subplots(figsize=(8, 3))
        plt.pie(data, labels = local, explode=explode, autopct='%.0f%%', wedgeprops = { 'linewidth' : 4, 'edgecolor' : 'black'}, colors=colors)

        st.pyplot(fig)
    with col32:
        fig32, ax32 = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=df_final2, x="year", y="price_m2", hue="type_local", palette=['red', 'darkblue'])
        plt.title("Price per m2")
        ax32.set_xlabel('Year')
        ax32.set_ylabel('m2')
        st.pyplot(fig32)


    fig33 = px.histogram(df_final2, x=df_final2["code_departement"].astype(str), color="type_local", color_discrete_sequence=["red", "darkblue"], title="Types of local per Departement", labels={"type_local": "Local", "x": "Departement"}).update_xaxes(categoryorder="total descending")
    plt.title("Price per m2")

    st.plotly_chart(fig33)
    
#     type = st.radio('  ' , options = ['Appartement', 'Maison'])  
        

#     if type == 'Appartement':
    st.header('Appartement')
    col5, col7= st.columns(2)

    with col5:
                fig2, ax4 = plt.subplots(figsize=(10, 8))
                df_apt_2= df_apt.sort_values(by='nombre_pieces_principales', ascending=True)
                sns.histplot(data=df_apt_2, x=df_apt_2["nombre_pieces_principales"].astype(str), color='darkblue')
                plt.title("Pieces Principales")
                ax4.set_xlabel('.')
                st.pyplot(fig2)
    with col7:
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                sns.boxplot(data=df_apt, x="surface_reelle_bati", color='darkblue')
                plt.title("Surface")
                ax3.set_xlabel('.')
                st.pyplot(fig3)

    # if type == 'Maison':
    st.header('Maison')
    col6, col8= st.columns(2)

    with col6:
                fig6, ax6 = plt.subplots(figsize=(10, 8))
                df_final_mai_2= df_mai.sort_values(by='nombre_pieces_principales', ascending=True)
                sns.histplot(data=df_final_mai_2, x=df_final_mai_2["nombre_pieces_principales"].astype(str), color='red')
                plt.title("Pieces Principales")
                ax6.set_xlabel('.')
                st.pyplot(fig6)
    with col8:
                fig9, ax8 = plt.subplots(figsize=(10, 8))
                sns.boxplot(data=df_mai, x="surface_reelle_bati", color='red')
                # ax8.set_xlabel('Surface')
                plt.title("Surface")
                ax8.set_xlabel('.')
                st.pyplot(fig9)



elif choose == "Neighborhood recomendation":  
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

    user_district = st.selectbox('Enter where you live', df_final['nom_commune'].sort_values().unique())
    district = recommend_neighborhoods(user_district)
    st.markdown('These are the recommended districts:')
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
    # st.subheader('')
    st.subheader(district.values[1])
    st.write('This is the predicted price:', round(reco1,2))
        
# elif choose == "3D Tour":   
#     col1, col2= st.columns(2)
#     image = Image.open("C:/Users/luisa/OneDrive/Ambiente de Trabalho/hackathon/eiffel-tower-4582649_1280.webp")
#     with col1:
#         st.header('Île-de-France')
#         st.subheader('*Where should you live?*')
#         st.video("https://www.youtube.com/watch?v=F5Zr_mdRIiU&t=36s")
#     with col2:

#         st.image(image, width=550)