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
from streamlit_option_menu import option_menu
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import io 
import streamlit as st 
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

#FALTAAAA MUDAR A BASE DE DADOSS!!!!!!!!
st.set_page_config(page_title="Datathon", page_icon="🗼", layout="wide", menu_items=None)

df_final_2 = pd.read_csv("C:/Users/luisa/Downloads/df_final (2).csv",compression= 'zip')
condition1 = df_final_2['nombre_pieces_principales'] > 0 
condition11 =  df_final_2['nombre_pieces_principales'] < 8 
condition111 =df_final_2['surface_reelle_bati'] < 110
condition1111 =df_final_2['surface_reelle_bati'] > 40

df_final = df_final_2[condition1 & condition11 & condition111 & condition1111][['id_mutation','surface_reelle_bati','nombre_pieces_principales','type_local','nature_mutation','nom_commune','code_departement','valeur_fonciere','longitude', 'latitude','date_mutation']]
df_final[['year','month','day']] = df_final.date_mutation.str.split("-", expand=True)
df_final['price_m2'] = df_final['valeur_fonciere']/df_final['surface_reelle_bati']
df_apt = df_final[df_final['type_local']=='Appartement']
df_mai = df_final[df_final['type_local']=='Maison']
   
with st.sidebar:
    choose = option_menu(None, ["The city", "EDA", "Prediction", "Machine Learning"],
                         icons=['house', 'bi bi-calendar3', 'kanban', 'bi bi-geo-alt'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#E5E5EA"},#E5E5EA
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#F91A00"},
        "nav-link-selected": {"background-color": "darkblue"},
    }
    )
# with st.sidebar:
#     choose = option_menu(None, ["The city", "EDA", "Prediction", "Machine Learning"],
#                          icons=['house', 'bi bi-calendar3', 'kanban', 'bi bi-geo-alt'],
#                          menu_icon="app-indicator", default_index=0,
#                          styles={
#         "container": {"padding": "5!important", "background-color": "#F0F2F6"},#E5E5EA
#         "icon": {"color": "black", "font-size": "25px"}, 
#         "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#F0F2F6"},
#         "nav-link-selected": {"background-color": "#F0F2F6"},
#     }
#     )
if choose == "The city":   
    col1, col2= st.columns(2)
    image = Image.open("C:/Users/luisa/OneDrive/Ambiente de Trabalho/hackathon/eiffel-tower-4582649_1280.webp")

    image2 = Image.open("C:/Users/luisa/OneDrive/Ambiente de Trabalho/hackathon/paris.jpg")

    with col1:
        st.image(image)
    with col2:
        st.header('Île-de-France')
        st.subheader('*Where should you live?*')
        st.video("https://www.youtube.com/watch?v=F5Zr_mdRIiU&t=36s")

if choose == "EDA":  
    col31, col32= st.columns(2)
    with col31:
        # type = st.radio('  ' , options = ['Appartement', 'Maison']) 
        local = ['Appartement', 'Maison']
        data = [398420, 172941]
        colors = ['darkblue', 'red']
        explode = [0.1, 0]
        fig, ax = plt.subplots(figsize=(8, 3))
        plt.pie(data, labels = local, explode=explode, autopct='%.0f%%', wedgeprops = { 'linewidth' : 4, 'edgecolor' : 'black'}, colors=colors)
        # plt.title("Type of locals in Ile de france")
        st.pyplot(fig)
    with col32:
        fig32, ax32 = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=df_final, x="year", y="price_m2", hue="type_local", palette=['red', 'blue'])
        plt.title("Price per m2")
        ax32.set_xlabel('Year')
        ax32.set_ylabel('m2')
        st.pyplot(fig32)

        # fig33 = plt.subplots(figsize=(6, 4))
    fig33 = px.histogram(df_final, x=df_final["code_departement"].astype(str), color="type_local", color_discrete_sequence=["red", "darkblue"], title="Types of local per Departement").update_xaxes(categoryorder="total descending")
    plt.title("Price per m2")
        # ax33.set_xlabel('Year')
        # ax33.set_ylabel('m2')
    st.plotly_chart(fig33)
    
    type = st.radio('  ' , options = ['Appartement', 'Maison'])  
        

    if type == 'Appartement':
       
        col5, col7, col9= st.columns(3)
            # with col3:
            #     local = ['Appartement', 'Maison', 'Local industriel. commercial ou assimilé']
            #     data = [439163, 202338, 65161]
            #     colors = ['darkblue', 'white', 'white']
            #     explode = [0.1, 0, 0]
            #     fig1, ax = plt.subplots(figsize=(2, 2))
            #     plt.pie(data, explode=explode, colors=colors)
            #     plt.title("Appartement")
            #     st.pyplot(fig1)
        with col5:
                fig2, ax4 = plt.subplots(figsize=(10, 8))
                sns.boxplot(data=df_apt, x="nombre_pieces_principales", color='darkblue')
                plt.title("Pieces Principales")
                ax4.set_xlabel('.')
                st.pyplot(fig2)
        with col7:
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                sns.boxplot(data=df_apt, x="surface_reelle_bati", color='white')
                plt.title("Surface")
                ax3.set_xlabel('.')
                st.pyplot(fig3)
        with col9:
                fig9, ax9 = plt.subplots(figsize=(14, 12))
                sns.histplot(data=df_apt, x=df_apt["code_departement"].astype(str), color='red')
                # px.histogram(df_apt, x=df_apt["code_departement"].astype(str)).update_xaxes(categoryorder="total descending")
                plt.title("Departement")
                ax9.set_xlabel('.')
                st.pyplot(fig9)
    if type == 'Maison':
            col6, col8, col10= st.columns(3)
            # with col4:
            #     local = ['Appartement', 'Maison', 'Local industriel. commercial ou assimilé']
            #     data = [439163, 202338, 65161]
            #     colors = ['white', 'red', 'white']
            #     explode = [0.1, 0, 0]
            #     fig4, ax = plt.subplots(figsize=(2, 2))
            #     plt.pie(data, explode=explode, colors=colors)
            #     st.pyplot(fig4)
            with col6:
                fig6, ax6 = plt.subplots(figsize=(10, 8))
                sns.boxplot(data=df_mai, x="nombre_pieces_principales", color='darkblue')
                plt.title("Pieces Principales")
                ax6.set_xlabel('.')
                st.pyplot(fig6)
            with col8:
                fig9, ax8 = plt.subplots(figsize=(10, 8))
                sns.boxplot(data=df_mai, x="surface_reelle_bati", color='white')
                # ax8.set_xlabel('Surface')
                plt.title("Surface")
                ax8.set_xlabel('.')
                st.pyplot(fig9)
            with col10:
                fig11, ax8 = plt.subplots(figsize=(14, 12))
                sns.histplot(data=df_mai, x=df_mai["code_departement"].astype(str), color='red')
                # ax8.set_xlabel('Surface')
                plt.title("Departement")
                ax8.set_xlabel('.')
                st.pyplot(fig11)


if choose == "Prediction":  
    col3, col4= st.columns(2)
    image = Image.open("C:/Users/luisa/OneDrive/Ambiente de Trabalho/hackathon/eiffel-tower-4582649_1280.webp")
    with col3:
        st.image(image)
    with col4:
        st.header('Machine Learning')
        
if choose == "Machine Learning":  
    col3, col4= st.columns(2)
    image = Image.open("C:/Users/luisa/OneDrive/Ambiente de Trabalho/hackathon/eiffel-tower-4582649_1280.webp")
    with col3:
        st.image(image)
    with col4:
        st.header('Machine Learning')