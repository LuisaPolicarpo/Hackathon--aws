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

df_final = pd.read_csv("C:/Users/luisa/Downloads/df_final (2).csv",compression= 'zip')
df_apt = pd.read_csv('https://raw.githubusercontent.com/LuisaPolicarpo/Hackathon--aws/main/apt%20(1).csv')
df_mai = pd.read_csv('https://raw.githubusercontent.com/LuisaPolicarpo/Hackathon--aws/main/apt%20(1).csv')    
with st.sidebar:
    choose = option_menu(None, ["The city", "EDA", "Prediction", "Machine Learning"],
                         icons=['house', 'bi bi-calendar3', 'kanban', 'bi bi-geo-alt'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#E5E5EA"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#F91A00"},
        "nav-link-selected": {"background-color": "darkblue"},
    }
    )
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
        # type = st.radio('  ' , options = ['Appartement', 'Maison']) 
        local = ['Appartement', 'Maison', 'Local industriel. commercial ou assimilé']
        data = [439163, 202338, 65161]
        colors = ['darkblue', 'red', 'white']
        explode = [0.1, 0, 0]
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.pie(data, labels = local, explode=explode, autopct='%.0f%%', wedgeprops = { 'linewidth' : 4, 'edgecolor' : 'black'}, colors=colors)
        plt.title("Type of locals in Ile de france")
        st.pyplot(fig)
        
        type = st.radio('  ' , options = ['Appartement', 'Maison'])  
        

        if type == 'Appartement':
       
            col3, col4, col5= st.columns(3)
            with col3:
                local = ['Appartement', 'Maison', 'Local industriel. commercial ou assimilé']
                data = [439163, 202338, 65161]
                colors = ['darkblue', 'white', 'white']
                explode = [0.1, 0, 0]
                fig1, ax = plt.subplots(figsize=(2, 2))
                plt.pie(data, explode=explode, colors=colors)
                plt.title("Appartement")
                st.pyplot(fig1)
            with col4:
                fig2, ax4 = plt.subplots(figsize=(10, 8))
                sns.boxplot(data=df_apt, x="nombre_pieces_principales", color='darkblue')
                plt.title("Pieces Principales")
                ax4.set_xlabel('.')
                st.pyplot(fig2)
            with col5:
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                sns.boxplot(data=df_apt, x="surface_reelle_bati", color='red')
                plt.title("Surface")
                ax3.set_xlabel('.')
                st.pyplot(fig3)
        if type == 'Maison':
            col6, col7, col8= st.columns(3)
            with col6:
                local = ['Appartement', 'Maison', 'Local industriel. commercial ou assimilé']
                data = [439163, 202338, 65161]
                colors = ['white', 'red', 'white']
                explode = [0.1, 0, 0]
                fig4, ax = plt.subplots(figsize=(2, 2))
                plt.pie(data, explode=explode, colors=colors)
                st.pyplot(fig4)
            with col7:
                fig6, ax6 = plt.subplots(figsize=(10, 8))
                sns.boxplot(data=df_mai, x="nombre_pieces_principales", color='darkblue')
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

if choose == "Prediction":  
    col3, col4= st.columns(2)
    image = Image.open("C:/Users/luisa/OneDrive/Ambiente de Trabalho/hackathon/eiffel-tower-4582649_1280.webp")
    with col3:
        st.image(image)
    with col4:
        st.header('Prediction')
        
if choose == "Machine Learning":  
    col3, col4= st.columns(2)
    image = Image.open("C:/Users/luisa/OneDrive/Ambiente de Trabalho/hackathon/eiffel-tower-4582649_1280.webp")
    with col3:
        st.image(image)
    with col4:
        st.header('Machine Learning')