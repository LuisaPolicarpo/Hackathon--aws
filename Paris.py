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

st.set_page_config(page_title="Datathon", page_icon="🗼", layout="wide", menu_items=None)

    
with st.sidebar:
    choose = option_menu(None, ["The city", "EDA", "Prediction", "Machine Learning"],
                         icons=['house', 'bi bi-calendar3', 'kanban', 'bi bi-geo-alt'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#E5E5EA"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#F91A00"},
        "nav-link-selected": {"background-color": "#0900F9"},
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
    col3, col4= st.columns(2)
    image = Image.open("C:/Users/luisa/OneDrive/Ambiente de Trabalho/hackathon/eiffel-tower-4582649_1280.webp")
    with col3:
        st.image(image)
    with col4:
        st.header('EDA')
        
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