#import libraries
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data Readers_app", layout="wide", menu_items=None)
st.title('Hello')

pd.read_csv('df_dummies2.csv')
st.table(df_dummies2.head())

