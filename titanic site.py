#!/usr/bin/env python
# coding: utf-8

# In[37]:
import pandas as pd
import requests
import plotly.express as px 
import folium
import numpy as np 
import streamlit as st 
from sklearn.ensemble import RandomForestRegressor
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Titanic Dashboard 🚢", layout="wide")

st.sidebar.title("🚢 Titanic Navigatie")
pagina = st.sidebar.radio(
    "Kies een onderdeel:",
    [
        "Titanic case verbetering 🚢",
        "Titanic case 1e poging🚢",
        "Titanic case verbetering (2e poging)🚢"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Gebruik het menu om te navigeren tussen de onderdelen.")

if pagina == "Titanic case verbetering 🚢":
    st.title("Titanic case verbetering 🚢")
    st.write("")

elif pagina == "Titanic case 1e poging🚢":
    st.title("Titanic case 1e poging🚢")
    st.write("")

elif pagina == "Titanic case verbetering (2e poging)🚢":
    st.title("Titanic case verbetering (2e poging)🚢")
    st.write("")


