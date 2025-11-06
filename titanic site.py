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
        "Titanic case intro",
        "Titanic case 1e poging",
        "Titanic case verbetering (2e poging)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Gebruik het menu om te navigeren tussen de onderdelen.")

if pagina == "Titanic case intro":
    st.title("Titanic case intro")
    st.markdown("### 👥 Team 1 — Matthijs de Wolff & Wessel IJskamp")

    st.header("Inleiding")
    st.write(
        "In deze presentatie gaan wij toelichten hoe we de eerste keer de Titanic case "
        "hebben uitgevoerd en hoe wij de case daarna hebben verbeterd."
    )

    st.subheader("🔬 Methode")
    st.write(
        "In de eerste case gebruikten wij enkel een set van variabelen om een voorspelling te doen, "
        "en in de verbeterpoging hebben wij een machine learning model gebruikt."
    )



elif pagina == "Titanic case 1e poging":
    st.title("Titanic case 1e poging")

    st.write("In de eerste poging hebben wij een voorspelling gemaakt op basis van een paar variabelen, er is hier geen gebruik gemaakt van een machine learning model")

    st.subheader("Variabelen gebruikt voor de eerste poging:")
    st.image("1e poging train set.png")

    st.subheader("Resultaat")
    st.image("submission 1e poging.png")
    st.write("Het resultaat van de eerste poging kwam uit op 78,2%")

    st.subheader("Discussie")
    st.write("""
    ### ✅ Waarom dit goed is
    - **Eenvoudige logica, goed resultaat:** zonder machine learning al bijna 80%.
    - **Goede baseline:** dit geeft een referentiepunt voor toekomstige modellen.
    - **Makkelijk te verklaren:** “Vrouwen en kinderen eerst” komt overeen met de historische realiteit.
    
    ### ❌ Waarom dit slecht is
    - **Te simpel:** geen gebruik van andere variabelen zoals klasse of ticketprijs.
    - **Geen nuance:** sommige mannen overleefden wel, sommige vrouwen niet.
    - **Geen leeralgoritme:** het model leert niets uit de data.
    """)


elif pagina == "Titanic case verbetering (2e poging)":
    st.title("Titanic case verbetering (2e poging)")
    st.write("")













