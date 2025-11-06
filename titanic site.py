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

st.set_page_config(page_title="Titanic case verbetering 🚢", layout="wide")
st.title("Titanic case verbetering 🚢")

# Tabs aanmaken
tab1, tab2, tab3 = st.tabs(["Titanic case verbetering intro🚢", "Titanic case 1e poging🚢", "Titanic case verbetering (2e poging)🚢"])
