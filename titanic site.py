#!/usr/bin/env python
# coding: utf-8

# In[37]:


# Importeer nodige libraries
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

