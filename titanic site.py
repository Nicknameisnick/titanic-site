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

train_df = pd.read_csv("train.csv", sep=";")
test_df = pd.read_csv("test.csv", sep=",")

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
    st.title("**Titanic case 1e poging**")

    st.write("In de eerste poging hebben wij een voorspelling gemaakt op basis van een paar variabelen, er is hier geen gebruik gemaakt van een machine learning model")

    st.subheader("**Variabelen gebruikt voor de eerste poging**:")
    st.image("1e poging train set.png")

    st.subheader("**Resultaat**")
    st.image("submission 1e poging.png")
    st.write("Het resultaat van de eerste poging kwam uit op 78,2%")

    st.subheader("**Discussie**")
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
    
    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv('train.csv')
        return df

    df = load_data()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Data opschoning", 
        "2. De data", 
        "3. Feature engineering", 
        "4. Algoritmes", 
        "5. Conclusies en eindscore"
    ])

    with tab1:
        st.header("Data opschoning")
        st.write("Per kolom wordt hier het aantal missende waardes weergegeven.")
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Kolom', 'Aantal missende waardes']
        st.dataframe(missing_data, use_container_width=True)

    with tab2:
        st.header("De data")

        # Survival count
        st.subheader("Hoeveel mensen hebben het overleefd?")
        survival_counts = df['Survived'].value_counts().reset_index()
        survival_counts.columns = ['Status', 'Aantal']
        survival_counts['Status'] = survival_counts['Status'].map({0: 'Niet overleefd', 1: 'Overleefd'})
        fig_survival = px.bar(survival_counts, x='Status', y='Aantal', title='Totaal aantal overlevenden')
        st.plotly_chart(fig_survival, use_container_width=True)

        # Survival by gender
        st.subheader("Hoeveel mensen hebben het overleefd per geslacht?")
        survival_gender = df.groupby(['Sex', 'Survived']).size().reset_index(name='Aantal')
        survival_gender['Survived'] = survival_gender['Survived'].map({0: 'Niet overleefd', 1: 'Overleefd'})
        survival_gender['Sex'] = survival_gender['Sex'].map({'male': 'Man', 'female': 'Vrouw'})
        fig_gender = px.bar(survival_gender, x='Sex', y='Aantal', color='Survived', barmode='group', title='Overleving per geslacht')
        st.plotly_chart(fig_gender, use_container_width=True)

        # Survival by Pclass
        st.subheader("Hoeveel mensen hebben het overleefd per Pclass?")
        survival_pclass = df.groupby(['Pclass', 'Survived']).size().reset_index(name='Aantal')
        survival_pclass['Survived'] = survival_pclass['Survived'].map({0: 'Niet overleefd', 1: 'Overleefd'})
        fig_pclass = px.bar(survival_pclass, x='Pclass', y='Aantal', color='Survived', barmode='group', title='Overleving per Pclass')
        st.plotly_chart(fig_pclass, use_container_width=True)

        # Age distribution
        st.subheader("Distributie van leeftijd")
        fig_age_dist = px.histogram(df, x='Age', nbins=50, title='Distributie van leeftijd')
        st.plotly_chart(fig_age_dist, use_container_width=True)

        # KDE plots of age for survived and not survived
        st.subheader("KDE plots van leeftijd voor overlevenden en niet-overlevenden")
        fig_kde = go.Figure()
        fig_kde.add_trace(go.Violin(x=df['Survived'][df['Survived']==1], y=df['Age'][df['Survived']==1],
                                   legendgroup='Yes', scalegroup='Yes', name='Overleefd',
                                   side='positive', line_color='blue'))
        fig_kde.add_trace(go.Violin(x=df['Survived'][df['Survived']==0], y=df['Age'][df['Survived']==0],
                                   legendgroup='No', scalegroup='No', name='Niet overleefd',
                                   side='negative', line_color='orange'))
        fig_kde.update_traces(meanline_visible=True)
        fig_kde.update_layout(violingap=0, violinmode='overlay', title="Leeftijdsverdeling van overlevenden versus niet-overlevenden")
        st.plotly_chart(fig_kde, use_container_width=True)

        # Influence of ticket price on survival
        st.subheader("Heeft de ticketprijs de overlevingskans beïnvloed?")
        st.write("Verdeling van de ticketprijzen")
        fig_fare_dist = px.histogram(df, x='Fare', nbins=50, title='Distributie van ticketprijzen')
        st.plotly_chart(fig_fare_dist, use_container_width=True)

        st.write("Overlevingskans per prijscategorie")
        df['FareCategory'] = pd.qcut(df['Fare'], 4, labels=['Laag', 'Gemiddeld', 'Hoog', 'Zeer hoog'])
        fare_survival = df.groupby('FareCategory')['Survived'].mean().reset_index()
        fig_fare_survival = px.bar(fare_survival, x='FareCategory', y='Survived', title='Overlevingskans per prijscategorie')
        st.plotly_chart(fig_fare_survival, use_container_width=True)

        # Key observations plot
        st.subheader("Belangrijke observaties")
        st.markdown("""
        - Ongeacht het geslacht overleefden alle passagiers met een ticketprijs van meer dan $500.
        - Alle mannelijke passagiers die tussen de $200 en $300 betaalden, zijn overleden
        - Alle vrouwelijke passagiers die tussen de $200 en $300 betaalden, hebben het overleefd
        """)
        fare_gender_survival = df.groupby(['Fare', 'Sex'])['Survived'].mean().unstack()
        st.line_chart(fare_gender_survival)

        # Embarked location and survival chance
        st.subheader("Vergelijking van opstapplaats en overlevingskans")
        embarked_survival = df.groupby('Embarked')['Survived'].mean().reset_index()
        fig_embarked_survival = px.bar(embarked_survival, x='Embarked', y='Survived', title='Overlevingskans per opstapplaats')
        st.plotly_chart(fig_embarked_survival, use_container_width=True)

        # High number of survivors from Cherbourg
        st.subheader("Was het hoge aantal overlevenden dat in Cherbourg aan boord ging te wijten aan een hoog aantal 1e klas passagiers?")
        embarked_pclass = df.groupby(['Embarked', 'Pclass']).size().reset_index(name='Aantal')
        fig_embarked_pclass = px.bar(embarked_pclass, x='Embarked', y='Aantal', color='Pclass', barmode='group', title='Verdeling van Pclass per opstapplaats')
        st.plotly_chart(fig_embarked_pclass, use_container_width=True)
        st.write(
            "Conclusie: Ja, het is waarschijnlijk dat het hoge aantal overlevenden uit Cherbourg te wijten is aan het "
            "grotere aandeel 1e klas passagiers dat daar aan boord ging in vergelijking met de andere opstapplaatsen."
        )

    with tab3:
        st.header("Feature engineering")
        st.write("Informatie over feature engineering.")

    with tab4:
        st.header("Algoritmes")
        st.write("Informatie over de gebruikte algoritmes.")

    with tab5:
        st.header("Conclusies en eindscore")
        st.write("Conclusies en de eindscore van het model.")
















