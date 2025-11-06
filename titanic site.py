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
import pydeck as pdk
import time

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
    # Teaminfo in een opvallend blok
    st.markdown("""
    ### 👥 **Team 1 — Matthijs de Wolff & Wessel IJskamp**
    
    Welkom bij onze presentatie over de Titanic case verbetering.  
    """)
    
    # Visuele scheiding (horizontale lijn)
    st.markdown("---")
    
    # Inleidingstekst
    st.subheader("📘 Inleiding")
    st.write("""
    De Titanic-case is een klassiek data science-project waarin we voorspellen wie de ramp overleefde, 
    op basis van kenmerken zoals geslacht, leeftijd en klasse.
    
    In deze presentatie gaan wij toelichten hoe wij de **eerste versie** van de Titanic case hebben uitgevoerd  
    en hoe we deze daarna hebben verbeterd.  
    
    We laten zien welke keuzes we maakten bij het opstellen van voorspellingen en wat we daarvan hebben geleerd.
    """)

    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])  # verhouding: iets meer ruimte voor tekst (links)

    # Linkerkolom — tekstblok
    with col1:
        st.success("""
        ### 🎯 Doel van de case
        Het doel van deze case is om inzicht te krijgen in:
        
        - Hoe simpele regels al een sterke voorspelling kunnen geven  
        - Hoe data-analyse kan helpen bij het verbeteren van modellen  
        - Hoe machine learning hierbij een volgende stap vormt
        """)
    
    # Rechterkolom — afbeelding
    with col2:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg",
            caption="RMS Titanic (1912)",
            width=550  # pas aan voor gewenste grootte
        )

    data = [
    {"step": 0, "lat": 51.7167, "lon": -8.2667, "event": "Vertrek Queenstown"},
    {"step": 1, "lat": 50.1067, "lon": -20.7167, "event": "Middag 12 Apr"},
    {"step": 2, "lat": 47.3667, "lon": -33.1667, "event": "Middag 13 Apr"},
    {"step": 3, "lat": 43.0283, "lon": -44.5233, "event": "Middag 14 Apr"},
    {"step": 4, "lat": 41.7667, "lon": -50.2333, "event": "Crash ijsberg"}
    ]
    df = pd.DataFrame(data)
    
    st.title("🚢 Titanic Journey Tracker")
    
    # Slider voor handmatige selectie
    step = st.slider("Selecteer het punt van de reis", 0, len(df)-1, 0)
    
    # Placeholder voor de kaart
    map_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Functie om kaart te tekenen
    def draw_map(current_step):
        map_placeholder.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=df.loc[current_step, 'lat'],
                longitude=df.loc[current_step, 'lon'],
                zoom=4,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df.iloc[:current_step+1],
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=200000,
                ),
                pdk.Layer(
                    "LineLayer",
                    data=df.iloc[:current_step+1],
                    get_source_position='[lon, lat]',
                    get_target_position='[lon, lat]',
                    get_color='[0, 0, 255]',
                    get_width=5,
                )
            ]
        ))
        status_placeholder.write(f"📍 **Huidige status:** {df.loc[current_step, 'event']}")
    
    # Knop voor animatie
    if st.button("▶️ Start animatie"):
        for i in range(len(df)):
            draw_map(i)
            time.sleep(1)
    else:
        draw_map(step)

elif pagina == "Titanic case 1e poging":
    st.title("**Titanic case 1e poging**")

    st.write("In de eerste poging hebben wij een voorspelling gemaakt op basis van een paar kenmerken, er is hier geen gebruik gemaakt van een machine learning model")

    st.subheader("**Kenmerken gebruikt voor de eerste poging**:")
    st.code("""
        train_pred = np.where(
            (train['Sex'] == 'female') | ((train['Sex'] == 'male') & (train['Age'] < 10)),
            1, 0
        )
        """)

    st.subheader("**Resultaat**")
    st.image("submission 1e poging.png")
    st.write("Het resultaat van de eerste poging kwam uit op 78,2% accuraatheid")

    st.markdown("---")
    
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

    st.markdown("---")
    
    st.subheader("**Feedback op eerste poging:**")
    st.write("""
    ### Punten:
    - **ML-model:** Een ML model zou fijn geweest zijn.
    - **Scatterplot:** Er moet een scatterplot in voor een uitgebreidere visualisatie.
    """)


elif pagina == "Titanic case verbetering (2e poging)":
    st.title("Titanic case verbetering (2e poging)")
    
    # Load data
    @st.cache_data
    def load_data():
        # Assuming the CSV is comma-separated as is standard. If it's truly semicolon, change back to sep=";".
        df = pd.read_csv("train.csv")
        return df

    df = load_data()
    
    # Create a copy for the cleaning tab to not affect other tabs
    df_cleaned = df.copy()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Data opschoning", 
        "2. De data", 
        "3. Feature engineering", 
        "4. Algoritmes", 
        "5. Conclusies en eindscore"
    ])

    with tab1:
        st.header("Data opschoning")

        st.subheader("1. Visualisatie van missende data")
        st.write("Eerst kijken we met een histogram hoeveel data er per kolom ontbreekt.")
        missing_data = df_cleaned.isnull().sum().reset_index()
        missing_data.columns = ['Kolom', 'Aantal missende waardes']
        missing_data_filtered = missing_data[missing_data['Aantal missende waardes'] > 0]
        
        fig_missing = px.bar(
            missing_data_filtered,
            x='Kolom',
            y='Aantal missende waardes',
            title='Aantal missende waardes per kolom'
        )
        st.plotly_chart(fig_missing, use_container_width=True)

        st.subheader("2. Kolommen verwijderen")
        st.write("Sommige kolommen zijn niet nuttig voor ons model of bevatten te veel missende waarden. We verwijderen 'Ticket', 'Cabin', 'Name', en 'PassengerId'.")
        st.code("""
# 'Cabin' heeft te veel missende waarden om bruikbaar te zijn.
# 'Ticket', 'Name', en 'PassengerId' zijn unieke identifiers die geen voorspellende waarde hebben voor een machine learning model.
df_cleaned.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
        """, language='python')

        # Execute the code
        df_cleaned.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
        st.write("Het dataframe nadat de kolommen zijn verwijderd:")
        st.dataframe(df_cleaned.head(), use_container_width=True)

        st.subheader("3. Missende numerieke waarden opvullen")
        st.write("Voor de numerieke kolommen `Age`, `Fare`, `SibSp` en `Parch` vullen we eventuele lege plekken op met de **mediaan** van de betreffende kolom.")
        st.info("We hebben op internetbronnen onderzocht wat de beste aanpak is en daaruit bleek dat het opvullen van missende waarden met de mediaan een robuuste methode is, omdat deze minder gevoelig is voor uitschieters (outliers) dan het gemiddelde.")
        
        st.code("""
# Vul missende waarden in 'Age' met de mediaan van 'Age'
df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)

# Doe hetzelfde voor Fare, SibSp, en Parch voor het geval er missende waarden zijn.
df_cleaned['Fare'].fillna(df_cleaned['Fare'].median(), inplace=True)
df_cleaned['SibSp'].fillna(df_cleaned['SibSp'].median(), inplace=True)
df_cleaned['Parch'].fillna(df_cleaned['Parch'].median(), inplace=True)
        """, language='python')

        # Execute the code to fill missing values
        df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)
        df_cleaned['Fare'].fillna(df_cleaned['Fare'].median(), inplace=True)
        df_cleaned['SibSp'].fillna(df_cleaned['SibSp'].median(), inplace=True)
        df_cleaned['Parch'].fillna(df_cleaned['Parch'].median(), inplace=True)

        st.subheader("4. Resultaat na opschoning")
        st.write("Nadat we de missende waarden hebben opgevuld, controleren we opnieuw hoeveel er nog over zijn.")
        missing_data_after = df_cleaned.isnull().sum().reset_index()
        missing_data_after.columns = ['Kolom', 'Aantal missende waardes']
        st.dataframe(missing_data_after, use_container_width=True)
        st.success("De missende waarden in de 'Age' kolom zijn succesvol opgevuld met de mediaan.")

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

























