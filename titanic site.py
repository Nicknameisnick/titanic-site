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
# Matplotlib en Seaborn zijn niet meer nodig
# import matplotlib.pyplot as plt 
# import seaborn as sns


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
    
    @st.cache_data
    def load_data():
        # Let op: De standaard Titanic dataset gebruikt een komma (,) als separator.
        # Als jouw bestand echt een puntkomma (;) gebruikt, laat dan sep=';' staan.
        df = pd.read_csv("train.csv")
        return df

    df = load_data()
    
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

        # Functie om de heatmap te plotten met Plotly
        def plot_missing_data_heatmap(dataset, title):
            fig = px.imshow(dataset.isnull(), title=title)
            fig.update_layout(width=500, height=500, yaxis_title="Rij Index", xaxis_title="Kolom")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("1. Visualisatie van missende data")
        st.write("We beginnen met een heatmap om te zien waar data ontbreekt.")
        plot_missing_data_heatmap(df_cleaned, "Heatmap van missende data (Origineel)")

        # --- NIEUWE SECTIE VOOR LEEFTIJD ---
        st.subheader("2. Onrealistische leeftijden corrigeren (Outliers)")
        st.write(
            "Volgens onderzoek was de oudste persoon aan boord van de Titanic 74 jaar oud ([bron](https://www.encyclopedia-titanica.org/titanic-oldest-on-board/)). "
            "Leeftijden hoger dan 74 in de dataset beschouwen we als datafouten en vervangen we door de mediaan."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Voor de correctie:**")
            fig_age_before = px.box(df_cleaned, y='Age', title='Leeftijdsverdeling (Origineel)')
            st.plotly_chart(fig_age_before, use_container_width=True)

        with col2:
            st.write("**Na de correctie:**")
            # Voer de correctie uit
            age_median = df_cleaned['Age'].median()
            df_cleaned.loc[df_cleaned['Age'] > 74, 'Age'] = age_median
            
            fig_age_after = px.box(df_cleaned, y='Age', title='Leeftijdsverdeling (Gecorrigeerd)')
            st.plotly_chart(fig_age_after, use_container_width=True)

        # --- NIEUWE SECTIE VOOR TICKETPRIJS ---
        st.subheader("3. Onrealistische ticketprijzen corrigeren (Outliers)")
        st.write(
            "De duurste ticketprijs was £870 voor een First Class Suite ([bron](https://www.cruisemummy.co.uk/titanic-ticket-prices/)). "
            "Waardes in de 'Fare'-kolom die significant hoger zijn, behandelen we als fouten en vervangen we door de mediaan."
        )

        col3, col4 = st.columns(2)

        with col3:
            st.write("**Voor de correctie:**")
            fig_fare_before = px.box(df_cleaned, y='Fare', title='Ticketprijsverdeling (Origineel)')
            st.plotly_chart(fig_fare_before, use_container_width=True)
            
        with col4:
            st.write("**Na de correctie:**")
            # Voer de correctie uit
            fare_median = df_cleaned['Fare'].median()
            # De bron is in ponden, de data waarschijnlijk in dollars, maar we gebruiken 870 als bovengrens.
            df_cleaned.loc[df_cleaned['Fare'] > 870, 'Fare'] = fare_median

            fig_fare_after = px.box(df_cleaned, y='Fare', title='Ticketprijsverdeling (Gecorrigeerd)')
            st.plotly_chart(fig_fare_after, use_container_width=True)


        st.subheader("4. Missende numerieke waarden opvullen")
        st.write("Nu vullen we de resterende lege plekken in de `Age` en `Fare` kolommen op met de mediaan.")
        st.info("De mediaan is een robuuste keuze omdat deze niet beïnvloed wordt door de extreme uitschieters die we zojuist hebben behandeld.")
        
        # Voer de code uit om missende waarden op te vullen
        df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)
        df_cleaned['Fare'].fillna(df_cleaned['Fare'].median(), inplace=True)
        st.success("Missende waarden in 'Age' en 'Fare' zijn opgevuld.")
        

        st.subheader("5. Onnodige kolommen verwijderen")
        st.write("Kolommen zoals 'Name', 'Ticket', en 'PassengerId' zijn uniek voor elke passagier en hebben geen voorspellende waarde. 'Cabin' heeft te veel missende data. Deze verwijderen we.")
        
        # Voer de code uit
        cols_to_drop = ['Ticket', 'Cabin', 'Name', 'PassengerId']
        df_cleaned.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        st.write("Het dataframe na het verwijderen van kolommen:")
        st.dataframe(df_cleaned.head())


        st.subheader("6. Eindresultaat na opschoning")
        st.write("Dit is de status van onze data na alle opschoningsstappen. De enige overgebleven missende waarden zitten in 'Embarked', die we later zullen aanpakken.")
        plot_missing_data_heatmap(df_cleaned, "Heatmap van missende data (Na opschoning)")


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
