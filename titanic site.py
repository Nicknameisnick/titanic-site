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
from scipy.stats import gaussian_kde
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
        
    st.markdown("---")
    
    data = [
        {"step": 0, "lat": 51.7167, "lon": -8.2667, "event": "Vertrek Queenstown"},
        {"step": 1, "lat": 50.1067, "lon": -20.7167, "event": "Middag 12 Apr"},
        {"step": 2, "lat": 47.3667, "lon": -33.1667, "event": "Middag 13 Apr"},
        {"step": 3, "lat": 43.0283, "lon": -44.5233, "event": "Middag 14 Apr"},
        {"step": 4, "lat": 41.7667, "lon": -50.2333, "event": "Crash ijsberg"}
    ]
    df = pd.DataFrame(data)
    
    st.title("🚢 Titanic Reis")
    
    # Slider eerst
    step = st.slider("Selecteer het punt van de reis", 0, len(df)-1, 0)
    
    # Dan de animatie-knop
    animate = st.button("▶️ Start animatie")
    
    # Placeholder voor kaart en status
    map_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Functie om kaart te tekenen
    def draw_map(current_step):
        map_placeholder.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=df.loc[current_step, 'lat'],
                longitude=df.loc[current_step, 'lon'],
                zoom=2,
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

    # Animatie logica
    if animate:
        n_steps = len(df)
        extra_time = 2  # extra seconden totaal
        sleep_time = 1 + extra_time / n_steps  # 1 sec standaard + verdeling van extra tijd
    
        for i in range(n_steps):
            draw_map(i)
            time.sleep(sleep_time)

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
    st.write("""
    Bij de eerste poging is EDA toegepast zonder ML, hier zitten echter een aantal beperkingen aan:
    - Interactie tussen features wordt moeilijk zichtbaar. 
        - Bijvoorbeeld: Mannen in 1e klas hebben een hogere overlevingskans dan mannen in 3e klas.
        - Eenvoudige grafieken of gemiddelden missen zulke combinaties.
    - Niet-lineaire patronen
        - Overleving kan afhangen van een combinatie van leeftijd, klasse, en geslacht.
        - EDA kan dit vaak niet goed vatten zonder uitgebreide, complexe plots.
    - Geen automatische gewichten
        - EDA vertelt je “er lijkt een verband te zijn”, maar je weet niet hoe belangrijk elke feature is voor voorspellingen.
    
    Resultaat: je krijgt intuïtie, maar nog geen krachtige voorspeller.
    """)
    st.markdown("---")

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
    - **Bronnen:** Er moeten bronnen weergeven worden
    """)


elif pagina == "Titanic case verbetering (2e poging)":
    st.title("Titanic case verbetering (2e poging)")
    
    # Load data
    @st.cache_data
    def load_data():
        # Gebruik de standaard comma-separator, wat het meest gebruikelijk is voor .csv-bestanden.
        df = pd.read_csv("train.csv", sep=';')
        return df

    df = load_data()
    
    # Maak een kopie voor de opschoning-tab om de originele data niet te beïnvloeden
    df_cleaned = df.copy()

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Data opschoning", 
        "2. De data", 
        "3. ML-model",  
        "4. Conclusies en eindscore"
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

        st.subheader("2. Missende numerieke waarden opvullen")
        st.write("Nu vullen we de resterende lege plekken in de `Age`, `cabin` en `Fare` kolommen op met de mediaan.")
        st.info("De mediaan is een robuuste keuze omdat deze niet beïnvloed wordt door de extreme uitschieters die we zojuist hebben behandeld.")
        
        # Voer de code uit om missende waarden op te vullen
        df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)
        df_cleaned['Fare'].fillna(df_cleaned['Fare'].median(), inplace=True)
    
        
                
        st.subheader("3. Eindresultaat na opschoning")
        st.write("Dit is de status van onze data na alle opschoningsstappen. De enige overgebleven missende waarden zitten in 'cabin', die we later zullen aanpakken.")
        plot_missing_data_heatmap(df_cleaned, "Heatmap van missende data (Na opschoning)")
     

        # --- NIEUWE SECTIE VOOR LEEFTIJD ---
        st.subheader("4. Onrealistische leeftijden corrigeren (Outliers)")
        st.write(
            "Volgens onderzoek was de oudste persoon aan boord van de Titanic 74 jaar oud ([bron](https://www.encyclopedia-titanica.org/titanic-oldest-on-board/)). "
            "Leeftijden hoger dan 74 in de dataset beschouwen we als datafouten en vervangen we door de mediaan.")
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
        st.subheader("5. Onrealistische ticketprijzen corrigeren (Outliers)")
        st.write(
            "De duurste ticketprijs was £870.0 voor een First Class Suite ([bron](https://www.cruisemummy.co.uk/titanic-ticket-prices/)). "
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
            df_cleaned.loc[df_cleaned['Fare'] > 8700, 'Fare'] = fare_median

            fig_fare_after = px.box(df_cleaned, y='Fare', title='Ticketprijsverdeling (Gecorrigeerd)')
            st.plotly_chart(fig_fare_after, use_container_width=True)


        
        

        st.subheader("6. Onnodige kolommen verwijderen")
        st.write("Kolommen zoals 'Name', 'Ticket', en 'PassengerId' zijn uniek voor elke passagier en hebben geen voorspellende waarde. 'Cabin' heeft te veel missende data. Deze verwijderen we.")
        
        # Voer de code uit
        cols_to_drop = ['Ticket', 'Cabin', 'Name', 'PassengerId']
        df_cleaned.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        st.write("Het dataframe na het verwijderen van kolommen:")
        st.dataframe(df_cleaned.head())


        



    with tab2:
        st.header("De data")

        # Survival count in percentage
        st.subheader("Hoeveel procent van de mensen heeft het overleefd?")
        survival_perc = df_cleaned['Survived'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
        survival_perc['Status'] = survival_perc['Survived'].map({0: 'Niet overleefd', 1: 'Overleefd'})
        
        fig_survival = px.bar(
            survival_perc, 
            x='Status', 
            y='Percentage', 
            title='Totaal percentage overlevenden',
            text=survival_perc['Percentage'].apply(lambda x: f'{x:.1f}%'),
            color='Status',
            color_discrete_map={'Overleefd': 'lightgreen', 'Niet overleefd': 'lightcoral'}
        )
        fig_survival.update_layout(showlegend=False)
        # Maak de tekst groter
        fig_survival.update_traces(textfont_size=16)
        st.plotly_chart(fig_survival, use_container_width=True)

        # Survival by gender with percentages on bars
        st.subheader("Hoeveel mensen hebben het overleefd per geslacht?")
        survival_gender = df_cleaned.groupby(['Sex', 'Survived']).size().reset_index(name='Aantal')
        survival_gender['Percentage'] = survival_gender['Aantal'] / survival_gender.groupby('Sex')['Aantal'].transform('sum') * 100
        survival_gender['Survived'] = survival_gender['Survived'].map({0: 'Niet overleefd', 1: 'Overleefd'})
        survival_gender['Sex'] = survival_gender['Sex'].map({'male': 'Man', 'female': 'Vrouw'})
        
        fig_gender = px.bar(
            survival_gender, 
            x='Sex', 
            y='Aantal', 
            color='Survived', 
            barmode='group', 
            title='Overleving per geslacht (met percentages)',
            text=survival_gender['Percentage'].apply(lambda x: f'{x:.1f}%')
        )
        # Maak de tekst groter
        fig_gender.update_traces(textfont_size=14)
        st.plotly_chart(fig_gender, use_container_width=True)

        # Survival by Pclass with percentages on bars
        st.subheader("Hoeveel mensen hebben het overleefd per Pclass?")
        survival_pclass = df_cleaned.groupby(['Pclass', 'Survived']).size().reset_index(name='Aantal')
        survival_pclass['Percentage'] = survival_pclass['Aantal'] / survival_pclass.groupby('Pclass')['Aantal'].transform('sum') * 100
        survival_pclass['Survived'] = survival_pclass['Survived'].map({0: 'Niet overleefd', 1: 'Overleefd'})
        
        fig_pclass = px.bar(
            survival_pclass, 
            x='Pclass', 
            y='Aantal', 
            color='Survived', 
            barmode='group', 
            title='Overleving per Pclass (met percentages)',
            text=survival_pclass['Percentage'].apply(lambda x: f'{x:.1f}%')
        )
        # Maak de tekst groter
        fig_pclass.update_traces(textfont_size=14)
        st.plotly_chart(fig_pclass, use_container_width=True)

        # NIEUW: KDE plot voor leeftijdsverdeling
        st.subheader("Leeftijdsverdeling: Overlevenden vs. Niet-overlevenden (KDE)")
        
        # Data voorbereiden
        age_survived = df_cleaned[df_cleaned['Survived'] == 1]['Age'].dropna()
        age_not_survived = df_cleaned[df_cleaned['Survived'] == 0]['Age'].dropna()
        
        # Creëer de KDE
        kde_survived = gaussian_kde(age_survived)
        kde_not_survived = gaussian_kde(age_not_survived)
        
        # Maak een range voor de x-as
        age_range = np.linspace(df_cleaned['Age'].min(), df_cleaned['Age'].max(), 500)
        
        # Maak de figuur
        fig_age_kde = go.Figure()
        fig_age_kde.add_trace(go.Scatter(
            x=age_range, y=kde_survived(age_range), mode='lines', name='Overleefd', line=dict(color='green')
        ))
        fig_age_kde.add_trace(go.Scatter(
            x=age_range, y=kde_not_survived(age_range), mode='lines', name='Niet overleefd', line=dict(color='red')
        ))
        fig_age_kde.update_layout(
            title="Dichtheid van leeftijden: Overlevenden vs. Niet-overlevenden",
            xaxis_title="Leeftijd", yaxis_title="Dichtheid"
        )
        st.plotly_chart(fig_age_kde, use_container_width=True)

        # NIEUW: Overlevingskans per prijscategorie en geslacht
        st.subheader("Heeft de ticketprijs de overlevingskans beïnvloed?")
        
        st.write("Verdeling van de ticketprijzen")
        fig_fare_dist = px.histogram(df_cleaned, x='Fare', nbins=50, title='Distributie van ticketprijzen')
        st.plotly_chart(fig_fare_dist, use_container_width=True)

        st.write("Overlevingskans per prijscategorie en geslacht")
        # Maak bins van 1000 breed
        max_fare = int(df_cleaned['Fare'].max())
        bins = np.arange(0, max_fare + 1000, 1000)
        labels = [f'{i}-{i+1000-1}' for i in bins[:-1]]
        
        df_cleaned['FareBin'] = pd.cut(df_cleaned['Fare'], bins=bins, labels=labels, right=False)

        # Groepeer op prijscategorie en geslacht
        fare_gender_survival = df_cleaned.groupby(['FareBin', 'Sex'], observed=False)['Survived'].mean().reset_index()
        fare_gender_survival['Sex'] = fare_gender_survival['Sex'].map({'male': 'Man', 'female': 'Vrouw'})
        
        fig_fare_gender = px.bar(
            fare_gender_survival,
            x='FareBin',
            y='Survived',
            color='Sex',
            barmode='group',
            title='Overlevingskans per Prijscategorie en Geslacht'
        )
        fig_fare_gender.update_yaxes(title="Overlevingskans", tickformat=".0%")
        fig_fare_gender.update_xaxes(title="Ticketprijs (€)")
        st.plotly_chart(fig_fare_gender, use_container_width=True)


        # LIJNGRAFIEK IS VERWIJDERD
        st.subheader("Belangrijke observaties")
        st.markdown("""
        - Ongeacht het geslacht overleefden alle passagiers met een ticketprijs van meer dan $500.
        - Alle mannelijke passagiers die tussen de $200 en $300 betaalden, zijn overleden
        - Alle vrouwelijke passagiers die tussen de $200 en $300 betaalden, hebben het overleefd
        """)

        # Embarked location and survival chance
        st.subheader("Vergelijking van opstapplaats en overlevingskans")
        embarked_survival = df_cleaned.groupby('Embarked')['Survived'].mean().reset_index()
        fig_embarked_survival = px.bar(embarked_survival, x='Embarked', y='Survived', title='Overlevingskans per opstapplaats')
        fig_embarked_survival.update_yaxes(title="Overlevingskans", tickformat=".0%")
        st.plotly_chart(fig_embarked_survival, use_container_width=True)

        # High number of survivors from Cherbourg
        st.subheader("Was het hoge aantal overlevenden dat in Cherbourg aan boord ging te wijten aan een hoog aantal 1e klas passagiers?")
        embarked_pclass = df_cleaned.groupby(['Embarked', 'Pclass']).size().reset_index(name='Aantal')
        fig_embarked_pclass = px.bar(embarked_pclass, x='Embarked', y='Aantal', color='Pclass', barmode='group', title='Verdeling van Pclass per opstapplaats')
        st.plotly_chart(fig_embarked_pclass, use_container_width=True)
        st.write(
            "Conclusie: Ja, het is waarschijnlijk dat het hoge aantal overlevenden uit Cherbourg te wijten is aan het "
            "grotere aandeel 1e klas passagiers dat daar aan boord ging in vergelijking met de andere opstapplaatsen."
        )
    with tab3:
        st.header("ML-model")
        st.write("De code van het model")

        st.subheader("Data preprocessing functie")
        st.code("""def clean(url):
        data= pd.read_csv(url)
        
        ports = ["S", "C", "Q"]
        rep = data["Embarked"].mode()[0]
        for idx, embark in enumerate(data["Embarked"]):
            if embark != embark:
                data.loc[idx, "Embarked"] = rep
                embark = rep
            if embark in ports:
                data.loc[idx, "Embarked"] = ports.index(embark)
            else:
                ports.append(embark)
                data.loc[idx, "Embarked"] = ports.index(embark)
        data["Embarked"] = data["Embarked"].astype("int")
        
        
        for idx, gender in enumerate(data["Sex"]):
            if gender == "female":
                data.loc[idx, "Sex"] = 2
            else:
                data.loc[idx, "Sex"] = 1
        data["Sex"] = data["Sex"].astype("int")
    
        
        return data.drop("Name", axis=1).drop("Ticket", axis=1).drop("PassengerId", axis=1).drop("Cabin", axis=1)
    """)
        st.write("""
        - De data word ingelezen
        - De kolommen Embarked en Sex worden numeriek gemaakt voor het model
        - De irrelevante kolommen worden verwijderd
        """)
    with tab4:
        st.header("Conclusies en eindscore")
        st.write("Conclusies en de eindscore van het model.")



































