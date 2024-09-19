import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import random
import faker

# Chargement du modèle et de l'imputer
model = load('race_position_predictor_model.joblib')
imputer = load('imputer.joblib')

# Chargement des données pour obtenir les IDs des pilotes
merged_df = pd.read_csv('merged_df.csv')

# Liste des IDs des pilotes (assurez-vous que ces IDs existent dans votre modèle)
driver_ids = merged_df['driverId'].unique()

# Fonction pour générer des données de course aléatoires
def generate_random_data(driver_ids):
    fake = faker.Faker()
    random_data = []
    for _ in range(len(driver_ids)):  # Générer des données pour chaque pilote
        data = {
            'grid': random.randint(1, 20),
            'fastestLapTime': random.uniform(60, 120),  # en secondes
            'points': random.randint(0, 25),
            'q1': f"{random.randint(0, 2):02}:{random.randint(0, 59):02}:{random.randint(0, 59):02}",
            'climate_temperature': random.uniform(-10, 40),  # en °C
            'climate_pressure': random.uniform(950, 1050),  # en hPa
            'circuitId': random.randint(1, 30),
            'gfs_wind_speed': random.uniform(0, 30),  # en km/h
            'gfs_precipitations': random.uniform(0, 50),  # en mm
            'driverId': random.choice(driver_ids)  # Choisir un ID pilote au hasard
        }
        random_data.append(data)
    return random_data

# Fonction pour prédire le résultat d'une course
def predict_race_results(data):
    # Préparer les données d'entrée
    df = pd.DataFrame(data)
    
    # Assurer que les colonnes sont des chaînes de caractères pour les conversions
    df['q1'] = df['q1'].astype(str)
    df['fastestLapTime'] = df['fastestLapTime'].astype(str)
    
    # Conversion des colonnes q1 et fastestLapTime en numérique
    df['q1'] = pd.to_numeric(df['q1'].str.replace(':', ''), errors='coerce')
    df['fastestLapTime'] = pd.to_numeric(df['fastestLapTime'].str.replace(':', ''), errors='coerce')
    
    # Imputer les valeurs manquantes
    X_new = imputer.transform(df)
    
    # Prédire les résultats
    predictions = model.predict(X_new)
    
    # Ajouter les prédictions aux données
    df['Predicted Position'] = predictions
    return df.sort_values(by='Predicted Position')

# Interface utilisateur avec Streamlit
st.title('Prédiction des Résultats de Course F1')

# Génération et prédiction des données de course aléatoires
if st.button('Générer une course et prédire les résultats'):
    random_data = generate_random_data(driver_ids)
    predicted_results_df = predict_race_results(random_data)
    
    st.write("Données de course générées et résultats prédits :")
    st.write(predicted_results_df)
