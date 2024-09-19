import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from joblib import dump

# Chargement des fichiers
merged_df = pd.read_csv('merged_df.csv')
weather_df = pd.read_parquet('weather.parquet')

# Conversion des dates
weather_df['apply_time_rl'] = pd.to_datetime(weather_df['apply_time_rl'], unit='s')
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')

# Fusion des données météo et des résultats
merged_df = merged_df.merge(
    weather_df,
    left_on='date',
    right_on='apply_time_rl',
    how='left'
)

# Préparation des features pour la prédiction
X = merged_df[['grid', 'fastestLapTime', 'points', 'q1', 'climate_temperature', 'climate_pressure',
               'circuitId', 'gfs_wind_speed', 'gfs_precipitations', 'driverId']].copy()

# Conversion des colonnes q1 et fastestLapTime en numérique
X['q1'] = pd.to_numeric(X['q1'].str.replace(':', ''), errors='coerce')
X['fastestLapTime'] = pd.to_numeric(X['fastestLapTime'].str.replace(':', ''), errors='coerce')

# Imputation des valeurs manquantes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Variable cible (position finale dans la course)
y = merged_df['positionOrder']

# Séparation des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Sauvegarde du modèle et de l'imputer
dump(model, 'race_position_predictor_model.joblib')
dump(imputer, 'imputer.joblib')

print("Le modèle a été entraîné et sauvegardé avec succès pour prédire le classement complet.")