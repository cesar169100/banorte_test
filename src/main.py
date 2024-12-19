import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_absolute_error
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from xgboost import XGBRegressor
import optuna
import shap

# Especificar ruta al repo clonado
os.chdir("/home/cesar/Documentos/banorte_test")
# Lectura de la informacion
df_final = pd.read_csv('./data/df_final.csv')
df_final = df_final.drop(['Unnamed: 0'],axis=1)

# Split para complementar con un modelo supervisado
X = df_final.drop(['Rating'], axis=1)
y = df_final['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Tuneo de hiperparametros
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 25, 500),
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 5),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
#         'random_state': 42
#     }

#     # Crear el modelo XGBoost utilizando el diccionario de parámetros
#     model = XGBRegressor(booster='gbtree', **params)

#     # Validación cruzada con MAE como métrica
#     mae = -cross_val_score(
#         model, X_train, y_train, cv=10, 
#         scoring=make_scorer(mean_absolute_error, greater_is_better=False),
#         n_jobs=-1
#     ).mean()

#     # Retornar la métrica (MAE promedio)
#     return mae

# # Correr la optimización de hiperparámetros con Optuna
# study = optuna.create_study(direction='minimize')  # Minimizar el MAE
# study.optimize(objective, n_trials=10)
# # Imprimir los mejores parámetros y el mejor f1
# print("Mejores parámetros:", study.best_params)
# print("Mejor mae:", study.best_value)

# # Entrenar modelo
# best_params = study.best_params
# best_model = XGBRegressor(
#     booster = 'gbtree',
#     n_estimators = best_params['n_estimators'],
#     max_depth = best_params['max_depth'],
#     learning_rate = best_params['learning_rate'],
#     colsample_bytree = best_params['colsample_bytree'],
#     subsample = best_params['subsample'],
#     gamma = best_params['gamma'],
#     reg_alpha = best_params['reg_alpha'],
#     reg_lambda = best_params['reg_lambda']
# )

# best_model.fit(X_train, y_train)
# Ajuste
best_model = XGBRegressor(booster = 'gbtree')
best_model.fit(X_train, y_train)

# Pronostico
y_pred = best_model.predict(X_test)
y_pred_rounded = np.round(y_pred)
# Limitar los valores mayores a 5 a exactamente 5
y_pred_rounded = np.clip(y_pred_rounded, None, 5)
mae = mean_absolute_error(y_test, y_pred_rounded)
print(f"Mean Absolute Error (MAE): {mae}")

# Calcular SHAP values
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test)
plt.savefig('figures/shap.png')
# plt.show()
plt.close()