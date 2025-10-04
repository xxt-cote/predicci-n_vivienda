# estudiante:"Maria José Garcia"#

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Datos entregados
datos = {
    "Superficie_m2": [50, 70, 65, 90, 45],
    "Num_Habitaciones": [1, 2, 2, 3, 1],
    "Distancia_Metro_km": [0.5, 1.2, 0.8, 0.2, 2.0],
    "Precio_UF": [2500, 3800, 3500, 5200, 2100],
}
df = pd.DataFrame(datos)

X = df[["Superficie_m2", "Num_Habitaciones", "Distancia_Metro_km"]]
y = df["Precio_UF"]

modelo = LinearRegression()
modelo.fit(X, y)


y_pred = modelo.predict(X)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("Coeficientes:", modelo.coef_)
print("Intercepto:", modelo.intercept_)
print("R^2:", r2)
print("RMSE:", rmse)
df_resultado = pd.DataFrame({"Real": y, "Predicho": y_pred.round(2)})
print(df_resultado)
