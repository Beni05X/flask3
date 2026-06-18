from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo
modelo = joblib.load("modelo_house_price.pkl")

# Columnas exactas con las que fue entrenado el modelo
COLUMNAS_MODELO = [
    "LotArea",
    "OverallQual",
    "BsmtFinSF1",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "GarageCars",
    "GarageArea"
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None

    if request.method == "POST":
        try:
            # Obtener datos del formulario
            datos = {
                "LotArea": float(request.form["LotArea"]),
                "OverallQual": float(request.form["OverallQual"]),
                "BsmtFinSF1": float(request.form["BsmtFinSF1"]),
                "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
                "1stFlrSF": float(request.form["FirstFlrSF"]),
                "2ndFlrSF": float(request.form["SecondFlrSF"]),
                "GrLivArea": float(request.form["GrLivArea"]),
                "GarageCars": float(request.form["GarageCars"]),
                "GarageArea": float(request.form["GarageArea"])
            }

            # Convertir a DataFrame respetando el orden de columnas
            entrada = pd.DataFrame([datos], columns=COLUMNAS_MODELO)

            # Hacer predicción
            resultado = modelo.predict(entrada)[0]

            prediccion = f"${resultado:,.2f}"

        except Exception as e:
            error = f"Ocurrió un error al realizar la predicción: {str(e)}"

    return render_template("index.html", prediccion=prediccion, error=error)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)