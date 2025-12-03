# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def loadDataset(filePath: str) -> pd.DataFrame:
    return pd.read_csv(filePath, compression="zip").copy()


def cleanDataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)]
    df.loc[df["EDUCATION"] >= 4, "EDUCATION"] = 4
    df = df.dropna()
    return df


def splitFeaturesTarget(df: pd.DataFrame):
    x = df.drop(columns=["default"])
    y = df["default"]
    return x, y


def buildPipeline(categoricalFeatures, numericFeatures) -> Pipeline:
    transformers = [
        ("categorical", OneHotEncoder(), categoricalFeatures),
        ("numeric", StandardScaler(), numericFeatures),
    ]
    preprocessor = ColumnTransformer(transformers=transformers)

    selector = SelectKBest(score_func=f_classif)
    pca = PCA()
    classifier = MLPClassifier(max_iter=15000, random_state=21)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("pca", pca),
            ("mlp", classifier),
        ]
    )
    return pipe


def getSearchModel(pipe: Pipeline) -> GridSearchCV:
    paramGrid = {
        "selector__k": [20],
        "pca__n_components": [None],
        "mlp__hidden_layer_sizes": [(50, 30, 40, 60)],
        "mlp__alpha": [0.26],
        "mlp__learning_rate_init": [0.001],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=paramGrid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )
    return search


def saveModel(model, modelPath: str):
    os.makedirs(os.path.dirname(modelPath), exist_ok=True)
    with gzip.open(modelPath, "wb") as f:
        pickle.dump(model, f)


def getMetricsDict(yTrue, yPred, datasetName: str) -> dict:
    return {
        "type": "metrics",
        "dataset": datasetName,
        "precision": precision_score(yTrue, yPred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(yTrue, yPred),
        "recall": recall_score(yTrue, yPred, zero_division=0),
        "f1_score": f1_score(yTrue, yPred, zero_division=0),
    }


def getConfusionDict(yTrue, yPred, datasetName: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(yTrue, yPred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": datasetName,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


def writeMetricsFile(records, outputPath: str):
    Path(os.path.dirname(outputPath)).mkdir(parents=True, exist_ok=True)
    with open(outputPath, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")


def run():
    trainDf = loadDataset("files/input/train_data.csv.zip")
    testDf = loadDataset("files/input/test_data.csv.zip")

    trainDf = cleanDataframe(trainDf)
    testDf = cleanDataframe(testDf)

    xTrain, yTrain = splitFeaturesTarget(trainDf)
    xTest, yTest = splitFeaturesTarget(testDf)

    categoricalCols = ["SEX", "EDUCATION", "MARRIAGE"]
    numericCols = [col for col in xTrain.columns if col not in categoricalCols]

    pipeline = buildPipeline(categoricalCols, numericCols)
    searchModel = getSearchModel(pipeline)
    searchModel.fit(xTrain, yTrain)

    saveModel(searchModel, "files/models/model.pkl.gz")

    yTrainPred = searchModel.predict(xTrain)
    yTestPred = searchModel.predict(xTest)

    trainMetrics = getMetricsDict(yTrain, yTrainPred, "train")
    testMetrics = getMetricsDict(yTest, yTestPred, "test")

    trainCm = getConfusionDict(yTrain, yTrainPred, "train")
    testCm = getConfusionDict(yTest, yTestPred, "test")

    records = [trainMetrics, testMetrics, trainCm, testCm]
    writeMetricsFile(records, "files/output/metrics.json")


if __name__ == "__main__":
    run()
