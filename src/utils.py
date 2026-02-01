print("UTILS FILE LOADED")

import os
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    report = {}

    for model_name, model in models.items():

        # 🚨 ABSOLUTE BLOCK: CatBoost & XGB NEVER go to GridSearch
        if model_name == "CatBoost Regressor" or model_name == "XGB Regressor":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[model_name] = score
            continue

        # 🟢 Safe GridSearch for sklearn models only
        if model_name not in param:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[model_name] = score
            continue

        gs = GridSearchCV(
            model,
            param[model_name],
            cv=3,
            n_jobs=-1
        )

        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)
        score = r2_score(y_test, y_pred)

        report[model_name] = score

    return report

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
