print("UTILS FILE LOADED")

import os
import pickle
import sys
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report[name] = r2_score(y_test, y_pred)
    return report
