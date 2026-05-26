from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib

#Fits model, evaluates performance, and prints metrics.

def evaluate_model(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    auc = roc_auc_score(
        (y_test == 'Yes').astype(int),
        y_prob
    )

    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_test,y_pred))

    print("\nConfusion Matrix")
    print("=" * 50)
    print(confusion_matrix(y_test, y_pred))

    print("\nMetrics")
    print("=" * 50)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label='Yes'):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, pos_label='Yes'):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, pos_label='Yes'):.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label='Yes'),
        "recall": recall_score(y_test, y_pred, pos_label='Yes'),
        "f1": f1_score(y_test, y_pred, pos_label='Yes'),
        "roc_auc": auc
    }

#Saves Trained pipeline
def save_model(model,filepath ="C:/Users/Dell/OneDrive/Desktop/ML Supervised Project/notebook/model_pipeline.pkl"):
    joblib.dump(model,filepath)
    print(f"Model saved to {filepath}")
    
