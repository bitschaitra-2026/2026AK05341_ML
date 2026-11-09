import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("E-Commerce Classification Model Evaluator")
st.write("BITS M.Tech AIML - Machine Learning Assignment 2")

# Sidebar for Test Data Upload
st.sidebar.header("1. Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload test data CSV", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("### Test Dataset Preview", test_df.head())
    
    # Target column selection
    target_column = st.sidebar.selectbox("Select Target Column", test_df.columns)
    
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Model Selection Dropdown
    st.sidebar.header("2. Select Model")
    model_name = st.sidebar.selectbox(
        "Choose Classification Model", 
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
    )

    model_files = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree": "model/decision_tree.joblib",
        "KNN": "model/knn.joblib",
        "Naive Bayes": "model/naive_bayes.joblib",
        "Support Vector Machine": "model/svm.joblib"
    }

    if st.sidebar.button("Evaluate Model"):
        model_path = model_files[model_name]
        if os.path.exists(model_path):
            # Load pre-trained model
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            try:
                y_prob = model.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            except Exception:
                auc = "N/A"

            # Display Evaluation Metrics
            st.write(f"### Evaluation Metrics for: {model_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("Precision", f"{precision:.4f}")
            col3.metric("Recall", f"{recall:.4f}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("F1 Score", f"{f1:.4f}")
            col5.metric("MCC Score", f"{mcc:.4f}")
            col6.metric("AUC Score", f"{auc}" if isinstance(auc, str) else f"{auc:.4f}")

            # Confusion Matrix & Classification Report
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            st.write("### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

        else:
            st.error(f"Model file `{model_path}` not found in the repository! Please create and upload the `model/` folder containing all 5 pkl/joblib files.")
else:
    st.info("Please upload your test dataset CSV file using the sidebar to proceed.")
