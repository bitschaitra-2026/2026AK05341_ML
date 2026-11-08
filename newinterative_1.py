
import streamlit as st

model_choice=st.selectbox(
    label="Choose the model:",
    options=[
        ('Logistic regression','lr')
         ],
         format_func=lambda x:x[0]
)
selected_model_key= model_choice[1]

#import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearnmetrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, roc_auc_score

# Dropdown menu to choose the model
model_dropdown = widgets.Dropdown(
    options=[
        ('Logistic Regression', 'lr'),
        ('Decision Tree', 'dt'),
        ('K-Nearest Neighbors', 'knn'),
        ('Gaussian Naive Bayes', 'nb'),
        ('Support Vector Machine (SVM)', 'svm')
    ],
    value='lr',
    description='Select Model:',
    style={'description_width': 'initial'}
)

# Button to trigger execution
run_button = widgets.Button(
    description='Train & Evaluate',
    button_style='success',
    tooltip='Click to train and evaluate the selected model',
    icon='play'
)

# Output area for results
output_area = widgets.Output()

def run_model_pipeline(b):
    with output_area:
        clear_output()
        selected_option = model_dropdown.value
        
        # Verify that preprocessed data exists
        if 'X_train' not in globals() or 'X_test' not in globals():
            print("Error: Preprocessed data (X_train, X_test, y_train, y_test) not found in the kernel. Please make sure to run the preprocessing cells first!")
            return
            
        print(f"Executing model pipeline with {model_dropdown.label}... Please wait.\n")
        
        # Initialize chosen model
        if selected_option == 'lr':
            model = LogisticRegression(max_iter=1000)
        elif selected_option == 'dt':
            model = DecisionTreeClassifier(random_state=1, max_depth=9, criterion='entropy')
        elif selected_option == 'knn':
            model = KNeighborsClassifier(n_neighbors=8)
        elif selected_option == 'nb':
            model = GaussianNB()
        elif selected_option == 'svm':
            model = SVC(kernel='rbf', probability=True, random_state=42)
            
        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results formatted nicely
        print("="*50)
        print(f"Evaluation Metrics for {model_dropdown.label}")
        print("="*50)
        print(f"Accuracy:   {acc:.4f}")
        print(f"Precision:  {prec:.4f}")
        print(f"Recall:     {rec:.4f}")
        print(f"F1 Score:   {f1:.4f}")
        print(f"MCC Score:  {mcc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("="*50)
        
        # ROC Curve Plotting
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Receiver Operating Characteristic - {model_dropdown.label}')
        ax.legend(loc="lower right")
        plt.show()

run_button.on_click(run_model_pipeline)

# Render dashboard interface
layout = widgets.VBox([
    widgets.HBox([model_dropdown, run_button], layout=widgets.Layout(margin='10px 0px')),
    output_area
])
display(layout)
