import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# Disable PyplotGlobalUseWarning
# You are calling st.pyplot() without any arguments.
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(persist=True)
def load_data():
    data = pd.read_csv("mushrooms.csv")
    labelencoder=LabelEncoder()
    for col in data.columns:
        data[col] = labelencoder.fit_transform(data[col])
    return data

@st.cache(persist=True)
def split(df):
    y = df.type
    x = df.drop(columns=['type'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

def create_model(name=None, C=None, kernel=None, gamma=None, penalty='l2', max_iter=None, n_estimators=None, max_depth=None, bootstrap=None, n_jobs=-1, x_train=None, x_test=None, y_train=None, y_test=None, class_names=None):
    st.subheader(name)
    if('SVM' in name):
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        evaluate_model(model, x_test, y_test, y_pred, class_names)
        return model
    elif('Logistic' in name):
        model = LogisticRegression(C=C, penalty=penalty, max_iter=max_iter)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        evaluate_model(model, x_test, y_test, y_pred, class_names)
        return model
    elif('Random Forest' in name):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=n_jobs)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        evaluate_model(model, x_test, y_test, y_pred, class_names)
        return model

def evaluate_model(model, x_test, y_test, y_pred, class_names):
    accuracy = model.score(x_test, y_test)
    st.write("Accuracy: ", accuracy.round(2))
    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))

def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-Recall Curve')
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
        
def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    df = load_data()
    class_names = ['edible', 'poisonous']
    
    x_train, x_test, y_train, y_test = split(df)

    # Classifier slider
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    # Modify different model parameters
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')

    if classifier == 'Support Vector Machine (SVM)':      
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            name = 'Support Vector Machine (SVM) Results'
            model = create_model(name=name, C=C, kernel=kernel, gamma=gamma, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, class_names=class_names)
            plot_metrics(metrics, model, x_test, y_test, class_names)
    
    if classifier == 'Logistic Regression':
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            name = 'Logistic Regression Results'
            model = create_model(name=name, C=C, penalty='l2', max_iter=max_iter, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, class_names=class_names)
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier == 'Random Forest':
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            name = 'Random Forest Results'
            model = create_model(name=name, n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, class_names=class_names)
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
        st.markdown("This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms "
        "in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, "
        "or of unknown edibility and not recommended. This latter class was combined with the poisonous one.")

if __name__ == '__main__':
    main()


