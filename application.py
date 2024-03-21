import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template

application = Flask(__name__)

app = application

model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home_page():
    return render_template('index.html')

@app.route("/submit", methods=['POST'])
def predict():
    """ Selected feature are Dependents, tenure, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, Contract,
        PaperlessBilling, MonthlyCharges, TotalCharges """
        
    Dependents = request.form['Dependents']
    tenure = float(request.form['tenure'])
    OnlineSecurity = request.form['OnlineSecurity']
    OnlineBackup = request.form['OnlineBackup']
    DeviceProtection = request.form['DeviceProtection']
    TechSupport = request.form['TechSupport']
    Contract = request.form['Contract']
    PaperlessBilling = request.form['PaperlessBilling']
    MonthlyCharges = float(request.form['MonthlyCharges'])
    TotalCharges = float(request.form['TotalCharges'])
    
    data = [[Dependents, tenure, OnlineSecurity, OnlineBackup,
            DeviceProtection, TechSupport, Contract, PaperlessBilling,
            MonthlyCharges, TotalCharges]]
    
    df = pd.DataFrame(data, columns=['Dependents', 'tenure', 'OnlineSecurity',
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                     'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])
    

    
    categorical_features = {feature for feature in df.columns if df[feature].dtypes == 'O'}
    print(categorical_features)
    
    encoder = LabelEncoder()
    for feature in categorical_features:
        df[feature] = encoder.fit_transform(df[feature])
        
    single = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    probability = np.round(probability*100, 2)
    
    
    return render_template('predict.html', churn_result=probability)
    
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")