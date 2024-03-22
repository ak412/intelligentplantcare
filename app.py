from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, static_url_path='/static')

# Load the trained model
with open('crop.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('predictorform.html')

@app.route('/results', methods=['POST'])
def results():
    # Get input features from the form
    nitrogen = float(request.form['nitrogen'])
    phosphorus = float(request.form['phosphorus'])
    potash = float(request.form['potash'])
    temp = float(request.form['temp'])
    humid = float(request.form['humid'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Preprocess the input features
    input_features = np.array([[nitrogen, phosphorus, potash, temp, humid, ph, rainfall]])

    # Make predictions
    prediction = model.predict(input_features)
    
    # Pass the input features and prediction to the template
    return render_template('resultsform.html', 
                           nitrogen=nitrogen,
                           phosphorus=phosphorus,
                           potash=potash,
                           temp=temp,
                           humid=humid,
                           ph=ph,
                           rainfall=rainfall,
                           prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
