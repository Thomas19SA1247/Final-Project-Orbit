import numpy as np
import pandas as pd
from process import preparation, generate_response
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

loaded_model = joblib.load('knn.sav')

# download nltk
preparation()

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/prediksi')
def pred():
    return render_template('prediksi.html')

@app.route('/prediksi', methods=['POST'])
def prediksi():
    nm, um, jm, Mood, Sering_Pusing, Sering_Menangis, Sulit_Tidur, Pola_Makan, Sering_Gelisah = [x for x in request.form.values()]

    data = []

    data.append(int(Mood))
    data.append(int(Sering_Pusing))
    data.append(int(Sering_Menangis))
    data.append(int(Sulit_Tidur))
    data.append(int(Pola_Makan))
    data.append(int(Sering_Gelisah))

    data = np.array(data)
    
    #reshape array
    data = data.reshape(1,-1)
    
    predicted_bit = np.round(loaded_model.predict(data)).astype('int')
    prediction = [np.argmax(element) for element in predicted_bit]

    if prediction==[1]:
        hasil = 'CUKUP SEHAT'
    elif prediction==[0]:
        hasil = 'KURANG SEHAT'
    elif prediction==[2]:
        hasil = 'SEHAT'
    elif prediction==[3]:
        hasil = 'TIDAK SEHAT'

    return render_template('hasil.html', hasil_prediksi=hasil, Mood=Mood, Sering_Pusing=Sering_Pusing, Sering_Menangis=Sering_Menangis, Sulit_Tidur=Sulit_Tidur, Pola_Makan=Pola_Makan, Sering_Gelisah=Sering_Gelisah)

    
@app.route('/chatbot')
def bot():
    return render_template('chatbot.html')

@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result
    
if __name__ == '__main__':
    app.run(debug=True)