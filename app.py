from flask import Flask, render_template, request, jsonify
import joblib
from flask_cors import CORS


model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sms_text = data.get('sms_text', '')
    vectorized_sms = vectorizer.transform([sms_text])
    prediction = model.predict(vectorized_sms.toarray())
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
