from flask import Flask, request, redirect, url_for, render_template,session
import joblib

application = Flask(__name__)

vectorizer = joblib.load('vectorizer.pkl')
spamorham_model = joblib.load("spam_ham_model.pkl")

@application.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        message = request.form.get('message')
        print(message)
        vectorized_message = vectorizer.transform([message])
        result = spamorham_model.predict(vectorized_message)
        print(result)
        result = result[0]
        return redirect(url_for('predict',result=result))
    return render_template('index.html')

@application.route('/predict/<result>',methods=['GET','POST'])
def predict(result):
    return render_template('result.html',result=result)


if __name__ == "__main__":
    application.run(debug=True)
