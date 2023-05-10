from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('required_model.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    s_len = request.form.get('s_len')
    s_wid = request.form.get('s_wid')
    p_len = request.form.get('p_len')
    p_wid = request.form.get('p_wid')


    input_query = np.array([[int(s_len),int(s_wid),int(p_len),int(p_wid)]])
    result = model.predict(input_query)[0]

    return jsonify({'variety':str(result)})


if __name__ == '__main__':
    app.run(debug=True)
