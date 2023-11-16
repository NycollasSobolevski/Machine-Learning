from flask import Flask, request
import Training
import pandas as pd
app = Flask(__name__)


def toFloat(data):
    return float(data)

@app.route("/")
def hello_word():
    return "<p>Hello world</p>"

@app.route("/exo", methods=['POST'])
def verify():
    data = request.get_json()
    data = data['data']
    data = data.split(",")
    data = [data]
    data = pd.DataFrame(data=data)
    for i in data.columns:
        data[i] = data[i].apply(toFloat)
    # print(data)
    result = Training.Predict(data)
    print(*result)
    if(result[0]==2):
        return "is a planet"
    
    if(result[0]==1):
        return "is not a planet"
    
    return "Uai"

if __name__ == "__main__":
    app.run()