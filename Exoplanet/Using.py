import Training
import pandas as pd
import json
def mudar(data):
    return float(data)


# Training.Fit()
dataJson = open("./json.json")
dataJson = json.load(dataJson)
dataJson = [(dataJson['data'].split(","))]
data2 = pd.DataFrame(dataJson)
for i in data2.columns:
    data2[i] = data2[i].apply(mudar)
print(data2)

data = pd.read_csv("exoTrain.csv")
row = 2

data = data.iloc[row-1:row, 1:]
print(data)
pred = Training.Predict(data2)

print("prediction from data:" )
# print(*pred)

dif=  0
print(f"'{data['FLUX.1']}' + '{data2[0]}' ")
# for key in data2:
#     col = f"FLUX.{key + 1}"
#     if data2[key] == data[col]:
#         dif+=1
# print(dif)
