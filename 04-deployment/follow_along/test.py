import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 10
}


url = ' http://127.0.0.1:9696/predict'
response = requests.post(url, json=ride)
print(response.json())


# this was from before we added the flask stuff to make it a web service
# features = predict.prepare_features(ride)
# pred = predict.predict(features)
# print(pred)