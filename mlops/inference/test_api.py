import requests

url = "http://localhost:5001/predict"
data = {"features": [5.1, 3.5, 1.4, 0.2]}

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response Content:", response.text)
    print("Response Headers:", response.headers)

    result = response.json()
    print("Prediction:", result['prediction'])
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)