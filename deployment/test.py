import requests

url = "http://1161e797-86d9-48a9-bc66-e132f190c003.eastus.azurecontainer.io/score"

payload="{\"SepalLengthCm\": 6.6, \"SepalWidthCm\": 3, \"PetalLengthCm\": 4.4, \"PetalWidthCm\": 1.4}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
#print(response.content)
print(response.text)