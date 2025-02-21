import requests

url = "http://127.0.0.1:5000/predict"
image_path = "input/rock01-018.png"

with open(image_path, "rb") as img_file:
    files = {"image": img_file}
    response = requests.post(url, files=files)

print(response.json())
