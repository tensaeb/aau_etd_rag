import requests

url = "http://192.168.1.22:1234/v1/chat/completions"

payload = {
    "model": "mistral",
    "messages": [
        {"role": "user", "content": "Explain artificial intelligence in one sentence."}
    ],
    "temperature": 0.0
}

response = requests.post(url, json=payload)
print(response.json()["choices"][0]["message"]["content"])
