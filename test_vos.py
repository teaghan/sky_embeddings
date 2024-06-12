import requests

url = "https://ws-cadc.canfar.net/vault/capabilities"
try:
    response = requests.get(url, timeout=2000000)  # Increase the timeout to 30 seconds
    response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
    print("Successfully accessed the URL")
    print("Content:", response.text)
except requests.exceptions.RequestException as e:
    print(f"Failed to access the URL: {e}")