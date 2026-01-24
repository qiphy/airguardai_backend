# keep_alive.py
import time
import requests

APP_URL = 'https://airguardai.onrender.com'
INTERVAL = 5 * 60  # 5 minutes in seconds
while True:
    try:
        response = requests.get(APP_URL)
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}: Pinged {APP_URL} - Status {response.status_code}")
    except Exception as err:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}: Error pinging {APP_URL}: {err}")
    time.sleep(INTERVAL)