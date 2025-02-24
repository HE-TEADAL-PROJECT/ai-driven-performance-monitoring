import requests
import pandas as pd
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def fetch_data(thanos_url, query, training_hours):
    end_time = int(time.time())
    start_time = end_time - (training_hours * 60 * 60)
    step = 60

    url = f'{thanos_url}/api/v1/query_range'
    params = {
        'query': query,
        'start': start_time,
        'end': end_time,
        'step': step
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            results = data['data']['result']
            timestamps = []
            values = []
            for result in results:
                for value in result['values']:
                    timestamps.append(datetime.fromtimestamp(float(value[0])))
                    values.append(float(value[1]))
            df = pd.DataFrame({'timestamp': timestamps, 'value': values})
            logger.info("Data fetched successfully.")
            return df
        else:
            logger.error(f"Error in response: {data.get('error', 'No error message')}")
            return None
    else:
        logger.error(f"HTTP Error: {response.status_code}")
        return None
