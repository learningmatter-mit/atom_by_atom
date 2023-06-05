import requests

url = 'http://example.com/large_dataset.zip'
local_path = 'path/to/save/large_dataset.zip'

response = requests.get(url, stream=True)
response.raise_for_status()
with open(local_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192): 
        if chunk: # filter out keep-alive new chunks
            f.write(chunk)

