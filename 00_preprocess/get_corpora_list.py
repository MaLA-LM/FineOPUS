import requests
import json

def save_opus_json():
    url = "https://opus.nlpl.eu/opusapi/"
    params = {
        "preprocessing": "moses"
    }
    
    try:
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Get JSON data
        data = response.json()
        
        # Save to file
        filename = "OPUS_API_collection.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON data successfully saved to {filename}")
        print(f"File size: {len(response.content)} bytes")
        
        return filename
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

# Execute
save_opus_json()