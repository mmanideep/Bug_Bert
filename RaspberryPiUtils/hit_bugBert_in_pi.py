import json
import requests

def send_api_requests(json_file_path, api_url="http://192.168.0.114:9090/predict/"):
    """Reads a JSON file, extracts descriptions, and sends them to an API.

    Args:
        json_file_path (str): The path to the JSON file.
        api_url (str, optional): The API endpoint URL. Defaults to "http://192.168.0.114:9090/predict/".
    """
    
    try:
        # Load the JSON file
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Ensure data is a list of dictionaries
        if not isinstance(data, list):
            raise ValueError("Invalid JSON format: Data must be a list of dictionaries.")

        for item in data:
            # Extract the description if the key exists
            description = item.get("description")
            if description:
                # Prepare the payload
                payload = {"text": description}

                try:
                    # Send POST request
                    response = requests.post(api_url, json=payload)
                    response.raise_for_status()

                    print(f"Sent key: {item["key"]}")  # Print first 50 characters of description
                    print(f"Response: {response.json()}\n")
                except requests.exceptions.RequestException as e:
                    print(f"Error sending request for key: {item["key"]} {e}")

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")


# Example usage
json_file_path = "D:\AI and ML Masters\BugBert\OldFiles\data\data_part1.json"  # Replace with your file path
send_api_requests(json_file_path)
