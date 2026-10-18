import os
import requests
import time

base_url = 'http://localhost:8000' 
endpoint = '/api/v1/process_frame'  
full_url = base_url + endpoint

def check_gate_access(url, image_path, verbose=True):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image {image_path} not found.")
        return None

    with open(image_path, "rb") as image_file:
        # Include filename and MIME type so FastAPI accepts the upload
        files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
        
        try:
            response = requests.post(url, files=files)
            status_code = response.status_code
            
            if status_code == 200:
                server_decision = response.json()
                if verbose:
                    print(f"\n--- Checking Image: {os.path.basename(image_path)} ---")
                    
                action = server_decision.get("action")
                if action == "open_gate":
                    truck_count = server_decision.get("trucks_in_frame")
                    print(f"🟢 [HARDWARE SIGNAL]: OPENING GATE! ({truck_count} New Truck(s) Detected)")
                else:
                    print("🔴 [HARDWARE SIGNAL]: KEEP GATE CLOSED. (No new trucks or nothing detected)")
                    
                return server_decision
            else:
                print(f"❌ Server returned an error: {status_code}")
                # Print error details from the server if present
                print(f"Details: {response.text}")
                return None
        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to the API. Is the server running?")
            return None

if __name__ == "__main__":
    # Path to your test image
    img = "/content/drive/MyDrive/Smart Gate/Data/processed/Trucks/test/images/100_resized_jpg.rf.9a6e56b12127d9b7ae25529613a3b5f6.jpg"
    image_files = [img, img]
    
    print(f"Starting Smart Gate Camera Simulation targeting: {full_url}")
    print("=" * 60)
    
    for img_path in image_files:
        result = check_gate_access(full_url, img_path)
        time.sleep(1)
