import cv2
import numpy as np
import requests

print("Creating dummy video...")
out = cv2.VideoWriter('dummy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (224, 224))
for _ in range(60): # 2 seconds
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    out.write(frame)
out.release()
print("Dummy video created.")

print("Uploading to localhost:8001 API...")
with open('dummy.mp4', 'rb') as f:
    files = {'video': ('dummy.mp4', f, 'video/mp4')}
    data = {'query': 'test query'}
    try:
        response = requests.post('http://127.0.0.1:8001/api/summarize', files=files, data=data)
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)
    except Exception as e:
        print("Request failed:", str(e))
