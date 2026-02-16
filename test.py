import requests

url = "http://127.0.0.1:8000/run_pipeline"

files = {
    "t0_file": open("Data/raw/OASIS_Fast/T0.nii.gz", "rb"),
    "t1_file": open("Data/raw/OASIS_Fast/T1.nii.gz", "rb"),
}

data = {
    "age": 62,
    "sex": "M",
    "interval_days": 517
}

response = requests.post(url, files=files, data=data)

print(response.json())
