from pathlib import Path

import requests
import json

#BASE_URL = "http://localhost"
BASE_URL = "http://s-brasil6g01"
METRICS_URL = "http://s-brasil6g01:7400"
PORT_AI = "8000"
PORT_RAN = "8080"
NUM_ITERATIONS = 10

def start_server_measurement(name):
    try:
        response = requests.get(METRICS_URL + "/start-measurement/", params={"file_name": name}, timeout=5)
        print(f"Server metrics: {response.json()}")
    except:
        pass

def end_server_measurement(name):
    try:
        response = requests.get(METRICS_URL + "/end-measurement/", params={"file_name": name}, timeout=5)
        print(f"Server metrics: {response.status_code}")

        output_dir = Path("log/")
        output_dir.mkdir(parents=True, exist_ok=True)

        if response.status_code != 200:
            return

        file_path = output_dir / name
        file_path.write_text(response.text, encoding="utf-8")
    except:
        pass


def save_image(response, filename):
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"[OK] Saved: {filename}")
    else:
        print(f"[ERROR] Status: {response.status_code}")
        try:
            print(response.json())
        except Exception:
            print(response.text)


def test_2d(image_path, model="nano", device="cpu"):
    url = f"{BASE_URL}:{PORT_AI}/2d-object-detection/"

    params = {
        "model": model,
        "device": device
    }

    files = {
        "file": open(image_path, "rb")
    }

    data = {
        "params": json.dumps(params)  # IMPORTANT: string JSON
    }

    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        time = response.headers["X-Time-Ms"]
        print(f"[OK] 2D Result: {time}")
        #save_image(response, "output_2d.jpg")
    else:
        print(f"[ERROR] Status: {response.status_code}")
        print(response.text)



def test_3d(
    image_path,
    model="Res",
    device="cpu",
    threshold=0.25,
    focal_length=0,
    principal_point=[]
):
    url = f"{BASE_URL}:{PORT_AI}/3d-object-detection/"

    params = {
        "model": model,
        "device": device,
        #"threshold": threshold,
        #"focal_length": focal_length,
        #"principal_point": principal_point
    }

    files = {
        "file": open(image_path, "rb")
    }

    data = {
        "params": json.dumps(params)  # IMPORTANT
    }

    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        time = response.headers["X-Time-Ms"]
        print(f"[OK] 3D Result: {time}")
        #save_image(response, "output_3d.jpg")
    else:
        print(f"[ERROR] Status: {response.status_code}")
        print(response.text)


def test_ldpc(esno_db=8.4, num_prb=100, num_layers=1):
    url = f"{BASE_URL}:{PORT_RAN}/ldpc/"

    data = {
        "esno_db": esno_db,
        "num_prb": num_prb,
        "num_layers": num_layers,
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print(f"[OK] LDPC Result: {response.json()}")
    else:
        print(f"[ERROR] Status: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    IMAGE_PATH = "coco_examples/3045175664_6e42bd43f3_z.jpg"
    device = "cuda"
    #device = "cpu"

    start_server_measurement("2d.csv")
    for i in range(NUM_ITERATIONS):
        print(f"ITERATION {i}")
        test_2d(IMAGE_PATH, model="large", device=device)
    end_server_measurement("2d.csv")

    start_server_measurement("3d.csv")
    for i in range(NUM_ITERATIONS):
        print(f"ITERATION {i}")
        test_3d(IMAGE_PATH, model="Res", device=device)
    end_server_measurement("3d.csv")

    start_server_measurement("ldpc.csv")
    for i in range(NUM_ITERATIONS):
        print(f"ITERATION {i}")
        test_ldpc(num_prb=100, num_layers=1, esno_db=8.4)
    end_server_measurement("ldpc.csv")
