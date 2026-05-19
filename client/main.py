import requests
import json

#BASE_URL = "http://localhost"
BASE_URL = "http://s-brasil6g01"
PORT_AI = "8000"
PORT_RAN = "8080"


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

    save_image(response, "output_2d.jpg")


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

    save_image(response, "output_3d.jpg")


def test_ldpc(esno_db=8.4, num_prb=100, num_layers=4, num_slots=10):
    url = f"{BASE_URL}:{PORT_RAN}/ldpc/"

    data = {
        "esno_db": esno_db,
        "num_prb": num_prb,
        "num_layers": num_layers,
        "num_slots": num_slots
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("[OK] LDPC Result:")
        print(response.json())
    else:
        print(f"[ERROR] Status: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    IMAGE_PATH = "coco_examples/3045175664_6e42bd43f3_z.jpg"
    device = "cuda"
    #device = "cpu"

    test_2d(IMAGE_PATH, model="nano", device=device)

    test_3d(IMAGE_PATH, model="Res", device=device)

    test_ldpc(num_layers=10)