from pathlib import Path
import json
import asyncio
import aiohttp

#BASE_URL = "http://localhost"
BASE_URL = "http://s-brasil6g01"
METRICS_URL = "http://s-brasil6g01:7400"
PORT_AI = "8000"
PORT_RAN = "8080"
NUM_ITERATIONS = 10

async def get(url, params):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            return response

async def post(url, files, data):
    form = aiohttp.FormData()

    for key in files.keys():
        form.add_field(
            key,
            files[key],
            #filename=files[key],
            content_type="image/jpeg"
        )

    for key in data.keys():
        form.add_field(key, data[key])

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form) as response:
            return response

async def start_server_measurement(name):
    try:
        response = await get(METRICS_URL + "/start-measurement/", params={"file_name": name})
        print(f"Server metrics: {await response.json()}")
    except:
        pass

async def end_server_measurement(name):
    try:
        response = await get(METRICS_URL + "/end-measurement/", params={"file_name": name})
        print(f"Server metrics: {response.status}")

        output_dir = Path("log/")
        output_dir.mkdir(parents=True, exist_ok=True)

        if response.status != 200:
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


async def test_2d(image_path, model="nano", device="cpu"):
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

    response = await post(url, files=files, data=data)

    if response.status == 200:
        time = response.headers["X-Time-Ms"]
        print(f"[OK] 2D Result: {time}")
        #save_image(response, "output_2d.jpg")
    else:
        print(f"[ERROR] Status: {response.status}")
        print(await response.text())



async def test_3d(
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

    response = await post(url, files=files, data=data)

    if response.status == 200:
        time = response.headers["X-Time-Ms"]
        print(f"[OK] 3D Result: {time}")
        #save_image(response, "output_3d.jpg")
    else:
        print(f"[ERROR] Status: {response.status}")
        print(await response.text())


async def test_ldpc(esno_db=8.4, num_prb=100, num_layers=1):
    url = f"{BASE_URL}:{PORT_RAN}/ldpc/"

    data = {
        "esno_db": esno_db,
        "num_prb": num_prb,
        "num_layers": num_layers,
    }

    response = await post(url, files={}, data=data)

    if response.status == 200:
        print(f"[OK] LDPC Result: {await response.json()}")
    else:
        print(f"[ERROR] Status: {response.status}")
        print(await response.text())

async def experiment(file_name, func):
    global NUM_ITERATIONS

    await start_server_measurement(file_name)
    requests = []

    for i in range(NUM_ITERATIONS):
        print(f"ITERATION {i}")
        requests.append(func())

    results = await asyncio.gather(*requests)
    await end_server_measurement(file_name)

async def main():
    global IMAGE_PATH

    #IMAGE_PATH = "images/coco_examples/3045175664_6e42bd43f3_z.jpg"
    IMAGE_PATH = "images/HYPERLAPSE_0001.JPG"
    device = "cuda"
    #device = "cpu"

    await experiment("2d.csv", lambda: test_2d(IMAGE_PATH, model="large", device=device))
    await experiment("3d.csv", lambda: test_3d(IMAGE_PATH, model="Res", device=device))
    #await experiment("ldpc.csv", lambda: test_ldpc(num_prb=100, num_layers=1, esno_db=8.4))

if __name__ == "__main__":
    asyncio.run(main())