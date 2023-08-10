#python3
import os
import glob
import requests
import json
import cv2
import numpy as np
import re
import sys
import torch
from PIL import Image
from pprint import pprint
import base64
from io import BytesIO
import torchvision.transforms.functional as F
from torchvision.io import read_video, read_image, ImageReadMode
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.io import read_video, read_image, ImageReadMode
from torchvision.utils import flow_to_image
import cv2
from torchvision.io import write_jpeg
import pickle

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to("cpu")
model = model.eval()


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files, key=natural_sort_key)


y_paths = get_image_paths("/home/streamline/sd-webui-docker/output/animation/tempv2/tempv2 test case/ref")
lineart_input_image_path = get_image_paths("/home/streamline/sd-webui-docker/output/animation/tempv2/tempv2 test case/line")
segmentation_input_image_path = get_image_paths("/home/streamline/sd-webui-docker/output/animation/tempv2/tempv2 test case/seg")
output_directory = "/home/streamline/sd-webui-docker/output/animation/tempv2/tempv2 test case/out"
init_image_path = "/home/streamline/sd-webui-docker/output/txt2img/2023-08-01/00041-1186734096.png"
encoded_input_image_path = get_image_paths("/home/streamline/sd-webui-docker/output/animation/tempv2/tempv2 test case/flow")
opitcal_flow_path = get_image_paths("/home/streamline/sd-webui-docker/output/animation/tempv2/tempv2 test case/flow")
last_image_path = get_image_paths("/home/streamline/sd-webui-docker/output/animation/tempv2/tempv2 test case/last")
height = 1024
width = 1024
positiveprompt = "1 girl,hatsune miku, black background,simple background,full body, (black stocking:1.2), (black thigh high:1.2), black dress, white shirt,white shoes,bare shoulder,"
negprompt = "(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), bad composition, inaccurate eyes, extra digit, fewer digits, (extra arms:1.2), easynegative,negative_hand-neg,"
seed = 1186734096
sampler = "DPM++ SDE Karras"


def get_controlnet_models():
    url = "http://localhost:7860/controlnet/model_list"

    temporalnet_model = None
    temporalnet_re = re.compile("^temporalnetversion2 \[.{8}\]")

    lineartanime_model = None
    hed_re = re.compile("^control_.*lineart_anime.* \[.{8}\]")

    seg_model = None
    seg_re = re.compile("^control_.*seg.* \[.{8}\]")

    response = requests.get(url)
    if response.status_code == 200:
        models = json.loads(response.content)
    else:
        raise Exception("Unable to list models from the SD Web API! "
                        "Is it running and is the controlnet extension installed?")

    for model in models['model_list']:
        if temporalnet_model is None and temporalnet_re.match(model):
            temporalnet_model = model
        elif lineartanime_model is None and hed_re.match(model):
            lineartanime_model = model
        elif seg_model is None and seg_re.match(model):
            seg_model = model

    assert temporalnet_model is not None, "Unable to find the temporalnet2 model!  Ensure it's copied into the stable-diffusion-webui/extensions/models directory!"
    assert lineartanime_model is not None, "Unable to find the lineart anime model!  Ensure it's copied into the stable-diffusion-webui/extensions/models directory!"
    assert seg_model is not None, "Unable to find the segmentation model!  Ensure it's copied into the stable-diffusion-webui/extensions/models directory!"

    return temporalnet_model, lineartanime_model, seg_model


TEMPORALNET_MODEL, Lineart_MODEL, Segmentation_MODEL = get_controlnet_models()


def send_request(encoded_image, current_image_path, lineart_input_image_path, segmentation_input_image_path):
    url = "http://localhost:7860/sdapi/v1/txt2img"

    # Load and process the current image
    with open(current_image_path[0], "rb") as b:
        current_image = base64.b64encode(b.read()).decode("utf-8")

    # Load and process the controlnet inputs
    with open(lineart_input_image_path[0], "rb") as b:
        lineart_input_image = base64.b64encode(b.read()).decode("utf-8")

    with open(segmentation_input_image_path[0], "rb") as b:
        seg_input_image = base64.b64encode(b.read()).decode("utf-8")
    

    data = {
        "init_images": [current_image],
        "inpainting_fill": 0,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 1,
        "inpainting_mask_invert": 1,
        "resize_mode": 0,
        "denoising_strength": 0.9,
        "prompt": positiveprompt,
        "negative_prompt": negprompt,
        "alwayson_scripts": {
            "ControlNet":{
                "args": [
                    {
                        "input_image": lineart_input_image,
                        "module": "none",
                        "model": "control_v11p_sd15s2_lineart_anime [3825e83e]",
                        "weight": 1.25,
                        "guidance_start": 0,
                        "guidance_end": 1,
                        "control_mode": 2,
                        "pixel_perfect": False,
                        "resize_mode": 0,
                   },
                    {
                        "input_image": current_image,
                        "module": "none",
                        "model": "control_v11f1e_sd15_tile [a371b31b]",
                        "weight": 0.6,
                        "guidance_start": 0.2,
                        "guidance_end": 0.6,
                        "control_mode": 1,
                        "pixel_perfect": False,
                        "resize_mode": 0,
                   },
                    {
                        "input_image": encoded_image,
                        "model": "temporalnetversion2 [b146ac48]",
                        "module": "none",
                        "weight": 1.2,
                        "guidance_start": 0.2,
                        "guidance_end": 1,
                        # "processor_res": 512,
                        "threshold_a": 64,
                        "threshold_b": 64,
                        "resize_mode": 0,
                    },
                    {
                        "input_image": seg_input_image,
                        "model": "control_v11p_sd15_seg [e1f51eb9]",
                        "module": "none",
                        "weight": 1.5,
                        "guidance_start": 0.2,
                        "guidance_end": 1,
                        "control_mode": 2,
                        "pixel_perfect": False,
                        "resize_mode": 0,
                    },
                    {
                        "input_image": current_image,
                        "module": "reference_only",
                        "weight": 0.6,
                        "guidance_start": 0,
                        "guidance_end": 0.2,
                        "control_mode": 0.8,
                        "pixel_perfect": False,
                        "resize_mode": 0,
                    }
                    
                  
                ]
            }
        },
        "seed": seed,
        "subseed": -1,
        "subseed_strength": -1,
        "sampler_index": sampler,
        "batch_size": 1,
        "n_iter": 1,
        "steps": 40,
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "restore_faces": False,
        "override_settings": {},
        "override_settings_restore_afterwards": True
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.content
    else:
        try:
            error_data = response.json()
            print("Error:")
            print(str(error_data))
            
        except json.JSONDecodeError:
            print(f"Error: Unable to parse JSON error data.")
        return None



def infer(frameA, frameB):
    
    
    input_frame_1 = read_image(str(frameA), ImageReadMode.RGB)
   
    input_frame_2 = read_image(str(frameB), ImageReadMode.RGB)
 
    
    #img1_batch = torch.stack([frames[0]])
    #img2_batch = torch.stack([frames[1]])

    img1_batch = torch.stack([input_frame_1])
    img2_batch = torch.stack([input_frame_2])
    
    
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()


    def preprocess(img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[512, 512])
        img2_batch = F.resize(img2_batch, size=[512, 512])
        return transforms(img1_batch, img2_batch)

    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

    list_of_flows = model(img1_batch.to("cpu"), img2_batch.to("cpu"))

    predicted_flow = list_of_flows[-1][0]
    opitcal_flow_path = os.path.join(output_directory, f"flow_{i}.png")

    flow_img = flow_to_image(predicted_flow).to("cpu")
    flow_img = F.resize(flow_img, size=[height, width])

    write_jpeg(flow_img, opitcal_flow_path)

    return opitcal_flow_path

output_images = []
output_paths = []

# Initialize with the first image path

result = init_image_path
output_image_path = os.path.join(output_directory, f"output_image_0.png")

#with open(output_image_path, "wb") as f:
   # f.write(result)

last_image = cv2.imread("/home/streamline/sd-webui-docker/output/animation/tempv2/tempv2 test case/last/00041-1186734096.png")
last_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB)
last_image = F.resize(Image.fromarray(last_image), size=[height, width])
flow_image = cv2.imread("/home/streamline/sd-webui-docker/output/animation/tempv2/tempv2 test case/flow/flow_1.png")
flow_image = cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB)
six_channel_image = np.dstack((last_image, flow_image))

    # Serializing the 6-channel image
serialized_image = pickle.dumps(six_channel_image)

    # Encoding the serialized image
encoded_images = base64.b64encode(serialized_image).decode('utf-8')


print(lineart_input_image_path[0])
result = send_request(encoded_images, y_paths, lineart_input_image_path, segmentation_input_image_path)
data = json.loads(result)

for j, encoded_image in enumerate(data["images"]):
    if j == 0:
        output_image_path = os.path.join(output_directory, f"output_image_1.png")
        last_image_path = output_image_path
    else:
        output_image_path = os.path.join(output_directory, f"controlnet_image_{j}_1.png")
    with open(output_image_path, "wb") as f:
        f.write(base64.b64decode(encoded_image))
    print(f"Written data for frame 1:")
