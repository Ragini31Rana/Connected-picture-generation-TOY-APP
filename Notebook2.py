"""
Importing the libraries
"""
import numpy as np
from PIL import Image
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

import torch
from diffusers import StableDiffusionInpaintPipeline

"""Getting images from paths"""

img_path = input('Give the path of the first image file\n')
img = cv2.imread(img_path) #Reading image
print("The first input image\n")
cv2.imshow(img)

img_path2 = input('Give the path of the second image file\n')
img2 = cv2.imread(img_path2) #Reading image
print("The second input image\n")
cv2.imshow(img2)

h, w, c = img2.shape

"""Defining the function to generate a gradient image"""

def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T

def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result

array = get_gradient_3d(w, h,(0, 0, 0),(140, 140, 140), (True, True, True))
I1 = Image.fromarray(np.uint8(array))

array = get_gradient_3d(w, h, (140, 140, 140), (0, 0, 0), (True, True, True))
I2 = Image.fromarray(np.uint8(array))

#creating white image
imgWhite = 225*(np.ones((h,w), np.uint8))
imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_GRAY2BGR)

"""Stacking the gradient images and grey image together"""

#stacking the masked images together
masked = np.hstack((I1,imgWhite, I2))

#creating white image
imgWhite1 = 255*(np.ones((h,w), np.uint8))
imgWhite1 = cv2.cvtColor(imgWhite1, cv2.COLOR_GRAY2BGR)

"""Stacking the images and white images together"""

#stacking the images and white image together
stacked = np.hstack((img, imgWhite1, img2))
height, width, channels = stacked.shape

"""Resizing the input image and the masked image"""

stacked2 = cv2.resize(stacked, (1520, 720), interpolation= cv2.INTER_AREA)  #, None, scale, scale)

masked2 = cv2.resize(masked, (1520, 720), interpolation= cv2.INTER_AREA)  #, None, scale, scale

"""Converting the original image and the masked image from array to PIL image to send them to the model"""

masked2 = Image.fromarray(masked2)
print("The masked image\n")
masked2


#converting the array image stacked to PIL
stacked2 = Image.fromarray(stacked2)
print("The stacked input image\n")
stacked2

from diffusers import DiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler

#scheduler = DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler")

"""Defining the pipeline of the model"""

print("Defining the pipeline\n")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = input("Enter the prompt\n")
choice = input("Do you want a negative prompt? Yes or No?\n")
if (choice == "Yes" or choice.lower() == "yes"):
  neg_prompt = input("Enter negative prompts\n")
else:
  neg_prompt = "Broken anatomy, Mutation, half body parts, two faces,more than two faces, distorted face"

image = pipe(prompt=prompt, image=stacked2, mask_image=masked2, num_inference_steps=50,
                        guidance_scale=8.5,
                        num_images_per_prompt=1, output_type = "pil",
                        negative_prompt=neg_prompt,
              height = height, width = width).images[0]


"""Converting the PIL image back to numpy array"""

#converting the PIL image back to numpy array image
img3 = np.array(image)
print("The resultant image\n")
cv2.imshow(img3)