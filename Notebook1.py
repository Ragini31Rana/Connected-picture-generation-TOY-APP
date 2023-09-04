
"""
Installing the libraries
"""
import numpy as np
from PIL import Image
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline

"""Getting images from urls and then saving it"""

img_path = input('Give the path of the image file\n')
img = cv2.imread(img_path) #Reading image
print("The input image\n")
cv2.imshow(img)

height, width, channels = img.shape

"""Defining the function to generate gradient images"""

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

array = get_gradient_3d(256, 512, (0, 0, 0), (255, 255, 255), (True, True, True))
I1 = Image.fromarray(np.uint8(array))

array = get_gradient_3d(256, 512, (255, 255, 255), (0, 0, 0), (True, True, True))
I2 = Image.fromarray(np.uint8(array))

"""Resizing the image"""

img = cv2.resize(img, (512, 512))
#, None, scale, scale)

"""Converting the image from array to PIL as an input for the model"""

img2 = Image.fromarray(img)

"""Stacking the gradient images together to use as a masked image"""

#stacking the black and white images together
masked = np.hstack((I1, I2))

"""Converting the masked image from array to PIL image"""

masked = Image.fromarray(masked)
print("The masked image\n")
masked

"""Defining the pipeline of the model and generating the result image"""

print("Defining the pipeline\n")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = input("Enter the prompt\n")
choice = input("Do you want a negative prompt? Yes or No?\n")
if (choice == "Yes" or choice.lower() == "yes"):
  neg_prompt = input("Enter negative prompts\n")
else:
  neg_prompt = "Broken anatomy, Mutation, half body parts, two faces,more than two faces, distorted face"

image = pipe(prompt=prompt, image=img2, mask_image=masked, 
                negative_prompt=neg_prompt, height = height, width= width).images[0]

"""Converting the output image from PIL to array again"""

#converting the PIL image back to numpy array image
img3 = np.array(image)
print("The resultant image\n")
cv2.imshow(img3)