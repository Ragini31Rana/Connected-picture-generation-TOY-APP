# Connected-picture-generation-TOY-APP
Connected picture generation TOY APP is an app that allows user to generate and connect pictures just with a prompt using stable diffusion model from hugging face.
- Provide two images and a prompt as an input
- Press Generate and then see the magic
> Google colab was used for the developing part of the project.

> This repository only contains the backend files of the project.
- Notebook1.ipynb contains the code for picture generation from one image.
- Notebook2.ipynb contains the code for picture generation from two images.
- requirements.txt contains all the necessary modules to be installed for the project.

## Requirements
The following modules are required to be installed in colab:
> pip install -r requirements.txt
- diffusers
- transformers
- accelerate

> Changing the runtime type to GPU is the first necessary step to run the notebook if you are using the google colab.

## Tech
TOY app uses the following open source model from hugging face to work:

- [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2) - Model that can be used to generate and modify images based on text prompts.







