# -*- coding: utf-8 -*-
"""Text_To_Image.ipynb


Original file is located at
    https://colab.research.google.com/drive/1Ve1zLJuUYMbXQCPHX5y3G21lRJABcz7I
"""

!pip uninstall torch torchvision torchaudio -y
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install diffusers transformers accelerate --upgrade
!pip install safetensors

import torch

from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

from huggingface_hub import login
login("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")

from diffusers import StableDiffusionPipeline
import torch

pipe=StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe=pipe.to("cuda")

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Create an input text field
prompt_input = widgets.Text(
    value='',
    placeholder='Enter your prompt...',
    description='Prompt:',
    disabled=False
)

# Create a button to trigger image generation
generate_button = widgets.Button(
    description='Generate Image',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to generate image',
    icon='camera' # (FontAwesome icons: https://fontawesome.com/icons?d=gallery&p=2)
)

# Output widget to display the generated image
output_widget = widgets.Output()

# Function to handle button click
def on_generate_button_clicked(b):
    with output_widget:
        clear_output(wait=True)
        user_prompt = prompt_input.value
        if user_prompt:
            print("Generating image...")

            image = pipe(user_prompt).images[0]

            plt.figure(figsize=(6, 6)) # Adjust figure size if needed
            plt.imshow(image)
            plt.axis("off")
            plt.title("Generated Image")
            plt.show()
        else:
            print("Please enter a prompt.")

# Link the button click to the function
generate_button.on_click(on_generate_button_clicked)

# Display the widgets
display(prompt_input, generate_button, output_widget)
