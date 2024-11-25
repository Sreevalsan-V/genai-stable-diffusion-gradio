## Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

## DEVELOPED BY: SREEVALSAN 
## REGISTER NUMBER:212223240158

### AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

### PROBLEM STATEMENT:
Generating high-quality images from textual descriptions is a powerful application of generative AI. The Stable Diffusion model, a text-to-image generation model, allows for the generation of realistic images based on user-provided prompts. The goal of this project is to develop a prototype application that utilizes the Stable Diffusion model for interactive image generation, enabling users to input prompts and adjust generation parameters like the number of steps and guidance scale to fine-tune the output images.

### DESIGN STEPS:

#### STEP 1:
Load the Stable Diffusion Model: Use the pre-trained Stable Diffusion model from the Hugging Face diffusers library.

#### STEP 2:
Process User Input: Capture the user input for the image prompt, number of steps, and guidance scale.

#### STEP 3:
Image Generation: Pass the user inputs to the Stable Diffusion model and generate the corresponding image.

#### STEP 4:
Gradio Interface: Create a Gradio interface where users can interact with the application by providing text prompts and adjusting generation parameters.

#### STEP 5:
Deploy the Application: Test and deploy the application to generate images based on various prompts.

### PROGRAM:
```py
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

pipe = load_model()

def generate_image(prompt, num_inference_steps=50, guidance_scale=7.5):
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

def main():
    interface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Textbox(label="Enter your prompt", placeholder="Describe the image you'd like to generate"),
            gr.Slider(10, 100, value=50, step=1, label="Number of Inference Steps"),
            gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
        ],
        outputs=gr.Image(label="Generated Image"),
        title="Stable Diffusion Image Generator"
    )
    interface.launch()

if __name__ == "__main__":
    main()

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/df4d169e-6454-4253-8f92-2f48be0b1ad6)


### RESULT:
The application will prompt the user to enter a description for the image, adjust the number of inference steps, and set the guidance scale.
Based on the user inputs, the Stable Diffusion model will generate an image that reflects the given prompt.
The generated image will be displayed in the Gradio interface.
