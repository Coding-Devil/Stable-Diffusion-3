import os
import random
import uuid

import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler, AutoencoderKL
from huggingface_hub import snapshot_download

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

model_path = snapshot_download(
    repo_id="stabilityai/stable-diffusion-3-medium", 
    revision="refs/pr/26",
    repo_type="model", 
    ignore_patterns=["*.md", "*..gitattributes"],
    local_dir="stable-diffusion-3-medium",
    token=huggingface_token, # yeni bir token-id yazın.
    )

DESCRIPTION = """# Stable Diffusion 3"""
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = False
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1536"))
USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@spaces.GPU(enable_queue=True)
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 7,
    randomize_seed: bool = False,
    num_inference_steps=30,
    NUM_IMAGES_PER_PROMPT=1,
    use_resolution_binning: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    pipe.to(device)
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)
    
    #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        output_type="pil",
    ).images

    return output


examples = [
    "A red sofa on top of a white building.",
    "A cardboard which is large and sits on a theater stage.",
    "A painting of an astronaut riding a pig wearing a tutu holding a pink umbrella.",
    "Studio photograph closeup of a chameleon over a black background.",
    "Closeup portrait photo of beautiful goth woman, makeup.",
    "A living room, bright modern Scandinavian style house, large windows.",
    "Portrait photograph of an anthropomorphic tortoise seated on a New York City subway train.",
    "Batman, cute modern Disney style, Pixar 3d portrait, ultra detailed, gorgeous, 3d zbrush, trending on dribbble, 8k render.",
    "Cinnamon bun on the plate, watercolor painting, detailed, brush strokes, light palette, light, cozy.",
    "A lion, colorful, low-poly, cyan and orange eyes, poly-hd, 3d, low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition.",
    "Long exposure photo of Tokyo street, blurred motion, streaks of light, surreal, dreamy, ghosting effect, highly detailed.",
    "A glamorous digital magazine photoshoot, a fashionable model wearing avant-garde clothing, set in a futuristic cyberpunk roof-top environment, with a neon-lit city background, intricate high fashion details, backlit by vibrant city glow, Vogue fashion photography.",
    "Masterpiece, best quality, girl, collarbone, wavy hair, looking at viewer, blurry foreground, upper body, necklace, contemporary, plain pants, intricate, print, pattern, ponytail, freckles, red hair, dappled sunlight, smile, happy."

]

css = '''
.gradio-container{max-width: 1000px !important}
h1{text-align:center}
'''
with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML(
            """
            <h1 style='text-align: center'>
            Stable Diffusion 3 Medium 
            </h1>
            """
        )
            gr.HTML(
                """
               
                """
        )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", elem_id="gallery", show_label=False)
    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True)
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                value = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
                visible=True,
            )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )

        steps = gr.Slider(
            label="Steps",
            minimum=0,
            maximum=60,
            step=1,
            value=30,
        )
        number_image = gr.Slider(
            label="Number of Image",
            minimum=1,
            maximum=4,
            step=1,
            value=4,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=10,
                step=0.1,
                value=7.0,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result],
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
            steps,
            number_image,
        ],
        outputs=[result],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue().launch()