import os
import argparse
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

def generate_images(base_model_path, controlnet_path, control_image_dir, output_dir, category_txt_path, num_images):
    # Load model
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.to("cuda")

    # Read categories
    with open(category_txt_path, "r") as file:
        categories = [line.strip() for line in file]

    # Generate images for each category
    for category in categories:
        prompt = f"A satellite image of {category}"

        for control_image_name in os.listdir(control_image_dir):
            control_image_path = os.path.join(control_image_dir, control_image_name)
            control_image = load_image(control_image_path)

            for i in range(num_images):
                image = pipe(
                    prompt,
                    num_inference_steps=20,
                    generator=None,
                    image=control_image
                ).images[0]

                # Save the generated image
                output_path = os.path.join(output_dir, f"{category}_{os.path.splitext(control_image_name)[0]}_{i+1}.png")
                image.save(output_path)
                print(f"Image saved at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Image Generation using Stable Diffusion ControlNet.")
    parser.add_argument('--base_model_path', type=str, required=True, help='Path to the base model.')
    parser.add_argument('--controlnet_path', type=str, required=True, help='Path to the controlnet model.')
    parser.add_argument('--control_image_dir', type=str, default="./demo/control", help='Directory of control images.')
    parser.add_argument('--output_dir', type=str, default="./demo/output", help='Directory to save the generated images.')
    parser.add_argument('--category_txt_path', type=str, default="./demo/class.txt", help='Path to the category txt file.')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate per category.')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    generate_images(
        args.base_model_path,
        args.controlnet_path,
        args.control_image_dir,
        args.output_dir,
        args.category_txt_path,
        args.num_images
    )

if __name__ == "__main__":
    main()
