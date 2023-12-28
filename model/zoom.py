from helpers import *
from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import numpy as np
import torch
import os
import time
import boto3 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

access_key = os.environ.get('AWS_ACCESS_KEY_ID')
secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
region = os.environ.get('AWS_DEFAULT_REGION')

inpaint_model_list = [
    "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-inpainting",
    "parlance/dreamlike-diffusion-1.0-inpainting",
    "ghunkins/stable-diffusion-liberty-inpainting",
    "ImNoOne/f222-inpainting-diffusers"
]
default_prompt = "A psychedelic jungle with trees that have glowing, fractal-like patterns, Simon stalenhag poster 1920s style, street level view, hyper futuristic, 8k resolution, hyper realistic"
default_negative_prompt = "frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur"


def zoom(
    model_id,
    prompts_array,
    negative_prompt,
    num_outpainting_steps,
    guidance_scale,
    num_inference_steps,
    custom_init_image
):
    prompts = {}
    for x in prompts_array:
        try:
            key = int(x[0])
            value = str(x[1])
            prompts[key] = value
        except ValueError:
            pass
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)
    pipe = pipe.to("cuda")

    def no_check(images, **kwargs):
        return images, False
    pipe.safety_checker = no_check
    pipe.enable_attention_slicing()
    g_cuda = torch.Generator(device='cuda')

    height = 512
    width = height

    current_image = Image.new(mode="RGBA", size=(height, width))
    mask_image = np.array(current_image)[:, :, 3]
    mask_image = Image.fromarray(255-mask_image).convert("RGB")
    current_image = current_image.convert("RGB")
    if (custom_init_image):
        current_image = custom_init_image.resize(
            (width, height), resample=Image.LANCZOS)
    else:
        init_images = pipe(prompt=prompts[min(k for k in prompts.keys() if k >= 0)],
                           negative_prompt=negative_prompt,
                           image=current_image,
                           guidance_scale=guidance_scale,
                           height=height,
                           width=width,
                           mask_image=mask_image,
                           num_inference_steps=num_inference_steps)[0]
        current_image = init_images[0]
    mask_width = 128
    num_interpol_frames = 30

    all_frames = []
    all_frames.append(current_image)

    for i in range(num_outpainting_steps):
        print('Outpaint step: ' + str(i+1) +
              ' / ' + str(num_outpainting_steps))

        prev_image_fix = current_image

        prev_image = shrink_and_paste_on_blank(current_image, mask_width)

        current_image = prev_image

        # create mask (black image with white mask_width width edges)
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255-mask_image).convert("RGB")

        # inpainting step
        current_image = current_image.convert("RGB")
        images = pipe(prompt=prompts[max(k for k in prompts.keys() if k <= i)],
                      negative_prompt=negative_prompt,
                      image=current_image,
                      guidance_scale=guidance_scale,
                      height=height,
                      width=width,
                      # generator = g_cuda.manual_seed(seed),
                      mask_image=mask_image,
                      num_inference_steps=num_inference_steps)[0]
        current_image = images[0]
        current_image.paste(prev_image, mask=prev_image)

        # interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
        for j in range(num_interpol_frames - 1):
            interpol_image = current_image
            interpol_width = round(
                (1 - (1-2*mask_width/height)**(1-(j+1)/num_interpol_frames))*height/2
            )
            interpol_image = interpol_image.crop((interpol_width,
                                                  interpol_width,
                                                  width - interpol_width,
                                                  height - interpol_width))

            interpol_image = interpol_image.resize((height, width))

            # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
            interpol_width2 = round(
                (1 - (height-2*mask_width) / (height-2*interpol_width)) / 2*height
            )
            prev_image_fix_crop = shrink_and_paste_on_blank(
                prev_image_fix, interpol_width2)
            interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

            all_frames.append(interpol_image)
        all_frames.append(current_image)
        # interpol_image.show()
    video_file_name = "infinite_zoom_" + str(time.time())
    fps = 30
    save_path = video_file_name + ".mp4"
    start_frame_dupe_amount = 15
    last_frame_dupe_amount = 15

    write_video(save_path, all_frames, fps, False,
                start_frame_dupe_amount, last_frame_dupe_amount)
    

    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    bucket_name = "pius-upwork-bucket"
    
    s3_client.upload_file(save_path, bucket_name, video_file_name )
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{video_file_name}"
    
    
    
    return s3_url

