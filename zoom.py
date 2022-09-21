import torch
import numpy as np

from SD_functions import *
from image_utils import *

from PIL import Image
import cv2 as cv

import matplotlib.pyplot as plt

def zooming_step(prompt, tokenizer, text_encoder, 
                scheduler, scheduler_inpaint, vae,
                unet, device="cuda", previous_image=None,
                width=512, height=512, steps=50, 
                cgs=7.5, seed=None, zoom_speed=2):
    
    if previous_image is None:
        return prompt_to_img([prompt], seed=seed, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler, scheduler_inpaint=scheduler_inpaint, unet=unet, device=device, vae=vae, width=width, height=height, num_inference_steps=steps, guidance_scale=cgs)
    
    # for code length
    z = zoom_speed

    # processed_image = np.random.random((width, height, 3)) * 200
    reshaped_image = Image.fromarray(previous_image).resize(((width // z) * (z-1), (height // z) * (z-1)))
    reshaped_image = cv.copyMakeBorder(np.array(reshaped_image),(width // z) // 2,(height // z) // 2,(width // z) // 2,(height // z) // 2,cv.BORDER_REPLICATE)

    min_width, min_height = int((width / 2) * (1 - ((z-1)/z))), int((height / 2) * (1 - ((z-1)/z)))
    max_width, max_height = int(width - ((width / 2) * (1 - ((z-1)/z)))), int(height - ((height / 2) * (1 - ((z-1)/z))))
    
    # processed_image[min_width : max_width, min_height : max_height, :] = np.array(reshaped_image) 
    # processed_image = np.array(processed_image).astype(np.uint8)
    
    processed_image = np.array(reshaped_image).astype(np.uint8)
    processed_image = preprocess_image(Image.fromarray(processed_image)).to(device)
    
    # encode the init image into latents and scale the latents
    init_latent_dist = vae.encode(processed_image)
    init_latents = init_latent_dist.sample()
    init_latents = 0.18215 * init_latents

    mask = np.ones((width, height)) * 255
    mask[min_width : max_width, min_height : max_height] = 0
    mask = preprocess_mask(Image.fromarray(mask)).to(device)

    return prompt_to_img([prompt], seed=seed, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler, scheduler_inpaint=scheduler_inpaint, unet=unet, vae=vae, device=device, height=height, width=width, num_inference_steps=steps, guidance_scale=cgs, latents=init_latents, mask=mask)

def zoom_out(prompts,
            tokenizer, text_encoder,
            scheduler, scheduler_inpaint, vae,
            unet, device="cuda",
            skip=0, previous_buffer=[],
            width=512, height=512, steps=50,
            cgs=7.5, seed=None, zoom_speed=2):
    image_buffer = previous_buffer[:skip]

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for rep, prompt in prompts:
                for _ in range(rep):
                    if skip:
                        skip -= 1
                    else:
                        image_buffer.append(zooming_step(prompt, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler, scheduler_inpaint=scheduler_inpaint, unet=unet, vae=vae, device=device, previous_image=image_buffer[-1] if len(image_buffer) else None, width=width, height=height, steps=steps,cgs=cgs, seed=seed, zoom_speed=zoom_speed)[0])
                        plt.imshow(image_buffer[-1])
                        plt.show()
    return image_buffer