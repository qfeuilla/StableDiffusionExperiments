import torch
from tqdm.auto import tqdm
import numpy as np
import PIL

def get_text_embeds(prompt, tokenizer, text_encoder, device="cuda"):
  # Tokenize text and get embeddings
  text_input = tokenizer(
      prompt, padding='max_length', max_length=tokenizer.model_max_length,
      truncation=True, return_tensors='pt')
  text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

  # Do the same for unconditional embeddings
  uncond_input = tokenizer(
      [''] * len(prompt), padding='max_length',
      max_length=tokenizer.model_max_length, return_tensors='pt')
  uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

  # Cat for final embeddings
  text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
  return text_embeddings

def produce_latents(text_embeddings, scheduler, scheduler_inpaint,
                    unet, device="cuda", height=512, width=512, 
                    num_inference_steps=50, guidance_scale=7.5, 
                    init_latents=None, mask=None, 
                    strength = 0.8, generator = None):
    current_scheduler = scheduler_inpaint if mask is not None else scheduler
    
    current_scheduler.set_timesteps(num_inference_steps)
    
    if init_latents is None:
        init_latents = torch.randn((text_embeddings.shape[0] // 2, unet.in_channels, \
                            height // 8, width // 8))

    init_latents = init_latents.to(device)
    init_latents_orig = init_latents

    t_start = 0

    if mask is not None:
        init_timestep = int(num_inference_steps * strength)
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = current_scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(np.array([timesteps]), dtype=torch.long, device=device)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=device)
        init_latents = current_scheduler.add_noise(init_latents, noise, timesteps)

        t_start = max(num_inference_steps - init_timestep, 0)

    latents = init_latents * (current_scheduler.sigmas[0] if mask is None else 1)

    for i, t in tqdm(enumerate(current_scheduler.timesteps[t_start:])):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        if mask is None:
            sigma = current_scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = current_scheduler.step(noise_pred, i if mask is None else t, latents)['prev_sample']

        if mask is not None:
            # masking
            init_latents_proper = current_scheduler.add_noise(init_latents_orig, noise, t)
            latents = (init_latents_proper * mask) + (latents * (1 - mask))

    return latents

def decode_img_latents(latents, vae):
  latents = 1 / 0.18215 * latents

  imgs = vae.decode(latents).sample

  imgs = (imgs / 2 + 0.5).clamp(0, 1)
  imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
  imgs = (imgs * 255).round().astype('uint8')
  np_images = [np.array(image) for image in imgs]
  
  return np_images

def prompt_to_img(prompts, tokenizer, text_encoder, 
                scheduler, scheduler_inpaint, vae,
                unet, device="cuda",
                height=512, width=512, num_inference_steps=50,
                guidance_scale=7.5, latents=None, 
                mask=None, strength=0.8, seed=None):

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
            
    if isinstance(prompts, str):
        prompts = [prompts]

    # Prompts -> text embeds
    text_embeds = get_text_embeds(prompts, tokenizer=tokenizer, text_encoder=text_encoder)

    # Text embeds -> img latents
    latents = produce_latents(
        text_embeds, scheduler=scheduler, scheduler_inpaint=scheduler_inpaint, 
        unet=unet, device=device, height=height, width=width, init_latents=latents,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, mask=mask, strength=strength)
    
    # Img latents -> imgs
    imgs = decode_img_latents(latents, vae=vae)

    return imgs