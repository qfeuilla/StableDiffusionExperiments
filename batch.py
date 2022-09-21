from SD_functions import *
from image_utils import *

def image_batch(prompt, tokenizer, text_encoder, 
                scheduler, scheduler_inpaint, vae,
                unet, batch_size=2, n_batch=3, device="cuda",
                width=512, height=512, steps=50, 
                cgs=7.5, seed=None):
    outs = torch.tensor([])

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for _ in range(n_batch):
                outs = torch.cat([
                        torch.tensor(np.array(
                            prompt_to_img(
                                [prompt] * batch_size, tokenizer=tokenizer, 
                                text_encoder=text_encoder, seed=seed,
                                scheduler=scheduler, scheduler_inpaint=scheduler_inpaint,
                                unet=unet, vae=vae, device=device, width=width, height=height,
                                num_inference_steps=steps, guidance_scale=cgs
                            )
                        )), 
                        outs
                    ]
                )
    
    outs = outs.detach().cpu().numpy().astype(np.uint8)

    return image_grid(list(outs), n_batch, batch_size)
