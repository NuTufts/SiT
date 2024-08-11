import os,sys
from train_lar import update_ema


for x, y in loader:
    x = x.to(device)  # image
    y = y.to(device)  # class label
    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    model_kwargs = dict(y=y)
    loss_dict = transport.training_losses(model, x, model_kwargs)
    loss = loss_dict["loss"].mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    update_ema(ema, model)
