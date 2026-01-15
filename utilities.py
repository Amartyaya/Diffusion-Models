import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


from ddpm import DDPM
from data import Data
from unet import UNet


# torch.manual_seed(2005)


def visualize_forward(ddpm, data, num_steps=10):

    assert num_steps <= ddpm.timesteps

    idx = random.randint(0, len(data.dataset) - 1)
    img_data = data.dataset[idx][0].unsqueeze(0).to(device=ddpm.device, dtype=ddpm.dtype)
    stepsize = ddpm.timesteps // num_steps

    _, ax = plt.subplots(1, num_steps+1)
    vtransform = data.visual_transform
    for t in range(num_steps+1):
        xt, _ = ddpm.forward_sample(img_data, torch.tensor([t * stepsize]).to(device=ddpm.device))
        ax[t].imshow(vtransform(xt.squeeze(0)))
        ax[t].set_xticks([])
        ax[t].set_yticks([])
    plt.show()

def train_one_epoch(model, dataloader, ddpm, optimizer):

    last_loss = 0.0

    model.train()

    for batch_img, _ in dataloader:

        batch_img = batch_img.to(ddpm.device)
        batch_timestep = torch.randint(1, ddpm.timesteps+1, (batch_img.shape[0],)).to(ddpm.device)

        optimizer.zero_grad()

        xt, eps = ddpm.forward_sample(batch_img, batch_timestep)
        out = model(xt, batch_timestep)
        loss = F.mse_loss(out, eps)
        loss.backward()

        optimizer.step()

        with torch.inference_mode():
            last_loss = loss.item()
            print(last_loss)

    return last_loss

@torch.inference_mode()
def display_generated(model, ddpm, data):
    x0 = ddpm.generate(model, (1,) + data.shape)
    x0 = data.visual_transform(x0.squeeze(0))
    plt.imshow(x0)
    plt.show()

def train(model, data, ddpm, optimizer, scheduler=None, epochs=1000, num_workers=0):

    loader = data.get_loader(num_workers)

    loss_list = []

    for epoch in range(epochs):

        if scheduler is not None:
            print(f"\nEpoch: {epoch + 1} | Learning Rate: {scheduler.get_last_lr()[0]}")
        else:
            print(f"\nEpoch: {epoch + 1}")

        loss = train_one_epoch(model, loader, ddpm, optimizer)
        loss_list.append(loss)

        if scheduler is not None:
            scheduler.step()

        model.eval()

        display_generated(model, ddpm, data)

    plt.plot(loss_list)
    plt.show()

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    ddpm_dict = {
        "beta_start": 1e-4,
        "beta_end": 2e-2,
        "timesteps": 512,
        "device": device,
        "dtype": dtype
    }

    data_dict = {
        "name": "mnist",
        "path": "data",
        "batch_size": 1024
    }

    ddpm = DDPM(**ddpm_dict)
    data = Data(**data_dict)

    unet_dict = {
        "input_channels": data.shape[0],
        "hid_channels": 16,
        "num_blocks": 3,
        "emb_dim": 512
    }

    unet = UNet(**unet_dict).to(dtype=ddpm.dtype, device=ddpm.device)

    # visualize_forward(ddpm, data)

    # print(sum([param.numel() for param in unet.parameters()]))
    #
    # optimizer = optim.Adam(unet.parameters(), lr=1e-3)
    #
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    #
    # train(unet, data, ddpm, optimizer, scheduler=scheduler, epochs=5, num_workers=2)

    ddpm.interpolate(data, 0, 1, unet)

    return



if __name__ == "__main__":
    main()
