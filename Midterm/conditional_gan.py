from pathlib import Path
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def plot_grids_of_images(
    base_imgs_output_path: Path,
    base_models_output_path: Path,
    image: torch.Tensor,  # ground truth images
    sample_size: int = 25,
    latent_dim: int = 64,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # save out grids of images for trained models from different epochs and real images
    gridded_all_imgs_output_path = base_imgs_output_path / "gridded-all-imgs.png"
    gridded_real_imgs_epoch_1_and_epoch_100_output_path = base_imgs_output_path / "gridded-real-imgs-epoch-1-epoch-100.png"
    gridded_real_imgs_output_path = base_imgs_output_path / "gridded-real-imgs.png"
    gridded_g_imgs1_output_path = base_imgs_output_path / "gridded-generated-imgs1.png"
    gridded_g_imgs50_output_path = base_imgs_output_path / "gridded-generated-imgs50.png"
    gridded_g_imgs100_output_path = base_imgs_output_path / "gridded-generated-imgs100.png"
    individual_gridded_imgs_output_path = base_imgs_output_path / "individual-gridded-imgs.png"

    model_file1 = base_models_output_path / "CDCGAN--1.pth"
    checkpoint1 = torch.load(str(model_file1))
    generator1 = Generator().to(device)
    generator1.load_state_dict(checkpoint1["Generator"])
    generator1.eval()

    model_file50 = base_models_output_path / "CDCGAN--50.pth"
    checkpoint50 = torch.load(str(model_file50))
    generator50 = Generator().to(device)
    generator50.load_state_dict(checkpoint50["Generator"])
    generator50.eval()

    model_file100 = base_models_output_path / "CDCGAN--100.pth"
    checkpoint100 = torch.load(str(model_file100))
    generator100 = Generator().to(device)
    generator100.load_state_dict(checkpoint100["Generator"])
    generator100.eval()

    sample_size = 25

    enc = OneHotEncoder()
    enc.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])

    fixed_noise = generate_z_noise(sample_size, latent_dim).to(device)
    fixed_labels = torch.randint(low=0, high=10, size=(sample_size, 1))
    fixed_labels = torch.Tensor(enc.transform(fixed_labels).toarray()).to(device)
    noise_gen_input = torch.concat([fixed_noise, fixed_labels], dim=1).unsqueeze(-1).unsqueeze(-1).to(device)

    imgs = image[:sample_size]
    generated_imgs1 = generator1(noise_gen_input)
    generated_imgs50 = generator50(noise_gen_input)
    generated_imgs100 = generator100(noise_gen_input)

    real_img_grid = make_grid(imgs, padding=2, normalize=True, nrows=5)
    save_image(real_img_grid, gridded_real_imgs_output_path)

    g_imgs1_grid = make_grid(generated_imgs1, padding=2, normalize=True, nrows=5)
    save_image(g_imgs1_grid, gridded_g_imgs1_output_path)

    g_imgs50_grid = make_grid(generated_imgs50, padding=2, normalize=True, nrows=5)
    save_image(g_imgs50_grid, gridded_g_imgs50_output_path)

    g_imgs100_grid = make_grid(generated_imgs100, padding=2, normalize=True, nrows=5)
    save_image(g_imgs100_grid, gridded_g_imgs100_output_path)

    all_imgs = torch.stack((imgs, generated_imgs1, generated_imgs50, generated_imgs100), dim=0).reshape(sample_size * 4,
                                                                                                        1, 28, 28)
    real_imgs_epoch_0_epoch_100 = torch.stack((imgs, generated_imgs1, generated_imgs100), dim=0).reshape(
        sample_size * 3, 1, 28, 28)

    gridded_imgs = make_grid(all_imgs, padding=2, normalize=True, nrows=sample_size)
    save_image(gridded_imgs, gridded_all_imgs_output_path)

    gridded_real_imgs_epoch_1_epoch_100 = make_grid(real_imgs_epoch_0_epoch_100, padding=2, normalize=True,
                                                    nrows=sample_size)
    save_image(gridded_real_imgs_epoch_1_epoch_100, gridded_real_imgs_epoch_1_and_epoch_100_output_path)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0][0].axis('off')
    ax[0][0].imshow(real_img_grid.cpu().permute(1, 2, 0))
    ax[0][0].set_title("Real Images")

    ax[0][1].axis('off')
    ax[0][1].imshow(g_imgs1_grid.cpu().permute(1, 2, 0))
    ax[0][1].set_title("Generated Images Epoch 1")

    ax[1][0].axis('off')
    ax[1][0].imshow(g_imgs50_grid.cpu().permute(1, 2, 0))
    ax[1][0].set_title("Generated Images Epoch 50")

    ax[1][1].axis('off')
    ax[1][1].imshow(g_imgs100_grid.cpu().permute(1, 2, 0))
    ax[1][1].set_title("Generated Images Epoch 100")

    fig.tight_layout()
    fig.savefig(str(individual_gridded_imgs_output_path))
    fig.show()


# Ex.
# gen = Generator()
# gen.apply(weights_init)
def weights_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 0, 0.02)
        nn.init.normal_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=74, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.Tanh(),

            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.Tanh(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.Tanh(),

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            # nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.deconv(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        kernel_size = (4, 4)
        stride = (2, 2)
        padding = (0, 0)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x)


def compare_dicts(dict0, dict1) -> bool:
    results = []
    wrong_keys = []
    for key in list(dict0.keys()):
        val0 = dict0[key]
        val1 = dict1[key]
        if int(torch.sum(~torch.eq(val0, val1))) > 0:
            results.append(False)
            wrong_keys.append(key)

    if len(wrong_keys) > 0:
        for wrong_key, result in zip(wrong_keys, results):
            print(f"{wrong_key} = {result}")
        return False
    return True


def compare_list_of_dicts(list_) -> bool:
    results = []
    for idx in range(len(list_) - 1):
        list0 = list_[idx]
        list1 = list_[idx+1]
        results.append(compare_dicts(list0, list1))

    if not all(results):
        return False
    return True


# labels for real images are '1' and labels for fake images are '0'
def label_real(batch_size: int) -> torch.Tensor:
    return torch.ones(batch_size,)


def label_fake(batch_size: int) -> torch.Tensor:
    return torch.zeros(batch_size,)


# generate our random noise to sample from for our generator
def generate_z_noise(batch_size: int, latent_dim: int) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim)


# function to plot discriminator vs generator loss
def gan_loss_discriminator_vs_generator_plot(disc_loss: List[float], gen_loss: List[float], output_path: Path):
    plot_labels = ["discriminator", "generator"]
    x_epochs = np.arange(1, len(disc_loss) + 1)
    plt.plot(x_epochs, disc_loss, label=plot_labels[0])
    plt.plot(x_epochs, gen_loss, label=plot_labels[1])
    plt.title("Discriminator vs Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(str(output_path))
    plt.close()


# reshape one hot encoded labels as stated in problem (make square matrices containing one hot encoded values for each one hot encoded vector entry)
def reshape_labels_to_repeat_onehotencoding_into_square_matrix(labels: torch.Tensor, square_shape: Tuple[int, int] = (28, 28)) -> torch.Tensor:
    return torch.repeat_interleave(torch.repeat_interleave(labels, square_shape[0]), square_shape[1]).reshape(labels.shape[0], labels.shape[1], square_shape[0], square_shape[1])


def main():
    batch_size = 128
    latent_dim = 64
    epochs = 100
    sample_size = 25  # number of generated images to save per chkp_freq
    params = {
        "lr": 2e-4,
        "betas": (0.5, 0.999),
        "chkp_freq": 1,  # frequency of epochs to save outputs
    }
    compare_training_w_test_set: bool = True
    base_output_path = Path("./cgan/model")
    base_output_path.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type in ["cuda"]:
        n_workers = 2
        pin_memory = True
    else:
        n_workers = 0
        pin_memory = False
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = MNIST(root="./mnist-data", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    # img_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5])
    # ])
    # dataloader_train = DataLoader(MNIST(str(mnist_data_sets_base_path.absolute()), train=True, download=False, transform=img_transform), batch_size=batch_size, shuffle=True)
    # dataloader_test = DataLoader(
    #     dataset=mnist_test_dataset,
    #     shuffle=True,
    #     batch_size=batch_size,
    #     num_workers=n_workers,
    #     drop_last=True,
    #     pin_memory=pin_memory,
    # )

    # initialize generator & discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # create optimizers for generator & discriminator
    optim_g = torch.optim.Adam(generator.parameters(), lr=params["lr"], betas=params["betas"])
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=params["lr"], betas=params["betas"])

    # one hot encoding scheme for hand-written digits between '0'-'9'
    enc = OneHotEncoder()
    enc.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])

    # use binary cross-entropy loss
    criterion = nn.BCELoss().to(device)

    # use fixed noise for outputting samples during each epoch
    fixed_noise = generate_z_noise(sample_size, latent_dim).to(device)
    fixed_labels = torch.randint(low=0, high=10, size=(sample_size, 1))
    fixed_labels = torch.Tensor(enc.transform(fixed_labels).toarray()).to(device)
    noise_gen_input = torch.concat([fixed_noise, fixed_labels], dim=1).unsqueeze(-1).unsqueeze(-1).to(device)

    # ensure models are in training mode
    # initialize weights of models with normal distribution
    generator.train()
    generator.apply(weights_init)
    discriminator.train()
    discriminator.apply(weights_init)

    # output paths to retain pertinent information
    base_imgs_output_path = base_output_path / "images"
    base_models_output_path = base_output_path / "models"
    base_loss_output_path = base_output_path / "loss"
    base_imgs_output_path.mkdir(exist_ok=True, parents=True)
    base_models_output_path.mkdir(exist_ok=True, parents=True)
    base_loss_output_path.mkdir(exist_ok=True, parents=True)

    # keep track of generator & discriminator losses
    losses_generator = []
    losses_discriminator = []
    epoch_tqdm = tqdm(total=epochs, position=0)
    print("Starting Training Loop...")
    for epoch in range(epochs):
        loss_g = 0.
        loss_d_real = 0.
        loss_d_fake = 0.
        epoch_img_output_path = base_imgs_output_path / str(epoch + 1)
        epoch_img_output_path.mkdir(exist_ok=True, parents=True)
        model_output_path = base_models_output_path / f"CDCGAN--{epoch+1}.pth"
        loss_output_path = base_loss_output_path / f"CDCGAN-loss--{epoch+1}.png"

        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            # transform labels to one hot encoding scheme
            labels = torch.Tensor(enc.transform(np.column_stack(labels).reshape(-1, 1)).toarray()).to(device)
            # reshape labels for concatentation for discriminator input
            labels_reshaped = reshape_labels_to_repeat_onehotencoding_into_square_matrix(labels)
            #normalize images between [0, 1]
            imgs = imgs.to(device)
            img_min = torch.min(imgs)
            img_max = torch.max(imgs)
            imgs = torch.divide(torch.subtract(imgs, img_min), img_max)
            # generate noise input with concatenated one hot encoded vector labels
            z = generate_z_noise(batch_size, latent_dim).to(device)
            gen_input = torch.concat([z, labels], dim=1).unsqueeze(-1).unsqueeze(-1)

            # STEP 1, train discriminator
            # produce fake image outputs from generator
            gen_fake = generator(gen_input).detach()  # need to detach() in order to not accumulate gradients in generator

            # concatenate images with reshaped labels for input to discriminator
            gen_fake_cat = torch.concat([gen_fake, labels_reshaped], dim=1)
            img_real_cat = torch.concat([imgs, labels_reshaped], dim=1)

            # target decisions to be used in binary cross entropy loss
            label_real_ = label_real(batch_size).to(device)
            label_fake_ = label_fake(batch_size).to(device)

            # zero out weights in optimizer
            optim_d.zero_grad()

            # get outputs from discriminator using real images & calculate loss
            disc_out_real = discriminator(img_real_cat).view(-1)
            loss_real = criterion(disc_out_real, label_real_)

            # get outputs from generator using fake images & calculate loss
            disc_out_fake = discriminator(gen_fake_cat).view(-1)
            loss_fake = criterion(disc_out_fake, label_fake_)

            # calculate gradients
            loss_real.backward()
            loss_fake.backward()
            # backpropagate gradients through weights of discriminator model
            optim_d.step()

            # keep track of loss during epoch
            loss_d_real += loss_real
            loss_d_fake += loss_fake

            # STEP 2, train generator
            # generate fake images using same generator inputs from before
            gen_fake = generator(gen_input)
            # prep fake images to be inputted to discriminator
            gen_fake_cat = torch.concat([gen_fake, labels_reshaped], dim=1)

            # zero out gradients for generator to not accumulate gradients for our weights
            optim_g.zero_grad()

            # generate discriminator decisions on our fake generated images
            disc_out_fake_g_z = discriminator(gen_fake_cat).view(-1)
            # calculate loss for generator using real labels, b/c we want to try and trick the discriminator; therefore, we want these outputs to have been labelled as real
            loss_gen = criterion(disc_out_fake_g_z, label_real_)
            # calculate gradients
            loss_gen.backward()

            # backpropagate gradients through weights of generator model
            optim_g.step()

            # keep track of generator loss
            loss_g += loss_gen

            if (batch_idx + 1) % 50 == 0:
                print("Epoch: [%d/%d] [%d/%d]\tLoss_G: %.6f\tLoss_D_fake: %.6f\tLoss_D_real: %.6f" % (
                    epoch + 1, epochs, batch_idx, len(train_dataloader),
                    float(loss_g / batch_idx), float(loss_d_fake / batch_idx), float(loss_d_real / batch_idx)
                ))

        epoch_tqdm.update(1)
        # determine loss for generator & discriminator for epoch
        losses_generator.append(float(loss_g / len(train_dataloader)))
        losses_discriminator.append(float((loss_d_real / len(train_dataloader)) + (loss_d_fake / len(train_dataloader))) / 2)
        print("Loss_G: %.6f\tLoss_D: %.6f" % (losses_generator[-1], losses_discriminator[-1]))
        if (epoch + 1) % params["chkp_freq"] == 0:
            # save generator & discriminator models
            torch.save(
                {
                    "Generator": generator.state_dict(),
                    "Discriminator": discriminator.state_dict(),
                },
                str(model_output_path)
            )
            # plot generator vs discriminator loss
            gan_loss_discriminator_vs_generator_plot(losses_discriminator, losses_generator, loss_output_path)

            # produce images from generator for the epoch
            generated_imgs = generator(noise_gen_input)
            for idx in range(sample_size):
                generated_img = generated_imgs[idx]
                label_generated_img = int(torch.argmax(fixed_labels[idx]))
                img_output_path = epoch_img_output_path / f"label={label_generated_img}--{idx}.png"
                save_image(generated_img, img_output_path)


if __name__ == '__main__':
    main()
