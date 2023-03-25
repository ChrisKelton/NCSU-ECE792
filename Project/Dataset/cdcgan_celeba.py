from typing import Tuple
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path


# Ex.
# gen = Generator()
# gen.apply(weights_init)
# apply normal distribution with mean of 0 and standard deviation of 0.02 to model weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CelebrityData(torch.utils.data.Dataset):
    def __init__(self,zip_file,transform=None) :
        self.zip_file = zip_file
        self.transform = transform

        # Create a ZipFile Object and load sample.zip in it
        with ZipFile(str(zip_file), 'r') as zipObj:
          # Extract all the contents of zip file in current directory
          zipObj.extractall()

        self.ListOfImages = zipObj.namelist()[1:]

    def __getitem__(self,index):
        image = Image.open(self.ListOfImages[index]).convert('RGB')

        if self.transform is not None:
          image = self.transform(image)

        return image

    def __len__(self):
        return len(self.ListOfImages)


class CelebrityDataCVAE(CelebrityData):
    def __init__(self, zip_file: Path, attr_file: Path, img_transform=None, seed=None):
        super().__init__(zip_file, img_transform)
        self.attrs = pd.read_csv(str(attr_file), index_col=0)

    def __getitem__(self, index):
        img_path = self.ListOfImages[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
          img = self.transform(img)

        # attr = torch.Tensor(self.attrs.loc[int(Path(img_path).with_suffix("").name)])
        attr = self.get_attr(int(Path(img_path).with_suffix("").name))

        return img, attr

    def get_attr(self, index):
        return torch.Tensor(self.attrs.loc[index])


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=104, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                               padding=(0, 0)),
            nn.Tanh(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2),
                               padding=(0, 0)),
            nn.Tanh(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.Tanh(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.Tanh(),

            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            # nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.deconv(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=43, out_channels=96, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(4, 4), stride=(3, 3), padding=(0, 0)),
            # nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x)


# labels for real images are '1' and labels for fake images are '0'
def label_real(batch_size: int) -> torch.Tensor:
    return torch.ones(batch_size, )


def label_fake(batch_size: int) -> torch.Tensor:
    return torch.zeros(batch_size, )


# generate our random noise to sample from for our generator
def generate_z_noise(batch_size: int, latent_dim: int) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim)


# function to plot discriminator loss vs generator loss
def gan_loss_discriminator_vs_generator_plot(disc_loss, gen_loss, output_path):
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


# reshape attributes into square matrices
def reshape_labels_to_repeat_attrs_into_square_matrix(
        labels: torch.Tensor,
        square_shape: Tuple[int, int] = (64, 64),
) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.repeat_interleave(
            labels, square_shape[0]
        ), square_shape[1]
    ).reshape(labels.shape[0], labels.shape[1], square_shape[0], square_shape[1])


def main(dataset_base_path: Path):
    output_dir = Path(Path("./models").absolute())
    output_dir.mkdir(exist_ok=True, parents=True)
    zip_file = dataset_base_path / "img_align_celeb.zip"
    attr_file = dataset_base_path / "img_align_celeba_attr.csv"

    # learning rate & beta1 of Adam optimizer from original DCGAN paper
    config = {
        "batch_size": 128,
        "epochs": 100,
        "latent_dim": 64,
        "lr": 2e-4,
        "betas": (0.5, 0.999),
        "chkp_freq": 1,  # frequency of epochs to save outputs
        "n_sampled_imgs": 25,
        "img_size": 64,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    celebrity_data = CelebrityDataCVAE(zip_file=zip_file,
                                       attr_file=attr_file,
                                       img_transform=transforms.Compose([
                                          # Important parts, above can be ignored
                                          transforms.Resize(int(config['img_size'] * 1.1)),
                                          transforms.CenterCrop(config['img_size']),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                               std=(0.5, 0.5, 0.5))
    ]))

    dataloader = torch.utils.data.DataLoader(dataset=celebrity_data,
                                             shuffle=True,
                                             batch_size=config['batch_size'],
                                             num_workers=2,
                                             drop_last=True,
                                             pin_memory=True)

    # use binary cross-entropy loss
    criterion = nn.BCELoss()

    # initialize generator & discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # create optimizers for generator & discriminator
    optim_g = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=config["betas"])
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr"], betas=config["betas"])

    # use fixed noise for outputting samples during each epoch
    fixed_noise = generate_z_noise(config["n_sampled_imgs"], config["latent_dim"]).to(device)
    attr_indices = torch.randint(low=1, high=len(celebrity_data), size=(config["n_sampled_imgs"], 1))
    fixed_attrs = torch.Tensor(config["n_sampled_imgs"], 40)
    for idx, attr_idx in enumerate(attr_indices):
      fixed_attrs[idx] = celebrity_data.get_attr(attr_idx.item()).to(device)
    noise_gen_input = torch.concat([fixed_noise, fixed_attrs.to(device)], dim=1).unsqueeze(-1).unsqueeze(-1).to(device)

    # ensure models are in training mode
    # initialize weights of models with normal distribution
    generator.train()
    generator.apply(weights_init)
    discriminator.train()
    discriminator.apply(weights_init)

    # output paths to retain pertinent information
    base_imgs_output_path = output_dir / "images"
    base_imgs_output_path.mkdir(exist_ok=True, parents=True)
    base_loss_output_path = output_dir / "loss"
    base_loss_output_path.mkdir(exist_ok=True, parents=True)
    base_models_output_path = output_dir / "models"
    base_models_output_path.mkdir(exist_ok=True, parents=True)

    # keep track of generator & discriminator losses
    losses_generator = []
    losses_discriminator = []
    epoch_tqdm = tqdm(total=config["epochs"], position=1)
    print("Starting Training...")
    for epoch in range(config["epochs"]):
        loss_g = 0.
        loss_d_real = 0.
        loss_d_fake = 0.
        epoch_img_output_path = base_imgs_output_path / str(epoch + 1)
        epoch_img_output_path.mkdir(exist_ok=True, parents=True)
        model_output_path = base_models_output_path / f"CDCGAN--{epoch + 1}.pth"
        loss_output_path = base_loss_output_path / f"CDCGAN-loss--{epoch + 1}.png"

        for batch_idx, (img, attr) in enumerate(dataloader):
            img = img.to(device)
            attr = attr.to(device)
            # reshape attributes for concatenation for discriminator input
            attr_reshaped = reshape_labels_to_repeat_attrs_into_square_matrix(attr)

            # generate noise input with concatenated attributes
            z = generate_z_noise(config["batch_size"], config["latent_dim"]).to(device)
            gen_input = torch.concat([z, attr], dim=1).unsqueeze(-1).unsqueeze(-1).to(device)

            # Step 1, train discriminator
            # produce fake image outputs from generator
            gen_fake = generator(gen_input).detach()

            # concatenate images with reshaped labels for input to discriminator
            gen_fake_cat = torch.concat([gen_fake, attr_reshaped], dim=1)
            img_real_cat = torch.concat([img, attr_reshaped], dim=1)

            # target decisions to be used in binary cross entropy loss
            label_real_ = label_real(config["batch_size"]).to(device)
            label_fake_ = label_fake(config["batch_size"]).to(device)

            # zero out weights in optimizer
            optim_d.zero_grad()

            # get outputs from discriminator using real images & calculate loss
            disc_out_real = discriminator(img_real_cat).view(-1)
            loss_real = criterion(disc_out_real, label_real_)

            # get outputs from discriminator using fake images & calculate loss
            disc_out_fake = discriminator(gen_fake_cat).view(-1)
            loss_fake = criterion(disc_out_fake, label_fake_)

            loss_D = 0.5 * (loss_fake + loss_real)
            loss_D.backward()
            optim_d.step()

            loss_d_real += loss_real
            loss_d_fake += loss_fake

            # Step 2, train generator
            optim_g.zero_grad()

            gen_fake = generator(gen_input)
            gen_fake_cat = torch.concat([gen_fake, attr_reshaped], dim=1)

            disc_out_fake_g_z = discriminator(gen_fake_cat).view(-1)

            loss_gen = criterion(disc_out_fake_g_z, label_real_)
            loss_gen.backward()

            optim_g.step()

            loss_g += loss_gen

            if (batch_idx + 1) % 50 == 0:
                print("Epoch: [%d/%d] [%d/%d]\tLoss_G: %.6f\tLoss_D_fake: %.6f\tLoss_D_real: %.6f" % (
                    epoch + 1, config["epochs"], batch_idx + 1, len(dataloader),
                    float(loss_g / batch_idx), float(loss_d_fake / batch_idx), float(loss_d_real / batch_idx)
                ))

        epoch_tqdm.update(1)
        losses_generator.append(float(loss_g / len(dataloader)))
        losses_discriminator.append(
            float(
                (loss_d_real / len(dataloader)) + (loss_d_fake / len(dataloader))
            ) / 2
        )
        print("Loss_G: %.6f\tLoss_D: %.6f" % (
            losses_generator[-1], losses_discriminator[-1]
        ))

        if (epoch + 1) % config["chkp_freq"] == 0:
            torch.save(
                {
                    "Generator": generator.state_dict(),
                    "Discriminator": discriminator.state_dict(),
                },
                str(model_output_path)
            )
            gan_loss_discriminator_vs_generator_plot(losses_discriminator, losses_generator, loss_output_path)

            generated_imgs = generator(noise_gen_input)
            for idx, gen_img in enumerate(generated_imgs):
                img_output_path = epoch_img_output_path / f"{idx}.png"
                save_image(gen_img, img_output_path)


if __name__ == '__main__':
    dataset_base_path = Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/realimages/celeba")
    main(dataset_base_path)
