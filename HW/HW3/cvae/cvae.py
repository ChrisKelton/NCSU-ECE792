from pathlib import Path
from typing import Optional, Union, Tuple
from tqdm import tqdm

import pandas as pd
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

from Detection.celebrity_data_loader import CelebrityData
from dataset import CelebrityDataCVAE
from mlp_mnist.training_utils import Loss, training_plot_paths, save_loss_plot


class Encoder(nn.Module):
    layer0 = None
    layer1 = None
    layer2 = None
    layer3 = None
    out = None
    out_conv_flattened = None

    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 128,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: Union[int, Tuple[int, int]] = (2, 2),
        padding: Union[int, Tuple[int, int]] = 1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.latent_dim = latent_dim
        self.conv0 = nn.Conv2d(in_channels, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.LeakyReLU()

    def sampling(self):
        z_mu = torch.mean(self.out_conv_flattened, dim=1)
        z_var = torch.var(self.out_conv_flattened, dim=1)
        epsilon = torch.randn((self.out_conv_flattened.shape[0], self.latent_dim), device=self.device)

        return torch.add(z_mu.unsqueeze(1), torch.multiply(z_var.unsqueeze(1), epsilon))

    def forward(self, x):
        self.layer0 = self.activation(self.batch_norm0(self.conv0(x)))
        self.layer1 = self.activation(self.batch_norm1(self.conv1(self.layer0)))
        self.layer2 = self.activation(self.batch_norm2(self.conv2(self.layer1)))
        self.layer3 = self.activation(self.batch_norm3(self.conv3(self.layer2)))
        self.out = self.activation(self.conv4(self.layer3))

        self.out_conv_flattened = torch.flatten(self.out, start_dim=1)

        return self.sampling()


class Decoder(nn.Module):
    layer0 = None
    layer1 = None
    layer2 = None
    layer3 = None
    layer4 = None
    out = None

    def __init__(
        self,
        in_channels: int = 512,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: Union[int, Tuple[int, int]] = (2, 2),
        padding: Union[int, Tuple[int, int]] = 1,
    ):
        super().__init__()
        self.conv_trans0 = nn.ConvTranspose2d(in_channels, 256, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm0 = nn.BatchNorm2d(256)
        self.conv_trans1 = nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.conv_trans2 = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv_trans3 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.conv_trans4 = nn.ConvTranspose2d(32, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.conv_trans5 = nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))

        self.activation = nn.LeakyReLU()

    def forward(self, x):
        self.layer0 = self.activation(self.batch_norm0(self.conv_trans0(x)))
        self.layer1 = self.activation(self.batch_norm1(self.conv_trans1(self.layer0)))
        self.layer2 = self.activation(self.batch_norm2(self.conv_trans2(self.layer1)))
        self.layer3 = self.activation(self.batch_norm3(self.conv_trans3(self.layer2)))
        self.layer4 = self.activation(self.batch_norm4(self.conv_trans4(self.layer3)))
        self.out = self.activation(self.conv_trans5(self.layer4))  # Might be better to upsample from (33x33) to (64x64)

        return self.out


class CVAE(nn.Module):
    attrs_reshaped = None
    encoder_in = None
    encoder_out = None
    encoder_concat = None
    encoder_latent_linear = None
    decoder_in = None
    decoder_out = None
    conv_out = None
    activation_out = None
    resized_out = None

    def __init__(
      self,
      latent_dim: int = 128,
      n_attrs: int = 40,
      lr: float = 1e-3,
      img_size: int = 64,
      out_match_img_size: bool = True,
      device: Optional[torch.device] = None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.img_size = img_size
        self.out_match_img_size = out_match_img_size
        self.latent_dim = latent_dim
        self.n_attrs = n_attrs

        self.attr_linear = nn.Linear(self.n_attrs, self.img_size * self.img_size)

        self.encoder = Encoder(latent_dim=latent_dim, device=device).to(device)
        self.latent_linear = nn.Linear(self.latent_dim + self.n_attrs, 512 * 2 * 2)

        self.decoder = Decoder().to(device)
        self.conv_final = nn.Conv2d(32, 3, kernel_size=(3, 3), padding=1)
        self.activation_tanh = nn.Tanh()

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, attrs):
        self.attrs_reshaped = self.attr_transform(attrs)
        self.encoder_in = torch.concat((x, self.attrs_reshaped), dim=1)

        self.encoder_out = self.encoder(self.encoder_in)
        self.encoder_concat = torch.concat((self.encoder_out, attrs), dim=1)
        self.encoder_latent_linear = self.latent_linear(self.encoder_concat)
        self.decoder_in = torch.reshape(self.encoder_latent_linear, (self.encoder_latent_linear.shape[0], 512, 2, 2))

        self.decoder_out = self.decoder(self.decoder_in)
        self.conv_out = self.conv_final(self.decoder_out)
        self.activation_out = self.activation_tanh(self.conv_out)

        if self.out_match_img_size:
            self.resized_out = F.interpolate(self.activation_out, size=(self.img_size, self.img_size), mode="bilinear")
        else:
            self.resized_out = self.activation_out

        return self.resized_out

    def attr_transform(self, attr):
        attr = self.attr_linear(attr)
        attr = torch.reshape(attr, (attr.shape[0], 1, self.img_size, self.img_size))

        return attr

    def loss_back_grad(self, output, target, back_grad: bool = True):
        criterion = self.loss(output, target)
        if back_grad:
            criterion.backward()
            self.optimizer.step()

        return criterion


def run(
    celebrity_data_zip_file: Path,
    attr_file: Path,
    base_out_path: Path,
    epochs: int = 15,
    batch_size: int = 88,
    latent_dim: int = 128,
    pin_memory: bool = False,  # turn on when using GPU
    lr: float = 1e-3,
    seed: Optional[int] = None,
    chkp_freq: int = 1,
    img_size: int = 64,
):
    validation_set = True
    test_set = True
    img_transforms = transforms.Compose(
        [
            transforms.Resize(int(64 * 1.1)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: torch.div(torch.subtract(x, torch.min(x)), torch.subtract(torch.max(x), torch.min(x)))
            ),
        ]
    )
    celebrity_data = CelebrityDataCVAE(celebrity_data_zip_file, attr_file, img_transforms, seed, use_tmpdir=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pin_memory or torch.cuda.is_available():
        num_workers = 2
    else:
        num_workers = 0
    dataloader = DataLoader(
        dataset=celebrity_data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )

    cvae = CVAE(latent_dim=latent_dim, lr=lr, img_size=img_size)

    cvae_train_loss = Loss.init()
    cvae_val_loss = Loss.init()
    cvae_test_loss = Loss.init()
    epoch_tqdm = tqdm(total=epochs, position=0)
    print("Starting Training Loop...")
    for epoch in range(epochs):
        celebrity_data.update_dataset_type()
        for batch_idx, (img, attr) in enumerate(dataloader):
            cvae.optimizer.zero_grad()
            img = img.to(device)
            attr = attr.to(device)
            out = cvae(img, attr)
            loss = cvae.loss_back_grad(out, img)
            cvae_train_loss += loss.item()

            if batch_idx % 50 == 0:
                print("[%d/%d][%d/%d]\tTrain Loss: %.8f" % (
                epoch, epochs, batch_idx, len(dataloader), cvae_train_loss.current_loss))

        if validation_set:
            celebrity_data.update_dataset_type(val_set=True)
            cvae.eval()
            for batch_idx, (img, attr) in enumerate(dataloader):
                img = img.to(device)
                attr = attr.to(device)
                out = cvae(img, attr)
                loss = cvae.loss_back_grad(out, img, back_grad=False)
                cvae_val_loss += loss.item()

                if batch_idx % 50 == 0:
                    print("Epoch: '%d' [%d/%d]\tValidation Loss: %.8f" % (
                    epoch, batch_idx, len(dataloader), cvae_val_loss.current_loss))

            cvae.train()

        if test_set:
            celebrity_data.update_dataset_type(test_set=True)
            cvae.eval()
            for batch_idx, (img, attr) in enumerate(dataloader):
                img = img.to(device)
                attr = attr.to(device)
                out = cvae(img, attr)
                loss = cvae.loss_back_grad(out, img, back_grad=False)
                cvae_test_loss += loss.item()

                if batch_idx % 50 == 0:
                    print("Epoch: '%d' [%d/%d]\tTest Loss: %.8f" % (
                    epoch, batch_idx, len(dataloader), cvae_val_loss.current_loss))

            cvae.train()

        cvae_train_loss.update_for_epoch()
        cvae_val_loss.update_for_epoch()
        cvae_test_loss.update_for_epoch()
        epoch_tqdm.update(1)
        if ((epoch + 1) % chkp_freq == 0) or (epoch + 1 == epochs):
            (
                model_output_path,
                confusion_matrix_path,
                accuracy_plot_path,
                roc_curve_plot_path,
                cum_stats_csv_path,
                loss_plot_path,
            ) = training_plot_paths(base_out_path, epoch + 1, accuracy=False, confusion=False, roc_curve=False,
                                    cum_stats=False)
            torch.save(
                {
                    "CVAE_state_dict": cvae.state_dict(),
                    "CVAE_optimizer": cvae.optimizer.state_dict()
                },
                str(model_output_path),
            )
            save_loss_plot(cvae_train_loss.loss_vals_per_epoch, loss_plot_path, cvae_test_loss.loss_vals_per_epoch, cvae_val_loss.loss_vals_per_epoch)


def main():
    celebrity_data_zip_file = Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/realimages/celeba/img_align_celeba")
    attr_file = Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/realimages/celeba/img_align_celeba_attr.csv")
    base_out_path = Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/HW/HW3/cvae-models")
    epochs = 15
    batch_size = 88
    latent_dim = 128
    pin_memory = False
    lr = 1e-3
    seed = None
    chkp_freq = 1
    run(
        celebrity_data_zip_file=celebrity_data_zip_file,
        attr_file=attr_file,
        base_out_path=base_out_path,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        pin_memory=pin_memory,
        lr=lr,
        seed=seed,
        chkp_freq=chkp_freq,
    )


if __name__ == '__main__':
    main()
