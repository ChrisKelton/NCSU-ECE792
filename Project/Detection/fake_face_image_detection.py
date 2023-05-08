import torch
import torchvision.transforms as transforms
from pathlib import Path
from cffn import CFFN
from celebrity_data_loader import CelebrityDataCFFN


def main():
    dataset_base_path = Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/Dataset")
    # We will be working with GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device : ', device)

    # Number of GPUs available.
    num_GPU = torch.cuda.device_count()
    print('Number of GPU : ', num_GPU)

    config_cffn = {'batch_size': 88,
                   'image_size': 64,
                   'n_channel': 3,
                   'num_epochs': 15,
                   'lr': 1e-3,
                   'growth_rate': 24,
                   'transition_layer_theta': 0.5,
                   'device': device,
                   'm_th': 0.5,
                   'n_combinations': 1000,
                   'seed': 999,
                   }
    celebrity_data_cffn = CelebrityDataCFFN(
        base_path=dataset_base_path,
        transform=transforms.Compose(
            [
                transforms.Resize(int(config_cffn["image_size"] * 1.1)),
                transforms.CenterCrop(config_cffn["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        ),
        seed=config_cffn["seed"],
        n_combinations=config_cffn["n_combinations"],
    )
    dataloader_cffn = torch.utils.data.DataLoader(
        dataset=celebrity_data_cffn,
        shuffle=True,
        batch_size=config_cffn["batch_size"],
        num_workers=2,
        drop_last=True,
        pin_memory=False,
    )
    cffn = CFFN(
        input_image_shape=(64, 64),
        growth_rate=config_cffn['growth_rate'],
        transition_layer_theta=config_cffn['transition_layer_theta'],
        learning_rate=config_cffn['lr'],
        m_th=config_cffn['m_th'],
    )
    epochs = 15
    loss = 0
    print("Starting Training Loop...")
    for epoch in range(epochs):
        batch_idx = 0
        for img0, img1, pair_indicator in dataloader_cffn:
            cffn.optimizer.zero_grad()
            _, img0_discriminative_features = cffn(img0)
            _, img1_discriminative_features = cffn(img1)
            cffn.loss_back_grad(
                img0_discriminative_features,
                img1_discriminative_features,
                pair_indicator,
            )
            loss = cffn.loss.item()

            if batch_idx % 50 == 0:
                print("[%d/%d][%d/%d]\tLoss: %.4f" % (epoch, epochs, batch_idx, len(dataloader_cffn), loss))
            batch_idx += 1
    a = 0


if __name__ == '__main__':
    main()