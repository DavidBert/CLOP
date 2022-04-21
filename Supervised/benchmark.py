import os
import argparse
from sklearn import datasets
import torch
from tqdm import tqdm
import pandas as pd
from models import VGG9, VGG11, MNISTClassifier
from train import train, train_mixup, test
from data_utils import get_stl10_loaders, get_imagenet_loaders, get_mnist_usps_loaders


def train_and_evaluate(model, trainloader, testloader, learning_rate, epochs, regul):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if regul == "mixup":
        train_accuracy = train_mixup(
            model, optimizer, trainloader, epochs=epochs, scheduler=scheduler, alpha=0.2
        )
    else:
        train_accuracy = train(
            model, optimizer, trainloader, epochs=epochs, scheduler=scheduler
        )
    test_accuracy = test(model=model, testloader=testloader)
    return train_accuracy, test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist_usps", help="dataset")
    parser.add_argument(
        "--dataset_folder", type=str, default="data", help="dataset folder"
    )
    parser.add_argument("--epochs", type=int, default=int(90), help="nb epochs")
    parser.add_argument(
        "--nb_runs", type=int, default=int(10), help="number of runs by config"
    )

    args = parser.parse_args()
    dataset = args.dataset

    assert dataset in {
        "mnist_usps",
        "stl10",
        "imagenette",
    }, f'specified dataset "{dataset}" is not supported'

    dataset_folder = args.dataset_folder
    if dataset == "mnist_usps":
        trainloader, testloader = get_mnist_usps_loaders(
            root_path=dataset_folder,
            batch_size=64,
            shuffle=True,
            num_workers=8,
        )
        MODEL = MNISTClassifier

    if dataset == "stl10":
        trainloader, testloader = get_stl10_loaders(
            root_path="/data/david.bertoin/data",
            batch_size=64,
            shuffle=True,
            num_workers=8,
        )
        MODEL = VGG9

    elif dataset == "imagenette":
        trainloader, testloader = get_imagenet_loaders(
            root_path="/data/david.bertoin/data/imagenette2",
            batch_size=64,
            shuffle=True,
            num_workers=8,
        )
        MODEL = VGG11

    epochs = args.epochs
    nb_runs = args.nb_runs
    learning_rate = 5e-4
    results = pd.DataFrame(
        columns=["regul", "train_accuracy", "test_accuracy", "run_id"]
    )

    for i in tqdm(range(nb_runs), desc="run_loop", leave=False):
        for regul in ["No_regul", "batch_norm", "clop", "dropout", "mixup"]:
            model = MODEL(regul=regul).cuda()
            train_accuracy, test_accuracy = train_and_evaluate(
                model, trainloader, testloader, learning_rate, epochs, regul
            )
            results = results.append(
                {
                    "run_id": i,
                    "regul": regul,
                    "test_accuracy": test_accuracy,
                    "train_accuracy": train_accuracy,
                },
                ignore_index=True,
            )

    outdir = f"./results/supervised"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    path = os.path.join(outdir, f"{dataset}.csv")
    results.to_csv(path)
