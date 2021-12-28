import matplotlib.pyplot as plt
import numpy as np
import torch


def seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model, optimizer, path):
    checkpoints = {
        "model_dict": model.state_dict(),
        "optimizer_dict": optimizer.state_dict(),
    }
    torch.save(checkpoints, path)


def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path)["model_dict"])


def visualize(train_loss, val_loss, args):
    plt.plot(
        train_loss["epoch"],
        train_loss["train_loss"],
        label="Training loss",
        linestyle="-.",
    )
    plt.plot(
        val_loss["epoch"], val_loss["val_loss"], label="Validation loss", linestyle=":"
    )
    plt.legend()
    plt.label("Training vs validation loss")
    plt.savefig(args.visualize_dir)


if __name__ == "__main__":
    lst_data = np.array([1,2,3,4,100])
    lst_rev_distance = np.array([0.1,0.2,0.3,0.2,0.2])

    print(get_interpolate(lst_data, lst_rev_distance))