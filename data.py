import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torchvision.datasets as datasets


class Data:

    def __init__(self, name, path=".\\data", batch_size=64):

        self.map = {
            "cifar10": datasets.CIFAR10,
            "mnist": datasets.MNIST
        }
        self.name = name
        if self.name not in self.map.keys():
            raise ValueError(f"Unused dataset: {name}")

        self.path = path
        self.batch_size = batch_size

        # At each index of the dataset is a tuple containing the input and output.
        self.dataset = self.get_dataset()
        self.shape = tuple(self.dataset[0][0].shape)

    @staticmethod
    def train_transform_fn(x):
        return 2 * x / 255 - 1

    @property
    def train_transform(self):
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32),
            v2.Lambda(self.train_transform_fn)
        ])
        return transform

    @staticmethod
    def visual_transform_fn(x):
        x = x.squeeze(0).permute((1, 2, 0)).clamp(-1.0, 1.0)
        x = 255 * (x + 1) / 2
        return x.cpu()

    @property
    def visual_transform(self):
        transform = v2.Compose([
            v2.Lambda(self.visual_transform_fn),
            v2.ToDtype(dtype=torch.uint8)
        ])
        return transform

    def get_dataset(self):
        dataset = self.map[self.name](self.path, train=True, download=True, transform=self.train_transform)
        return dataset

    def get_loader(self, num_workers=0):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=num_workers, persistent_workers=True, shuffle=True)

    def show_img(self, ncols=5, nrows=4):
        fig, ax = plt.subplots(nrows, ncols, figsize=(5, 5))
        fig.tight_layout()
        for i in range(nrows):
            for j in range(ncols):
                ax[i, j].imshow(self.visual_transform(self.dataset[i * ncols + j][0]))
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
        plt.show()


def main():

    data = Data("cifar10")
    #data.show_img()
    print(tuple(data.dataset[0][0].shape))

    return



if __name__ == "__main__":
    main()
