from multiclass_function import *
from models import VGGnet
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import os

class ClassificationTest:
    def __init__(self, model_type, dataset, epoch, batch_size):
        self.model_type = model_type
        self.dataset = dataset
        self.epoch = epoch
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.save_model_path = f"./results/{model_type}_{dataset}.pt"
        self.save_history_path = f"./results/{model_type}_history_{dataset}.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_data(self):
        transform_test = transforms.ToTensor()

        ds = self.dataset.lower()
        if ds == "stl10":
            test_DS = datasets.STL10(root='./data/', split="test", download=True, transform=transform_test)
        elif ds == "cifar10":
            test_DS = datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform_test)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        test_DL = torch.utils.data.DataLoader(test_DS, batch_size=self.batch_size, shuffle=False)
        return test_DL
    
    def load_model(self):
        loaded = torch.load(self.save_model_path, map_location=self.device)
        model = loaded["model"].to(device)
        model.eval()

        history = torch.load(self.save_history_path, map_location=self.device)
        loss_history = history["loss_history"]
        acc_history = history["acc_history"]
        epoch = history["epoch"]

        return model, loss_history, acc_history, epoch
    
    
    def Tester(self, model, test_DL):
        test_acc = Test(model, test_DL, self.criterion)
        print(f"Final Test Accuracy: {test_acc} %")

    
    def plot_loss_accuracy(self,loss_history, acc_history, epoch):
        plt.figure()
        plt.plot(range(1, epoch + 1), loss_history["train"], label="train")
        plt.plot(range(1, epoch + 1), loss_history["val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train, Val Loss")
        plt.grid()
        plt.legend()
        loss_path = os.path.join("/Users/wjsong/dev_ws/DeepLearningStudy/src/metrics/", f"{self.model_type}_{self.dataset}_loss.png")
        plt.savefig(loss_path)

        plt.figure()
        plt.plot(range(1, epoch + 1), acc_history["train"], label="train")
        plt.plot(range(1, epoch + 1), acc_history["val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train, Val Accuracy")
        plt.grid()
        plt.legend()
        loss_path = os.path.join("/Users/wjsong/dev_ws/DeepLearningStudy/src/metrics", f"{self.model_type}_{self.dataset}_accuracy.png")
        plt.savefig(loss_path)

def main():
    parser = argparse.ArgumentParser(description='Model Testing')
    parser.add_argument('--model_type', type=str, default='VGG')
    parser.add_argument('--dataset', type=str, default='STL10')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    clft = ClassificationTest(args.model_type, args.dataset, args.epoch, args.batch_size)

    test_DL = clft.load_dataset(args.dataset, args.batch_size)
    model, loss_history, acc_history, epoch = clft.load_model(args.model_type, args.dataset)

    clft.plot_loss_accuracy(loss_history, acc_history, epoch)

    clft.Tester(model, test_DL)


if __name__ == "__main__":
    main()