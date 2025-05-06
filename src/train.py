from multiclass_function import *
from models.VGGnet import VGG
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import argparse

class ClassificationTrain:
    def __init__(self,  model_type, dataset, epoch, batch_size, lr, lr_step, lr_gamma):
        self.model_type = model_type
        self.dataset = dataset
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.train_ratio = 0.8
        self.criterion = nn.CrossEntropyLoss()
        self.save_model_path = f"/Users/wjsong/dev_ws/DeepLearningStudy/src/results/{self.model_type}_{self.dataset}.pt"
        self.save_history_path = f"/Users/wjsong/dev_ws/DeepLearningStudy/src/results/{self.model_type}_history_{self.dataset}.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_uint8(self, x):
        return (255*x).type(torch.uint8)
    
    def load_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])

        transform_test = transforms.ToTensor()

        ds = self.dataset.lower()

        if ds == "stl10":
            train_DS = datasets.STL10(root="/Users/wjsong/dev_ws/DeepLearningStudy/src/data/", split="train", download=True, transform=transform_train)
        elif ds == "cifar10":
            train_DS = datasets.CIFAR10(root="/Users/wjsong/dev_ws/DeepLearningStudy/src/data/", train=True, download=True, transform=transform_train)
        else:
            raise ValueError(f"Unsupported dataset: {self.data}")
        
        NoT = int(len(train_DS) * self.train_ratio); NoV = len(train_DS) - NoT
        train_DS, val_DS = torch.utils.data.random_split(train_DS, [NoT, NoV])
        val_DS.transform = transform_test

        train_DL = torch.utils.data.DataLoader(train_DS, batch_size = self.batch_size, shuffle = True)
        val_DL = torch.utils.data.DataLoader(val_DS, batch_size = self.batch_size, shuffle = False)
        return train_DL, val_DL
    
    def trainer(self, train_DL, val_DL):
        if self.model_type.lower() == "vgg":
            model = VGG(cfg="E", batch_norm=True, num_classes=10).to(device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        Train(model, train_DL, val_DL, self.criterion,
              LR=self.lr, LR_STEP=self.lr_step, LR_GAMMA=self.lr_gamma,
              EPOCH=self.epoch, BATCH_SIZE=self.batch_size, TRAIN_RATIO=self.train_ratio,
              save_model_path=self.save_model_path,
              save_history_path=self.save_history_path)
    

def main():
    parser = argparse.ArgumentParser(description='Trainer')

    parser.add_argument('--model_type', type=str, default="VGG")
    parser.add_argument('--dataset', type=str, default="STL10")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_step', type=int, default=3)
    parser.add_argument('--lr_gamma', type=float, default=0.9)

    args = parser.parse_args()

    clft = ClassificationTrain(args.model_type, args.dataset, args.epoch, args.batch_size, args.lr, args.lr_step, args.lr_gamma)
    
    train_DL, val_DL = clft.load_data()
    clft.trainer(train_DL, val_DL)


if __name__ == "__main__":
    main()