import os
import torch
from torch.utils.data import DataLoader

from dataloader import CLEVRDataset
from model import Generator, Discriminator, init_weight
from train import train
from evaluate import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_size = 100
condition_size = 200
epochs = 100
lr = 0.0002
batch_size = 64

def main(method):
    if method == "training":
        dataset_train = CLEVRDataset(img_path = 'images', json_path = os.path.join('dataset','train.json'))
        loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 2)
        generator = Generator(latent_size,condition_size).to(device)
        init_weight(generator)
        discrimiator = Discriminator((batch_size, batch_size, 3), condition_size).to(device)
        init_weight(discrimiator)
        train(loader_train,generator,discrimiator,latent_size,epochs,lr)

    elif method == "testing":
        generator = Generator(latent_size,condition_size).to(device)
        generator.load_state_dict(torch.load('check_point/0.7083333333333334.pt'))
        torch.random.manual_seed(823)
        score = evaluate(generator, "test.json", latent_size,"result1.png" )
        print("Score: ", score)
        torch.random.manual_seed(547)
        score = evaluate(generator, "new_test.json", latent_size, "result2.png")
        print("Score: ", score)

main("testing")
