import os
import torch
import torch.nn as nn
import numpy as np
import copy
import json

from torchvision.utils import save_image
from evaluator import EvaluationModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Condition():
    with open(os.path.join('dataset', 'objects.json'), 'r') as file:
        classes = json.load(file)

    with open(os.path.join('dataset','test.json')) as file:
        test_conditions_list = json.load(file)

    labels = torch.zeros(len(test_conditions_list),len(classes))
    for i in range(len(test_conditions_list)):
        for condition in test_conditions_list[i]:
            labels[i, int(classes[condition])] = 1.

    return labels

def sample(batch_size, z_dim):
    return torch.randn(batch_size,z_dim)

def train(dataloader, generator, discriminator, z_dim, epochs, lr):

    optimizerG = torch.optim.Adam(generator.parameters(), lr, betas = (0.5,0.99))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr, betas = (0.5,0.99))
    criterion = nn.BCELoss()
    evaluation_model = EvaluationModel()
    test_conditions = Condition().to(device)
    fz = sample(len(test_conditions), z_dim).to(device)

    best_score = 0.5

    for epoch in range(epochs):
        Total_gen_loss = 0
        Total_dis_loss = 0
        for i, (images, conditions) in enumerate(dataloader):
            generator.train()
            discriminator.train()
            images = images.to(device)
            conditions = conditions.to(device)
            batch_size=len(images)

            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)
            """
            train discriminator
            """
            optimizerD.zero_grad()
            # for real images
            prediction = discriminator(images, conditions)
            loss_real = criterion(prediction, real)
            # for fake images
            z = sample(batch_size, z_dim).to(device)
            gen_imgs = generator(z, conditions)
            prediction = discriminator(gen_imgs.detach(), conditions)
            loss_fake = criterion(prediction, fake)
            # bp
            dis_loss = loss_real + loss_fake
            dis_loss.backward()
            optimizerD.step()

            """
            train generator
            """
            for _ in range(4):
                optimizerG.zero_grad()
                z = sample(batch_size, z_dim).to(device)
                gen_imgs = generator(z, conditions)
                prediction = discriminator(gen_imgs,conditions)
                gen_loss = criterion(prediction,real)
                gen_loss.backward()
                optimizerG.step()

            print(f'epoch [{epoch}] Image: {i}/{len(dataloader)}  Gen_loss: {gen_loss.item():.3f}  Dis_loss: {dis_loss.item():.3f}')
            Total_gen_loss += gen_loss.item()
            Total_dis_loss += dis_loss.item()

        # evaluate
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            gen_imgs = generator(fz, test_conditions)
        score = evaluation_model.eval(gen_imgs, test_conditions)
        if score > best_score:
            best_score = score
            best_model = copy.deepcopy(generator.state_dict())
            torch.save(best_model,os.path.join('check_point',f'epoch{epoch}_score{score:.2f}.pt'))
        print(f'Total_gen_loss: {Total_gen_loss}  Total_dis_loss: {Total_dis_loss} Testing score: {score:.2f}')
        print('='*80)
        save_image(gen_imgs, os.path.join('results', f'epoch{epoch}.png'), nrow = 8, normalize = True)
