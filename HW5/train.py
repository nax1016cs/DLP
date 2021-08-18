import os
import torch
import torch.nn as nn
import numpy as np
import copy
from torchvision.utils import save_image
from evaluator import EvaluationModel
from utils import save_checkpoint, Condition, sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, generator, discriminator, latent_size, epochs, lr):

    optimizerG = torch.optim.Adam(generator.parameters(), lr, betas = (0.5,0.99))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr, betas = (0.5,0.99))
    criterion = nn.BCELoss()
    evaluation_model = EvaluationModel()
    test_conditions = Condition("test.json").to(device)
    fz = sample(len(test_conditions), latent_size).to(device)

    best_testing_acc = 0.65

    for epoch in range(epochs):
        Total_gen_loss = 0
        Total_dis_loss = 0
        for i, (images, conditions) in enumerate(dataloader):
            generator.train()
            discriminator.train()
            images = images.to(device)
            conditions = conditions.to(device)
            batch_size = len(images)
            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)
            # train discriminator

            # feed real images
            optimizerD.zero_grad()
            prediction = discriminator(images, conditions)
            loss_real = criterion(prediction, real)

            # feed fake images
            z = sample(batch_size, latent_size).to(device)
            gen_imgs = generator(z, conditions)
            prediction = discriminator(gen_imgs.detach(), conditions)
            loss_fake = criterion(prediction, fake)
            dis_loss = loss_real + loss_fake
            dis_loss.backward()
            optimizerD.step()

            # train generator
            for _ in range(4):
                optimizerG.zero_grad()
                z = sample(batch_size, latent_size).to(device)
                gen_imgs = generator(z, conditions)
                prediction = discriminator(gen_imgs,conditions)
                gen_loss = criterion(prediction,real)
                gen_loss.backward()
                optimizerG.step()
           
            Total_gen_loss += gen_loss.item()
            Total_dis_loss += dis_loss.item()

        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            gen_imgs = generator(fz, test_conditions)
        testing_acc = evaluation_model.eval(gen_imgs, test_conditions)
        if testing_acc > best_testing_acc:
            best_testing_acc = testing_acc
            save_checkpoint("check_point/" + str(testing_acc) + ".pt", generator )
        print('epoch[\033[35m{:>4d}\033[00m/{:>4d}]  \033[32m Generator loss:\033[00m {:.6f} \033[34m Discriminator loss:\033[00m {:.6f} | \033[33mTest Acc:\033[00m {:.6f}'.format(
            epoch+1, epochs,  Total_gen_loss/len(dataloader), Total_dis_loss/len(dataloader), testing_acc))
        save_image(gen_imgs, os.path.join('results', f'epoch{epoch}.png'), nrow = 8, normalize = True)

