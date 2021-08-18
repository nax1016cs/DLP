import torch
import torch.nn as nn


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, img_shape, condition_size, ndf = 64, nc = 4):
        super(Discriminator, self).__init__()
        self.H,self.W,self.C = img_shape
        self.condition = nn.Sequential(
            nn.Linear(24,self.H*self.W*1),
            nn.LeakyReLU()
        )
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x , c):
        c = self.condition(c).view(-1,1,self.H,self.W)
        out = torch.cat((x,c),dim = 1) 
        out = self.main(out)
        out = out.view(-1)
        return out

class Generator(nn.Module):
    def __init__(self, latent_size, condition_size , ngf = 64, nc = 3):
        super(Generator,self).__init__()
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.condition = nn.Sequential(
            nn.Linear(24, condition_size),
            nn.ReLU()
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d( latent_size + condition_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self,z,c):
        z = z.view(-1, self.latent_size, 1, 1)
        c = self.condition(c).view(-1, self.condition_size, 1 , 1)
        out = torch.cat((z,c),dim = 1)
        out = self.main(out)
        return out