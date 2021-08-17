import PIL
from PIL import Image

import torch
import json
from torchvision import transforms
from tqdm import tqdm


with open('objects.json', 'r') as f:
    objects = json.load(f)

with open('train.json', 'r') as f:
    labels = json.load(f)

trans = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


label_list = []
img_list = []
num_images = 6003
for idx in tqdm(range(num_images)):
    for sub_idx in range(3):
        key = f'CLEVR_train_{idx:06}_{sub_idx}.png'
        label_list.append(torch.LongTensor([objects[labels[key][-1]]]))
        img_list.append(trans(PIL.Image.open(f'images/{key}').convert("RGB")))

label_list = torch.stack(label_list)
img_list = torch.stack(img_list)

torch.save(label_list, 'labels.pth')
torch.save(img_list, 'images.pth')