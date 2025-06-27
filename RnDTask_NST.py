import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

#model = models.vgg19(pretrained = True).features
#print(model)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_features = ['0','5','10','19','28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
image_size1_width = 512
image_size2_height = 682


loader = transforms.Compose(
    [
          transforms.Resize((image_size2_height, image_size1_width)),
          transforms.ToTensor(),
          # transforms.Normalize(mean=[], std[])
    ]
)

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

original_img = load_image("dhruv2.jpg")
style_img = load_image("mosaic.jpg")

model = VGG().to(device).eval()
#generated = torch.randn(original_img.shape, device=device, requires_grad=True)
generated = original_img.clone().requires_grad_(True)

total_steps = 6000
learning_rate = 0.03
alpha = 1
beta = 1000

optimizer = optim.Adam([generated], lr=learning_rate)

#original_img_features = model(original_img)
#style_features = model(style_img)

#for steps in range(total_steps):
#    generated_features = model(generated)


for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)
    
    style_loss = original_loss = 0

    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        #computing gram matrix
        G = gen_feature.view(channel, height*width).mm(
            gen_feature.view(channel, height*width).t()
        )

        A = style_feature.view(channel, height*width).mm(
            style_feature.view(channel, height*width).t()
        )

        style_loss += torch.mean((G-A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss 
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated, "generated.png")