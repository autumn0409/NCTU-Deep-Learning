import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import torch.nn.functional as F

from models import Generator, Discriminator
from dataset import TrainingDataset, TestingDataset
from evaluator import evaluation_model

import numpy as np
from tqdm import tqdm
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = evaluation_model()

num_epochs = 400
batch_size = 128
lr_D = 0.0002
lr_G = 0.0002

# Size of z latent vector (i.e. size of generator input)
latent_size = 104

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

d_aux_weight = 48


# Weights
def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def g_aux_weight(iter):
    max_w = d_aux_weight / 2
    return min(max_w, iter / (1000 / max_w))


# Data
train_dataset = TrainingDataset(transform=transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
test_dataset = TestingDataset('test.json')

# Create the dataLoader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, 32, shuffle=False)

# Initialize the models
netG = Generator(latent_size).to(device)
netG.apply(weights_init)

netD = Discriminator().to(device)
netD.apply(weights_init)

# Initialize loss function
adversarial_loss = nn.BCELoss()
classification_loss = nn.BCEWithLogitsLoss()

# Setup Adam optimizers
optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))

# Fixed noise
fixed_noise = Variable(torch.cuda.FloatTensor(
    np.random.normal(0, 1, (32, latent_size))))

# Training Loop
history = {"loss_G": [], "loss_D": [], "test_acc": []}
iter = 0
highest_test_acc = 0

for epoch in range(1, num_epochs + 1):
    pbar = tqdm(total=len(train_loader), unit=' batches',  ascii=True)
    pbar.set_description("({}/{})".format(epoch, num_epochs))

    for batch_idx, (real_img, condition) in enumerate(train_loader):
        real_img, condition = real_img.to(device), condition.to(device)
        b_size = real_img.size(0)
        iter += 1

        # Adversarial ground truths
        real = torch.full((b_size, 1), 1., device=device)
        fake = torch.full((b_size, 1), 0., device=device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real images
        dis_output, aux_output = netD(real_img)
        D_x = dis_output.mean().item()
        d_real_loss = adversarial_loss(
            dis_output, real) + classification_loss(aux_output, condition) * d_aux_weight
        d_real_loss.backward()

        # Loss for fake images
        z = Variable(torch.cuda.FloatTensor(
            np.random.normal(0, 1, (b_size, latent_size))))
        fake_img = netG(z, condition)
        dis_output, _ = netD(fake_img.detach())
        D_G_before = dis_output.mean().item()
        d_fake_loss = adversarial_loss(dis_output, fake)
        d_fake_loss.backward()

        # Net Loss for the discriminator
        D_loss = d_real_loss + d_fake_loss
        history['loss_D'].append(D_loss.item())
        # Update parameters
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Classify all fake batch with D
        z = Variable(torch.cuda.FloatTensor(
            np.random.normal(0, 1, (b_size, latent_size))))
        gen_img = netG(z, condition)
        dis_output, aux_output = netD(gen_img)
        D_G_after = dis_output.mean().item()
        G_loss = adversarial_loss(
            dis_output, real) + classification_loss(aux_output, condition) * g_aux_weight(iter)
        history['loss_G'].append(G_loss.item())
        # Calculate gradients
        G_loss.backward()
        # Update parameters
        optimizer_G.step()

        # ------------
        #  Evaluation
        # ------------
        test_acc = 0
        for test_label in test_loader:
            test_label = test_label.to(device)

            with torch.no_grad():
                test_img = netG(fixed_noise, test_label)
                test_img = F.interpolate(test_img, size=64)
                test_acc += evaluator.eval(test_img, test_label)

            if iter % 100 == 0:
                save_image(make_grid(test_img * 0.5 + 0.5),
                           f'./test_images/iter_{iter}.png')

        test_acc = test_acc / len(test_loader)
        history['test_acc'].append(test_acc)

        pbar.set_postfix({
            'D_x': D_x,
            'D_G_before': D_G_before,
            'D_G_after': D_G_after,
            'test_acc': history['test_acc'][-1]
        })
        pbar.update()

        if history['test_acc'][-1] > highest_test_acc:
            highest_test_acc = history['test_acc'][-1]
            torch.save(netG, './models_weight/netG.pkl')
            torch.save(netD, './models_weight/netD.pkl')

    pbar.close()


with open('history.pkl', "wb") as fp:
    pickle.dump(history, fp)
