from Models.Discriminator import *
from Models.Generator import *
from torch.utils.tensorboard import SummaryWriter
from RandomAugment import *
from Metrics.AdaptiveAugHeuristic import *
from Metrics.Loss import *
import torch
import torchvision


def train(trainloader, epochs, discriminator, generator, random_dim):
    d_opt, g_opt = [torch.optim.SGD(discriminator.parameters(), lr=1e-6),
                    torch.optim.SGD(generator.parameters(), lr=1e-6)]
    writer = SummaryWriter()
    augmenter = RandomAugment()
    # heuristic function that informs p-adaptation
    heuristic = Heuristic()
    p_increment = 1e-2
    lss = Loss()
    for epoch in range(epochs):
        avg = [0, 0]
        ct = 0
        for i, batch in enumerate(trainloader, 0):
            # Zero gradients
            d_opt.zero_grad()
            g_opt.zero_grad()
            rand_vec = torch.randn([len(batch), random_dim, 1, 1])
            generated_images = generator(rand_vec)
            # Augment generated images according to Low Data GAN Paper (Using non-leaky augmentations)
            aug_gen_images = generated_images.clone()
            for i in range(generated_images.shape[0]):
                aug_gen_images[i] = augmenter.apply_transforms(generated_images[i])
            # Compute discriminator probabilities
            discriminator_fk_output = discriminator(aug_gen_images)
            discriminator_rl_output = discriminator(batch.float().view(-1, 3, 32, 32))

            # Adjust probability of transforms accordingly
            if heuristic.get_heur_inf_dec(torch.stack([discriminator_fk_output, discriminator_rl_output])):
                augmenter.p += p_increment
            else:
                augmenter.p = max([0, augmenter.p - p_increment])
            # Compute Loss
            disc_loss = lss.disc_loss(discriminator_fk_output, discriminator_rl_output)
            gen_loss = lss.gen_loss(discriminator_fk_output)

            # Compute gradients
            disc_loss.backward(retain_graph=True)
            gen_loss.backward()

            # Apply gradients
            d_opt.step()
            g_opt.step()

            # Add to summary stats
            avg[0] += disc_loss
            avg[1] += gen_loss
            ct += 1
        if epoch % 5 == 0:
            torch.save(discriminator.state_dict(), 'saves/discriminator')
            torch.save(generator.state_dict(), 'saves/generator')
        avg = [e / ct for e in avg]
        # Report losses
        writer.add_scalar('Loss/generator', avg[0])
        writer.add_scalar('Loss/discriminator', avg[1])

        # Need to add Frechet Distance for objective comparison


if __name__ == '__main__':
    BATCH_SIZE, IMG_SIDE_LEN, EPOCHS, RANDOM_DIM, MAX_DATA = [5, 32, 50, 100, 5000]
    # Use standard normalization for images
    transform = transforms.Compose(
        [transforms.Resize(IMG_SIDE_LEN), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Get and subset dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    sub_trainset = torch.utils.data.Subset(trainset.data, [i for i in range(MAX_DATA)])
    trainloader = torch.utils.data.DataLoader(sub_trainset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=10)
    gen, disc = [Generator(RANDOM_DIM), Discriminator(sub_trainset[0].shape[0])]
    train(trainloader, epochs=EPOCHS, discriminator=disc, generator=gen, random_dim=RANDOM_DIM)
