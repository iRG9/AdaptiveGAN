from Models.Discriminator import *
from Models.Generator import *
from torch.utils.tensorboard import SummaryWriter
from RandomAugment import *
from Metrics.AdaptiveAugHeuristic import *
from Metrics.Loss import *


def train(dataset, epochs, discriminator, generator, random_dim = 100):
    d_opt, g_opt = [torch.optim.SGD(discriminator.parameters(), lr=1e-6),
                    torch.optim.SGD(generator.parameters(), lr=1e-6)]
    writer = SummaryWriter()
    augmenter = RandomAugment()
    # p-heuristic function that informs adaptation
    h_dec_fn = Heuristic()
    p_increment = 1e-2
    lss = Loss()
    for epoch in range(epochs):
        avg = [0, 0]
        ct = 0
        for i, batch in enumerate(dataset, 0):
            d_opt.zero_grad()
            g_opt.zero_grad()
            rand_vec = torch.randn([1, random_dim])
            generated_images = generator(rand_vec)
            discriminator_output = discriminator(generated_images)
            # Augment generated images according to Low Data GAN Paper (Using non-leaky augmentations)
            generated_images = augmenter.apply_transforms(generated_images)
            # Adjust probability of transforms accordingly
            if Heuristic():
                augmenter.p += p_increment
            else:
                augmenter.p = max([0, augmenter.p - p_increment])
            disc_loss = lss.disc_loss(generated_images, batch)
            gen_loss = lss.g_loss(discriminator_output)

            # Compute gradients
            disc_loss.backward()
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
        avg = [e/ct for e in avg]
        # Report losses
        writer.add_scalar('Loss/generator', avg[0])
        writer.add_scalar('Loss/discriminator', avg[1])

        # Need to add Frechet Distance
