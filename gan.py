import numpy as np
from utils import sigmoid, sigmoid_deriv, relu, relu_deriv, binary_cross_entropy, binary_cross_entropy_grad, generate_noise, save_generated_image
from dataset import load_letter_dataset, get_data_loader

class Dense:

    def __init__(self, input_dim, output_dim, weight_init_scale=0.01):
        self.W = np.random.randn(input_dim, output_dim) * weight_init_scale
        self.b = np.zeros((1, output_dim))
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = x @ self.W + self.b
        return self.output

    def backward(self, grad_output, lr):
        grad_input = grad_output @ self.W.T
        grad_W = self.input.T @ grad_output
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return grad_input

class Generator:

    def __init__(self, noise_dim, hidden_dim, output_dim):
        self.l1 = Dense(noise_dim, hidden_dim, weight_init_scale=0.02)
        self.l2 = Dense(hidden_dim, output_dim, weight_init_scale=0.02)
        self.layers = [self.l1, self.l2]

    def forward(self, z):
        h = relu(self.l1.forward(z))
        out = sigmoid(self.l2.forward(h))
        return out

    def backward(self, grad, lr):
        grad = self.l2.backward(grad * sigmoid_deriv(self.l2.output), lr)
        grad = self.l1.backward(grad * relu_deriv(self.l1.output), lr)
        return grad

class Discriminator:

    def __init__(self, input_dim, hidden_dim):
        self.l1 = Dense(input_dim, hidden_dim, weight_init_scale=0.02)
        self.l2 = Dense(hidden_dim, 1, weight_init_scale=0.02)
        self.layers = [self.l1, self.l2]

    def forward(self, x):
        h = relu(self.l1.forward(x))
        out = sigmoid(self.l2.forward(h))
        return out

    def backward(self, grad, lr):
        grad = self.l2.backward(grad * sigmoid_deriv(self.l2.output), lr)
        grad = self.l1.backward(grad * relu_deriv(self.l1.output), lr)
        return grad

class GAN:

    def __init__(self, noise_dim=64, hidden_dim=128, image_dim=784, lr=0.01):
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.image_dim = image_dim
        self.lr = lr
        self.generator = Generator(noise_dim, hidden_dim, image_dim)
        self.discriminator = Discriminator(image_dim, hidden_dim)
        self.d_losses = []
        self.g_losses = []

    def train_step(self, real_batch, batch_size):
        z = generate_noise(batch_size, self.noise_dim)
        fake_samples = self.generator.forward(z)
        d_real = self.discriminator.forward(real_batch)
        d_fake = self.discriminator.forward(fake_samples)
        d_loss_real = binary_cross_entropy(d_real, np.ones((batch_size, 1)))
        d_loss_fake = binary_cross_entropy(d_fake, np.zeros((batch_size, 1)))
        d_loss = d_loss_real + d_loss_fake
        grad_real = binary_cross_entropy_grad(d_real, np.ones((batch_size, 1)))
        grad_fake = binary_cross_entropy_grad(d_fake, np.zeros((batch_size, 1)))
        self.discriminator.backward(grad_real, self.lr)
        self.discriminator.backward(grad_fake, self.lr)
        z = generate_noise(batch_size, self.noise_dim)
        fake_samples = self.generator.forward(z)
        d_fake = self.discriminator.forward(fake_samples)
        g_loss = binary_cross_entropy(d_fake, np.ones((batch_size, 1)))
        grad_g = binary_cross_entropy_grad(d_fake, np.ones((batch_size, 1)))
        grad_fake = self.discriminator.backward(grad_g, self.lr)
        self.generator.backward(grad_fake, self.lr)
        return (d_loss, g_loss)

    def train(self, data, epochs=10000, batch_size=64, save_interval=500):
        print(f'Starting GAN training for {epochs} epochs...')
        print(f'Data shape: {data.shape}')
        print(f'Batch size: {batch_size}, Learning rate: {self.lr}')
        if data.ndim == 3:
            X_real = data.reshape((-1, self.image_dim))
        else:
            X_real = data
        get_batch = get_data_loader(X_real, batch_size)
        for epoch in range(epochs):
            real_batch = get_batch()
            d_loss, g_loss = self.train_step(real_batch, batch_size)
            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)
            if epoch % save_interval == 0:
                print(f'Epoch {epoch:5d} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}')
                sample = self.generate_samples(1)
                save_generated_image(sample.reshape(28, 28), f'generated_epoch_{epoch}.png', epoch)
        print('Training completed!')
        return (self.d_losses, self.g_losses)

    def generate_samples(self, n_samples=1):
        z = generate_noise(n_samples, self.noise_dim)
        samples = self.generator.forward(z)
        return samples

    def save_model(self, filename_prefix='gan_model'):
        try:
            gen_weights = {'l1_W': self.generator.l1.W, 'l1_b': self.generator.l1.b, 'l2_W': self.generator.l2.W, 'l2_b': self.generator.l2.b}
            np.save(f'{filename_prefix}_generator.npy', gen_weights)
            disc_weights = {'l1_W': self.discriminator.l1.W, 'l1_b': self.discriminator.l1.b, 'l2_W': self.discriminator.l2.W, 'l2_b': self.discriminator.l2.b}
            np.save(f'{filename_prefix}_discriminator.npy', disc_weights)
            print(f'Model saved as {filename_prefix}_*.npy')
        except Exception as e:
            print(f'Error saving model: {e}')

    def load_model(self, filename_prefix='gan_model'):
        try:
            gen_weights = np.load(f'{filename_prefix}_generator.npy', allow_pickle=True).item()
            self.generator.l1.W = gen_weights['l1_W']
            self.generator.l1.b = gen_weights['l1_b']
            self.generator.l2.W = gen_weights['l2_W']
            self.generator.l2.b = gen_weights['l2_b']
            disc_weights = np.load(f'{filename_prefix}_discriminator.npy', allow_pickle=True).item()
            self.discriminator.l1.W = disc_weights['l1_W']
            self.discriminator.l1.b = disc_weights['l1_b']
            self.discriminator.l2.W = disc_weights['l2_W']
            self.discriminator.l2.b = disc_weights['l2_b']
            print(f'Model loaded from {filename_prefix}_*.npy')
        except Exception as e:
            print(f'Error loading model: {e}')

def main():
    print('Loading alphabet letter dataset...')
    data = load_letter_dataset(n_samples=1000, dataset_type='alphabet')
    gan = GAN(noise_dim=64, hidden_dim=256, image_dim=784, lr=0.005)
    d_losses, g_losses = gan.train(data=data, epochs=8000, batch_size=64, save_interval=500)
    print('\nGenerating final samples...')
    samples = gan.generate_samples(16)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 12))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(samples[i].reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.title(f'Sample {i + 1}')
        plt.tight_layout()
        plt.savefig('final_generated_samples.png', bbox_inches='tight', dpi=150)
        plt.close()
        print('Final samples saved as final_generated_samples.png')
    except ImportError:
        print('Matplotlib not available. Skipping final visualization.')
    gan.save_model()
    print('GAN training completed successfully!')
if __name__ == '__main__':
    main()
