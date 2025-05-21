import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def binary_cross_entropy(preds, targets):
    epsilon = 1e-08
    return -np.mean(targets * np.log(preds + epsilon) + (1 - targets) * np.log(1 - preds + epsilon))

def binary_cross_entropy_grad(preds, targets):
    epsilon = 1e-08
    return (preds - targets) / (preds * (1 - preds) + epsilon)

def generate_noise(batch_size, noise_dim):
    return np.random.randn(batch_size, noise_dim)

def save_generated_image(image, filename, epoch):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.title(f'Generated Image - Epoch {epoch}')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', dpi=100)
        plt.close()
        print(f'Image saved as {filename}')
    except ImportError:
        print('Matplotlib not available. Skipping image save.')
