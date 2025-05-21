import numpy as np

def load_mnist(n_samples=1000):
    try:
        from sklearn.datasets import fetch_openml
        print('Loading MNIST dataset...')
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data[:n_samples]
        X = X / 255.0
        print(f'Loaded {n_samples} MNIST samples')
        return X.reshape((-1, 28, 28))
    except ImportError:
        print('sklearn not available. Generating synthetic data instead.')
        return generate_synthetic_data(n_samples)
    except Exception as e:
        print(f'Error loading MNIST: {e}. Generating synthetic data instead.')
        return generate_synthetic_data(n_samples)

def generate_alphabet_letters(n_samples=1000):
    print(f'Generating {n_samples} synthetic alphabet letters...')
    letters = {'A': ['  ***  ', ' *   * ', '*     *', '*     *', '*******', '*     *', '*     *'], 'B': ['***** ', '*     *', '*     *', '***** ', '*     *', '*     *', '***** '], 'C': [' **** ', '*     *', '*      ', '*      ', '*      ', '*     *', ' **** '], 'D': ['***** ', '*     *', '*     *', '*     *', '*     *', '*     *', '***** '], 'E': ['*******', '*      ', '*      ', '*****  ', '*      ', '*      ', '*******'], 'F': ['*******', '*      ', '*      ', '*****  ', '*      ', '*      ', '*      '], 'G': [' **** ', '*     *', '*      ', '*   ***', '*     *', '*     *', ' **** '], 'H': ['*     *', '*     *', '*     *', '*******', '*     *', '*     *', '*     *'], 'I': ['*******', '   *   ', '   *   ', '   *   ', '   *   ', '   *   ', '*******'], 'J': ['*******', '     * ', '     * ', '     * ', '*    * ', '*    * ', ' ***  '], 'K': ['*     *', '*    * ', '*   *  ', '***    ', '*   *  ', '*    * ', '*     *'], 'L': ['*      ', '*      ', '*      ', '*      ', '*      ', '*      ', '*******'], 'M': ['*     *', '**   **', '* * * *', '*  *  *', '*     *', '*     *', '*     *'], 'N': ['*     *', '**    *', '* *   *', '*  *  *', '*   * *', '*    **', '*     *'], 'O': [' **** ', '*     *', '*     *', '*     *', '*     *', '*     *', ' **** '], 'P': ['***** ', '*     *', '*     *', '***** ', '*      ', '*      ', '*      '], 'Q': [' **** ', '*     *', '*     *', '*     *', '*  *  *', '*   * *', ' **** '], 'R': ['***** ', '*     *', '*     *', '***** ', '*   *  ', '*    * ', '*     *'], 'S': [' **** ', '*     *', '*      ', ' **** ', '      *', '*     *', ' **** '], 'T': ['*******', '   *   ', '   *   ', '   *   ', '   *   ', '   *   ', '   *   '], 'U': ['*     *', '*     *', '*     *', '*     *', '*     *', '*     *', ' **** '], 'V': ['*     *', '*     *', '*     *', '*     *', ' *   * ', ' *   * ', '  ***  '], 'W': ['*     *', '*     *', '*     *', '*  *  *', '* * * *', '**   **', '*     *'], 'X': ['*     *', ' *   * ', '  * *  ', '   *   ', '  * *  ', ' *   * ', '*     *'], 'Y': ['*     *', ' *   * ', '  * *  ', '   *   ', '   *   ', '   *   ', '   *   '], 'Z': ['*******', '      *', '     * ', '    *  ', '   *   ', '  *    ', '*******']}
    data = []
    letter_list = list(letters.keys())
    for _ in range(n_samples):
        letter = np.random.choice(letter_list)
        letter_template = letters[letter]
        img = np.zeros((28, 28))
        start_y = 10
        start_x = 10
        for i, row in enumerate(letter_template):
            for j, char in enumerate(row):
                if char == '*':
                    for dy in range(2):
                        for dx in range(2):
                            y = start_y + i * 2 + dy
                            x = start_x + j * 2 + dx
                            if 0 <= y < 28 and 0 <= x < 28:
                                img[y, x] = np.random.uniform(0.7, 1.0)
        noise = np.random.normal(0, 0.1, (28, 28))
        img = np.clip(img + noise, 0, 1)
        if np.random.random() < 0.3:
            angle = np.random.uniform(-5, 5)
            try:
                from scipy.ndimage import rotate
                img = rotate(img, angle, reshape=False)
            except ImportError:
                pass
        data.append(img)
    return np.array(data)

def generate_synthetic_data(n_samples=1000):
    print(f'Generating {n_samples} synthetic 28x28 images...')
    data = []
    for _ in range(n_samples):
        img = np.zeros((28, 28))
        for _ in range(np.random.randint(1, 4)):
            x1, y1 = (np.random.randint(0, 20), np.random.randint(0, 20))
            x2, y2 = (x1 + np.random.randint(4, 8), y1 + np.random.randint(4, 8))
            img[y1:y2, x1:x2] = np.random.uniform(0.3, 1.0)
        for _ in range(np.random.randint(1, 3)):
            cx, cy = (np.random.randint(4, 24), np.random.randint(4, 24))
            r = np.random.randint(2, 6)
            y, x = np.ogrid[:28, :28]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
            img[mask] = np.random.uniform(0.2, 0.8)
        noise = np.random.normal(0, 0.1, (28, 28))
        img = np.clip(img + noise, 0, 1)
        data.append(img)
    return np.array(data)

def load_letter_dataset(n_samples=1000, dataset_type='alphabet'):
    if dataset_type == 'alphabet':
        return generate_alphabet_letters(n_samples)
    elif dataset_type == 'mnist':
        return load_mnist(n_samples)
    else:
        return generate_synthetic_data(n_samples)

def get_data_loader(data, batch_size=64):
    n_samples = len(data)
    indices = np.arange(n_samples)

    def get_batch():
        batch_indices = np.random.choice(indices, batch_size, replace=False)
        return data[batch_indices]
    return get_batch

def visualize_samples(data, n_samples=16, filename='sample_data.png'):
    try:
        import matplotlib.pyplot as plt
        if data.ndim == 3:
            pass
        elif data.ndim == 2:
            data = data.reshape(-1, 28, 28)
        else:
            print('Unexpected data format')
            return
        indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
        samples = data[indices]
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        plt.figure(figsize=(12, 3 * n_rows))
        for i, sample in enumerate(samples):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(sample, cmap='gray')
            plt.axis('off')
            plt.title(f'Sample {i + 1}')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'Sample data visualization saved as {filename}')
    except ImportError:
        print('Matplotlib not available. Skipping visualization.')
    except Exception as e:
        print(f'Error in visualization: {e}')
if __name__ == '__main__':
    data = generate_alphabet_letters(100)
    print(f'Letter data shape: {data.shape}')
    print(f'Data range: [{data.min():.3f}, {data.max():.3f}]')
    visualize_samples(data, 16, 'alphabet_samples.png')
