import numpy as np
from gan import GAN
from dataset import load_letter_dataset, visualize_samples

def demo_generate_samples():
    print('GAN Demo: Generating Samples')
    print('=' * 40)
    gan = GAN(noise_dim=64, hidden_dim=128, image_dim=784, lr=0.01)
    try:
        gan.load_model('gan_model')
        print('✓ Loaded trained model successfully')
    except Exception as e:
        print(f'✗ Could not load trained model: {e}')
        print('Training a new model for demo...')
        data = load_letter_dataset(100, dataset_type='alphabet')
        gan.train(data, epochs=100, batch_size=32, save_interval=50)
    print('\nGenerating 16 samples...')
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
        plt.savefig('demo_samples.png', bbox_inches='tight', dpi=150)
        plt.close()
        print('✓ Samples saved as demo_samples.png')
    except ImportError:
        print('Matplotlib not available. Skipping visualization.')
        print('Sample statistics:')
        print(f'  Shape: {samples.shape}')
        print(f'  Range: [{samples.min():.3f}, {samples.max():.3f}]')
        print(f'  Mean: {samples.mean():.3f}')
        print(f'  Std: {samples.std():.3f}')

def demo_interactive():
    print('\nGAN Demo: Interactive Generation')
    print('=' * 40)
    gan = GAN(noise_dim=64, hidden_dim=128, image_dim=784, lr=0.01)
    try:
        gan.load_model('gan_model')
        print('✓ Loaded trained model')
    except:
        print('✗ No trained model found. Using untrained model for demo.')
    print('\nGenerating samples with different random seeds...')
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i in range(8):
            sample = gan.generate_samples(1)
            axes[i].imshow(sample[0].reshape(28, 28), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Random Sample {i + 1}')
        plt.tight_layout()
        plt.savefig('interactive_demo.png', bbox_inches='tight', dpi=150)
        plt.close()
        print('✓ Interactive demo saved as interactive_demo.png')
    except ImportError:
        print('Matplotlib not available. Skipping interactive demo.')

def demo_compare_real_fake():
    print('\nGAN Demo: Real vs Generated Comparison')
    print('=' * 40)
    print('Loading real data...')
    real_data = load_letter_dataset(16, dataset_type='alphabet')
    gan = GAN(noise_dim=64, hidden_dim=128, image_dim=784, lr=0.01)
    try:
        gan.load_model('gan_model')
        print('✓ Loaded trained model')
    except:
        print('✗ No trained model found. Using untrained model.')
    print('Generating fake samples...')
    fake_samples = gan.generate_samples(16)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
        for i in range(8):
            axes[0, i].imshow(real_data[i], cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Real {i + 1}')
        for i in range(8):
            axes[1, i].imshow(fake_samples[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Fake {i + 1}')
        for i in range(8):
            axes[2, i].imshow(real_data[i + 8], cmap='gray')
            axes[2, i].axis('off')
            axes[2, i].set_title(f'Real {i + 9}')
        for i in range(8):
            axes[3, i].imshow(fake_samples[i + 8].reshape(28, 28), cmap='gray')
            axes[3, i].axis('off')
            axes[3, i].set_title(f'Fake {i + 9}')
        plt.tight_layout()
        plt.savefig('real_vs_fake.png', bbox_inches='tight', dpi=150)
        plt.close()
        print('✓ Comparison saved as real_vs_fake.png')
    except ImportError:
        print('Matplotlib not available. Skipping comparison.')
        print('Real data statistics:')
        print(f'  Shape: {real_data.shape}')
        print(f'  Range: [{real_data.min():.3f}, {real_data.max():.3f}]')
        print('Fake data statistics:')
        print(f'  Shape: {fake_samples.shape}')
        print(f'  Range: [{fake_samples.min():.3f}, {fake_samples.max():.3f}]')

def main():
    print('GAN from Scratch - Demo Suite')
    print('=' * 50)
    demo_generate_samples()
    demo_interactive()
    demo_compare_real_fake()
    print('\n' + '=' * 50)
    print('Demo completed! Check the generated PNG files.')
    print('=' * 50)
if __name__ == '__main__':
    main()
