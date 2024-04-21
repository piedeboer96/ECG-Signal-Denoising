import torchvision.transforms as transforms
from torchvision import datasets

def download_mnist(data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images to range [-1, 1]
    ])
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return mnist_train, mnist_test

mnist_train, mnist_test = download_mnist()
