import numpy as np


def generate_line(size):
    data = np.random.uniform(-1, 1, (size, 2))
    labels = np.where(data[:, 1] > data[:, 0], 1, 0)
    return data, labels
    
def generate_xor(size):
    data = np.random.uniform(-1, 1, (size, 2))
    labels = np.logical_xor(data[:, 0] > 0, data[:, 1] > 0).astype(int)
    return data, labels

def generate_circle(size):
    data = np.random.uniform(-1, 1, (size, 2))
    labels = np.sqrt(data[:, 0]**2 + data[:, 1]**2) <= np.sqrt(2/np.pi)
    labels = labels.astype(int)
    return data, labels

def generate_spiral(size, noise=0.5):
    n = size // 2  # Number of points per class
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi  # np.linspace(0, 2 * np.pi, n)

    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    data_a += np.random.randn(n, 2) * noise

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    data_b += np.random.randn(n, 2) * noise

    data = np.concatenate([data_a, data_b])
    labels = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    
    # Shuffle the dataset
    indices = np.arange(size)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    
    return data, labels