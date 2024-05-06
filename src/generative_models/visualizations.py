import matplotlib.pyplot as plt


class Visualizations:
    def __init__(self):
        pass

    def visualize_tensor(self,tensor):
        print('Shape', tensor.shape)

        # Convert the tensor to a NumPy array
        image_array = tensor.numpy()

        # Transpose the array to (H, W, C) format
        image_array = image_array.transpose(1, 2, 0)

        # Display the image using Matplotlib
        plt.imshow(image_array)
        plt.axis('off')  # Turn off axis
        plt.show()
    
    def plot_multiple_timeseries(self, signals, names):
        num_signals = len(signals)
        
        plt.figure(figsize=(5 * num_signals, 4))

        for i, (signal, name) in enumerate(zip(signals, names), 1):
            plt.subplot(1, num_signals, i)
            plt.plot(signal)
            plt.title(name)
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()