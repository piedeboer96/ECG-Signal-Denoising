import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Check the number of available CUDA devices
    num_devices = torch.cuda.device_count()
    print("CUDA is available! You have {} CUDA device(s) available.".format(num_devices))
    
    # Print information about each CUDA device
    for i in range(num_devices):
        print("CUDA Device {}: {}".format(i, torch.cuda.get_device_name(i)))
else:
    print("CUDA is not available. PyTorch cannot run on CUDA.")
