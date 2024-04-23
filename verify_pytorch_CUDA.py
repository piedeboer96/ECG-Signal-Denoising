import torch

# Creating a test tensor
x = torch. randint(1, 100, (100, 100))
# Checking the device name:
# Should return 'cpu' by default
print(x. device)


import torch
y = torch. cuda. is_available()

print('Is CUDA available')

print(torch. cuda. is_available())