import torch

print("PyTorch version :", torch.__version__)
print("")

if torch.backends.mps.is_available():
    print("GPU: Apple MPS available — you have GPU acceleration!")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("GPU: CUDA available")
    device = torch.device("cuda")
else:
    print("GPU: Not available — will use CPU")
    device = torch.device("cpu")

print("Device to use:", device)