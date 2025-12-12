import torch
try:
    import vllm
    print(f"vLLM version: {vllm.__version__}")
except ImportError:
    print("vLLM not installed")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
