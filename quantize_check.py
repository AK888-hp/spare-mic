import torch
import torch.nn as nn
from dataset import get_dataloaders, CLASSES
from train_snn import SpikeRouterSNN, evaluate
import time
import os

def check_quantization(model_path="spikerouter_snn.pth"):
    # Ensure model exists
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run train_snn.py first.")
        return
        
    device = torch.device("cpu") # Quantization usually evaluated on CPU
    
    print("Loading datasets...")
    _, val_dl = get_dataloaders(batch_size=32)
    
    print("Loading trained Float32 model...")
    model_fp32 = SpikeRouterSNN(num_inputs=32, num_hidden=128, num_outputs=len(CLASSES)).to(device)
    model_fp32.load_state_dict(torch.load(model_path, map_location=device))
    model_fp32.eval()
    
    # 1. Evaluate FP32 Model
    print("Evaluating FP32 Model...")
    start_time = time.time()
    acc_fp32 = evaluate(model_fp32, val_dl, device=device)
    time_fp32 = time.time() - start_time
    print(f"FP32 Accuracy: {acc_fp32*100:.2f}% | Evaluation Time: {time_fp32:.2f}s")
    
    # 2. Apply Dynamic Quantization
    # We quantize the nn.Linear layers to qint8.
    # Note: snnTorch stateful nodes (snn.Leaky) maintain float32 states natively in PyTorch.
    # In a real ESP32 deployment, the state accumulation would also be fixed-point or int32/int16.
    print("\nApplying INT8 Dynamic Quantization to Linear Layers...")
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_fp32, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    # 3. Evaluate INT8 Model
    print("Evaluating INT8 Model...")
    start_time = time.time()
    acc_int8 = evaluate(model_int8, val_dl, device=device)
    time_int8 = time.time() - start_time
    print(f"INT8 Accuracy: {acc_int8*100:.2f}% | Evaluation Time: {time_int8:.2f}s")
    
    # 4. Compare Size
    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p")/1e3
        os.remove('temp.p')
        return size
        
    size_fp32 = print_size_of_model(model_fp32)
    size_int8 = print_size_of_model(model_int8)
    
    print(f"\nModel Size Comparison:")
    print(f"FP32 Size: {size_fp32:.2f} KB")
    print(f"INT8 Size: {size_int8:.2f} KB")
    print(f"Size Reduction: {100 * (1 - size_int8/size_fp32):.1f}%")
    print(f"Accuracy Drop: {(acc_fp32 - acc_int8)*100:.2f}%\n")
    
    print("NOTE FOR ESP32 DEPLOYMENT:")
    print("- This simulation only quantizes the weights. The ESP32 will also need quantized activations and membrane potentials (e.g., int32/int16 state accumulation).")
    print("- Consider using a framework like TFLite Micro, TVM, or writing custom C code (e.g., CMSIS-NN) to handle the Leaky Integrate-and-Fire state updates natively in fixed-point math on the ESP32.")

if __name__ == "__main__":
    check_quantization()
