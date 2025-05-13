import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
        
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test creating a tensor on GPU
    print("\nCreating test tensor on GPU...")
    try:
        x = torch.ones(10, device='cuda')
        print(f"Test tensor created on {x.device}")
        print(f"  Current memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Error creating tensor on GPU: {e}")
        
    # Try to load the Qwen model
    print("\nLoading Qwen model onto GPU...")
    try:
        model_name = "Qwen/Qwen3-0.6B"
        
        # First load tokenizer (should be fast)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully")
        
        # Now try loading model onto GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Auto-assign to available devices
            torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        )
        print(f"Model loaded successfully")
        print(f"Model is on device: {next(model.parameters()).device}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Try to generate some text to verify the model works
        inputs = tokenizer("Hello, I am a", return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=20)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated text: {generated_text}")
        
        print("\nGPU TEST PASSED - Model is running on GPU!")
        
    except Exception as e:
        print(f"Error loading model on GPU: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main() 