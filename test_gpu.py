import torch

print("=" * 60)
print("TORCH CUDA & CUDNN TEST")
print("=" * 60)

# 1. Basic CUDA Availability
print("\n1. CUDA AVAILABILITY:")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA version: {torch.version.cuda}")
print(f"   PyTorch version: {torch.__version__}")

# 2. CUDA Device Information
if torch.cuda.is_available():
    print("\n2. CUDA DEVICES:")
    print(f"   Number of CUDA devices: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   Device {i}:")
        print(f"     Name: {torch.cuda.get_device_name(i)}")
        print(f"     Capability: {torch.cuda.get_device_capability(i)}")
        print(f"     Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"     Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"     Memory Total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
    # Set current device
    torch.cuda.set_device(0)
    print(f"\n   Current device: {torch.cuda.current_device()}")
    
    # Test memory operations
    print("\n3. MEMORY OPERATIONS TEST:")
    try:
        # Allocate a tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        print(f"   ✓ Tensor allocated on GPU: {x.shape}, device: {x.device}")
        
        # Perform operation
        y = x @ x.t()
        print(f"   ✓ Matrix multiplication successful: {y.shape}")
        
        # Move back to CPU
        z = y.cpu()
        print(f"   ✓ Tensor moved to CPU: {z.shape}, device: {z.device}")
        
        # Clear cache
        del x, y, z
        torch.cuda.empty_cache()
        print(f"   ✓ CUDA cache cleared")
        
    except Exception as e:
        print(f"   ✗ Memory test failed: {e}")

# 3. cuDNN Test
print("\n4. CUDNN TEST:")
try:
    # Check if cuDNN is available
    print(f"   cuDNN available: {torch.backends.cudnn.enabled}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    
    # Test cuDNN with convolution
    if torch.cuda.is_available() and torch.backends.cudnn.enabled:
        print("\n   cuDNN CONVOLUTION TEST:")
        
        # Create test tensors
        input_tensor = torch.randn(1, 3, 224, 224).cuda()
        weights = torch.randn(64, 3, 3, 3).cuda()
        
        # Use cuDNN convolution
        with torch.backends.cudnn.flags(enabled=True, benchmark=True):
            output = torch.nn.functional.conv2d(input_tensor, weights, padding=1)
            print(f"   ✓ cuDNN convolution successful: {output.shape}")
            
            # Benchmark
            import time
            start = time.time()
            for _ in range(100):
                _ = torch.nn.functional.conv2d(input_tensor, weights, padding=1)
            torch.cuda.synchronize()
            end = time.time()
            print(f"   ✓ cuDNN benchmark: {100/(end-start):.2f} ops/sec")
            
        # Clean up
        del input_tensor, weights, output
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"   ✗ cuDNN test failed: {e}")

# 4. Advanced Tests
print("\n5. ADVANCED TESTS:")
if torch.cuda.is_available():
    try:
        # Test different data types
        print("   DATA TYPE TESTS:")
        
        # Float32
        a_fp32 = torch.randn(100, 100, dtype=torch.float32).cuda()
        b_fp32 = torch.randn(100, 100, dtype=torch.float32).cuda()
        c_fp32 = a_fp32 @ b_fp32
        print(f"   ✓ FP32 matmul: {c_fp32.shape}")
        
        # Float16 (if supported)
        if torch.cuda.get_device_capability(0)[0] >= 5:  # Pascal or newer
            try:
                a_fp16 = torch.randn(100, 100, dtype=torch.float16).cuda()
                b_fp16 = torch.randn(100, 100, dtype=torch.float16).cuda()
                c_fp16 = a_fp16 @ b_fp16
                print(f"   ✓ FP16 matmul: {c_fp16.shape}")
            except:
                print(f"   ✗ FP16 not supported on this GPU")
        
        # Benchmark matrix multiplication
        print("\n   MATRIX MULTIPLICATION BENCHMARK:")
        size = 1024
        A = torch.randn(size, size).cuda()
        B = torch.randn(size, size).cuda()
        
        # Warm up
        for _ in range(10):
            _ = A @ B
        
        # Time it
        import time
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            _ = A @ B
        torch.cuda.synchronize()
        end = time.time()
        
        gflops = (2 * size**3 * iterations) / ((end - start) * 1e9)
        print(f"   ✓ Performance: {gflops:.2f} GFLOPS")
        
        # Clean up
        del A, B
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ✗ Advanced tests failed: {e}")

# 5. Error Diagnostics
print("\n6. ERROR DIAGNOSTICS:")
try:
    # Check for common issues
    if torch.cuda.is_available():
        # Check CUDA initialization
        print(f"   CUDA initialized: {torch.cuda.is_initialized()}")
        
        # Check compute capability
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            print(f"   Device {i} compute capability: {capability[0]}.{capability[1]}")
            
            # Check if capability is supported
            if capability[0] < 3:
                print(f"   ⚠️  Warning: Device {i} has compute capability {capability}, which might not support all features")
    
    # Check cuDNN status
    print(f"\n   cuDNN status:")
    print(f"     Enabled: {torch.backends.cudnn.enabled}")
    print(f"     Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"     Deterministic: {torch.backends.cudnn.deterministic}")
    
    # Test tensor operations
    if torch.cuda.is_available():
        print(f"\n   TENSOR OPERATION TESTS:")
        
        # Test 1: Simple tensor creation
        t1 = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"     ✓ Tensor creation: {t1.device}")
        
        # Test 2: Random number generation
        t2 = torch.randn(10).cuda()
        print(f"     ✓ Random generation: {t2.mean().item():.4f}")
        
        # Test 3: BLAS operation
        t3 = torch.ones(100).cuda()
        t4 = torch.ones(100).cuda()
        dot = torch.dot(t3, t4)
        print(f"     ✓ DOT product: {dot.item():.2f}")
        
except Exception as e:
    print(f"   ✗ Diagnostics failed: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

# Final Summary
print("\nSUMMARY:")
if torch.cuda.is_available():
    print("✅ CUDA is working correctly")
    if torch.backends.cudnn.enabled:
        print("✅ cuDNN is enabled")
    else:
        print("⚠️  cuDNN is disabled")
    
    # Show current memory usage
    print(f"\nCurrent Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # Reset peak memory stats for accurate measurement
    torch.cuda.reset_peak_memory_stats()
else:
    print("❌ CUDA is not available")
    print("\nTroubleshooting tips:")
    print("  1. Check if NVIDIA drivers are installed: nvidia-smi")
    print("  2. Check if CUDA toolkit is installed: nvcc --version")
    print("  3. Reinstall PyTorch with CUDA support:")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")