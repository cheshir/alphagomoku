#!/usr/bin/env python3
"""Test MPS usage in multiprocessing contexts"""

import torch
import multiprocessing as mp
import threading
import time
from alphagomoku.model.network import GomokuNet

def test_mps_in_subprocess(queue):
    """Test if MPS works in subprocess"""
    try:
        device = torch.device('mps')
        tensor = torch.randn(100, 100, device=device)
        result = tensor.sum().item()
        queue.put(('success', result))
    except Exception as e:
        queue.put(('error', str(e)))

def worker_inference(model_state, worker_id, results_queue):
    """Worker that loads model and runs inference"""
    try:
        # Load model in subprocess
        model = GomokuNet(board_size=15, num_blocks=30, channels=192)
        model.load_state_dict(model_state)

        # Try to move to MPS
        device = torch.device('mps')
        model = model.to(device)
        model.eval()

        # Run inference
        dummy_input = torch.randn(1, 5, 15, 15, device=device)
        with torch.no_grad():
            policy, value = model(dummy_input)

        results_queue.put((worker_id, 'success', 'MPS worked!'))
    except Exception as e:
        results_queue.put((worker_id, 'error', str(e)))

def test_concurrent_mps():
    """Test concurrent MPS access from threads"""
    model = GomokuNet(board_size=15, num_blocks=30, channels=192)
    device = torch.device('mps')
    model = model.to(device)
    model.eval()

    def run_inference(thread_id, results):
        try:
            dummy_input = torch.randn(8, 5, 15, 15, device=device)
            start = time.time()
            with torch.no_grad():
                policy, value = model(dummy_input)
            elapsed = time.time() - start
            results.append((thread_id, 'success', elapsed))
        except Exception as e:
            results.append((thread_id, 'error', str(e)))

    # Run multiple threads
    results = []
    threads = []
    for i in range(4):
        t = threading.Thread(target=run_inference, args=(i, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results

def test_larger_batch_sizes():
    """Test how large batches can get on MPS"""
    model = GomokuNet(board_size=15, num_blocks=30, channels=192)
    device = torch.device('mps')
    model = model.to(device)
    model.eval()

    batch_sizes = [16, 32, 64, 128, 256, 512]
    results = []

    for batch_size in batch_sizes:
        try:
            dummy_input = torch.randn(batch_size, 5, 15, 15, device=device)
            start = time.time()
            with torch.no_grad():
                policy, value = model(dummy_input)
            elapsed = time.time() - start
            throughput = batch_size / elapsed
            results.append((batch_size, 'success', elapsed, throughput))
        except Exception as e:
            results.append((batch_size, 'error', str(e), 0))

    return results

if __name__ == '__main__':
    print("=" * 60)
    print("Test 1: MPS in subprocess (multiprocessing)")
    print("=" * 60)

    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()
    process = mp.Process(target=test_mps_in_subprocess, args=(queue,))
    process.start()
    process.join(timeout=10)

    if process.is_alive():
        process.terminate()
        print("❌ Process hung")
    else:
        status, result = queue.get()
        print(f"Status: {status}")
        if status == 'error':
            print(f"❌ Error: {result}")
        else:
            print(f"✅ Success! Result: {result:.2f}")

    print("\n" + "=" * 60)
    print("Test 2: Model inference in subprocess")
    print("=" * 60)

    # Get model state dict
    model = GomokuNet(board_size=15, num_blocks=30, channels=192)
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    results_queue = mp.Queue()
    workers = []
    for i in range(2):
        p = mp.Process(target=worker_inference, args=(model_state, i, results_queue))
        workers.append(p)
        p.start()

    for p in workers:
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()

    while not results_queue.empty():
        worker_id, status, msg = results_queue.get()
        if status == 'success':
            print(f"✅ Worker {worker_id}: {msg}")
        else:
            print(f"❌ Worker {worker_id}: {msg}")

    print("\n" + "=" * 60)
    print("Test 3: Concurrent MPS access from threads")
    print("=" * 60)

    results = test_concurrent_mps()
    for thread_id, status, data in results:
        if status == 'success':
            print(f"✅ Thread {thread_id}: {data*1000:.1f}ms")
        else:
            print(f"❌ Thread {thread_id}: {data}")

    print("\n" + "=" * 60)
    print("Test 4: Batch size performance on MPS")
    print("=" * 60)

    results = test_larger_batch_sizes()
    print(f"{'Batch Size':<12} {'Status':<10} {'Time (ms)':<12} {'Throughput (infer/s)':<20}")
    print("-" * 60)
    for batch_size, status, elapsed, throughput in results:
        if status == 'success':
            print(f"{batch_size:<12} {status:<10} {elapsed*1000:<12.1f} {throughput:<20.1f}")
        else:
            print(f"{batch_size:<12} {status:<10} ERROR")

    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
