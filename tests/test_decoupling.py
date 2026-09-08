import queue
import threading
import time
import json

# Test decoupling pattern
def simulate_gpu_and_network(with_queue=False):
    t0 = time.time()
    tokens = []
    
    if not with_queue:
        # Synchronous socket write simulation (e.g. 12ms network overhead per token)
        for i in range(50):
            # GPU time
            time.sleep(0.0218) # 21.8ms
            # Network time
            time.sleep(0.014) # 14ms
            tokens.append(i)
    else:
        # Decoupled
        q = queue.Queue()
        done_ev = threading.Event()
        
        def writer():
            while not done_ev.is_set() or not q.empty():
                try:
                    tok = q.get(timeout=0.05)
                    time.sleep(0.014) # network write
                    q.task_done()
                except queue.Empty:
                    pass
                    
        t = threading.Thread(target=writer)
        t.start()
        
        t_gen_start = time.time()
        for i in range(50):
            time.sleep(0.0218) # GPU time
            q.put(i)
        t_gen = time.time() - t_gen_start
        done_ev.set()
        t.join()
        t_tot = time.time() - t0
        return 50 / t_gen, 50 / t_tot
        
    t_tot = time.time() - t0
    return 50 / t_tot, 50 / t_tot

sync_rate, _ = simulate_gpu_and_network(with_queue=False)
gen_rate, tot_rate = simulate_gpu_and_network(with_queue=True)
print(f"Sync rate: {sync_rate:.1f} tok/s")
print(f"Decoupled GPU rate: {gen_rate:.1f} tok/s, E2E client rate: {tot_rate:.1f} tok/s")
