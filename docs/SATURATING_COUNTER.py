import os
import time
from collections import deque

# ==== Predictor class ====
class HistoryBasedCounterPredictor:
    def __init__(self, history_len=1, table_size=128):
        self.history_len = history_len
        self.global_history = deque([0] * history_len, maxlen=history_len)
        self.table_size = table_size
        self.counter_table = [2] * table_size  # Start at 'weakly taken'

    def get_index(self, addr):
        history_bits = ''.join(str(b) for b in self.global_history)
        history_int = int(history_bits, 2)
        addr_int = int(addr, 16)
        return (addr_int ^ history_int) % self.table_size

    def predict(self, addr):
        index = self.get_index(addr)
        return 1 if self.counter_table[index] >= 2 else -1

    def update(self, addr, actual):
        index = self.get_index(addr)
        if actual == 1:
            self.counter_table[index] = min(3, self.counter_table[index] + 1)
        else:
            self.counter_table[index] = max(0, self.counter_table[index] - 1)
        self.global_history.appendleft(1 if actual == 1 else 0)

    def run(self, trace):
        correct = 0
        for addr, taken in trace:
            actual = 1 if taken == 1 else -1
            prediction = self.predict(addr)
            if prediction == actual:
                correct += 1
            self.update(addr, actual)
        return correct

# ==== Load trace file ====
fileIn = input('Provide the filepath for the history-dependent trace file (with quotes): ').strip('"')
assert os.path.exists(fileIn), 'ERROR: The input file does not exist.'

trace = []
with open(fileIn, 'r') as f:
    for line in f:
        tok = line.strip().split()
        trace.append([tok[0], int(tok[1])])

# ==== Run and print ====
print("\nRunning History-Based Counter Predictor on:", fileIn)

results = []
for i in range(1, 50):
    start_time = time.time()
    predictor = HistoryBasedCounterPredictor(history_len=i)
    num_correct = predictor.run(trace)
    end_time = time.time()
    acc = num_correct / len(trace)
    results.append((acc, end_time - start_time))
    print(f"i: {i} --> Accuracy: {acc:.4f}, Time: {end_time - start_time:.4f}s")

# ==== Save final result ====
with open("results_counter", "a") as f:
    f.write(f"{results[-1][0] * 100:.2f}\n")

# ==== For plotting (optional) ====
counter_accuracies = [(i, acc, runtime) for i, (acc, runtime) in enumerate(results, start=1)]
