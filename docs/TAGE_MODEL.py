import os
import time
from collections import deque

class TAGEEntry:
    def __init__(self, tag, counter=2):
        self.tag = tag
        self.counter = counter  # 2-bit saturating counter

class TAGEPredictor:
    def __init__(self, table_sizes=[128, 128, 128, 128], history_lengths=[2, 4, 8, 16]):
        self.num_tables = len(table_sizes)
        self.table_sizes = table_sizes
        self.history_lengths = history_lengths
        self.tables = [dict() for _ in range(self.num_tables)]
        self.global_history = deque([0]*max(history_lengths), maxlen=max(history_lengths))

    def _compute_index_and_tag(self, addr, history_len, table_size):
        addr_int = int(addr, 16)
        history_bits = list(self.global_history)[:history_len]
        hval = int(''.join(str(b) for b in history_bits), 2) if history_bits else 0
        index = (addr_int ^ hval) % table_size
        tag = (addr_int ^ (hval << 1)) % 256  # Small tag
        return index, tag

    def predict(self, addr):
        for i in reversed(range(self.num_tables)):
            index, tag = self._compute_index_and_tag(addr, self.history_lengths[i], self.table_sizes[i])
            entry = self.tables[i].get(index)
            if entry and entry.tag == tag:
                return 1 if entry.counter >= 2 else -1, i  # Return prediction and table used
        return 1, -1  # Default weakly taken if no match

    def update(self, addr, actual):
        pred, used_table = self.predict(addr)
        actual_val = 1 if actual else -1

        if used_table != -1:
            index, tag = self._compute_index_and_tag(addr, self.history_lengths[used_table], self.table_sizes[used_table])
            entry = self.tables[used_table][index]
            if actual_val == 1:
                entry.counter = min(3, entry.counter + 1)
            else:
                entry.counter = max(0, entry.counter - 1)
        else:
            # Allocate new entry in longest-history table
            index, tag = self._compute_index_and_tag(addr, self.history_lengths[-1], self.table_sizes[-1])
            self.tables[-1][index] = TAGEEntry(tag, 2 if actual_val == 1 else 1)

        self.global_history.appendleft(1 if actual_val == 1 else 0)

    def run(self, trace):
        correct = 0
        for addr, taken in trace:
            prediction, _ = self.predict(addr)
            actual_val = 1 if taken == 1 else -1
            if prediction == actual_val:
                correct += 1
            self.update(addr, taken)
        return correct

# ==== Load trace ====
fileIn = input('Provide the filepath for the trace file (with quotes): ').strip('"')
assert os.path.exists(fileIn), 'ERROR: The input file does not exist.'
trace = []
with open(fileIn, 'r') as f:
    for line in f:
        tok = line.strip().split()
        trace.append([tok[0], int(tok[1])])

# ==== Run and print ====
print("\nRunning TAGE Predictor on:", fileIn)

results = []
for i in range(1, 50):
    start_time = time.time()
    predictor = TAGEPredictor(
        table_sizes=[128]*4,
        history_lengths=[i, i*2, i*4, i*8]
    )
    num_correct = predictor.run(trace)
    end_time = time.time()
    acc = num_correct / len(trace)
    results.append((acc, end_time - start_time))
    print(f"i: {i} --> Accuracy: {acc:.4f}, Time: {end_time - start_time:.4f}s")

# ==== Save ====
with open("results_tage", "a") as f:
    f.write(f"{results[-1][0] * 100:.2f}\n")

tage_accuracies = [(i, acc, runtime) for i, (acc, runtime) in enumerate(results, start=1)]
