import os
import time
from collections import deque

class HistoryBasedCounterPredictor:
    def __init__(self, history_len=1, table_size=128):
        self.history_len = history_len
        self.global_history = deque([0]*history_len, maxlen=history_len)
        self.table_size = table_size
        self.counter_table = [2] * table_size  # 2-bit counters initialized to weakly taken

    def get_index(self, addr):
        history_bits = ''.join(str(bit) for bit in self.global_history)
        history_int = int(history_bits, 2)
        addr_int = int(addr, 16) if isinstance(addr, str) else addr
        return (addr_int ^ history_int) % self.table_size

    def predict(self, addr):
        index = self.get_index(addr)
        counter = self.counter_table[index]
        return 1 if counter >= 2 else -1

    def update(self, addr, actual):
        index = self.get_index(addr)
        if actual == 1:
            self.counter_table[index] = min(self.counter_table[index] + 1, 3)
        else:
            self.counter_table[index] = max(self.counter_table[index] - 1, 0)
        self.global_history.appendleft(1 if actual == 1 else 0)

    def run(self, trace):
        correct = 0
        for addr, taken in trace:
            actual = 1 if taken else -1
            pred = self.predict(addr)
            if pred == actual:
                correct += 1
            self.update(addr, actual)
        return correct

# ==== LOAD TRACE FILE ====
fileIn = input('Provide the filepath for the trace (with quotes): ').strip('"')
assert os.path.exists(fileIn), 'ERROR: The input file does not exist.'
with open(fileIn, 'r') as branchfile:
    trace = []
    for line in branchfile.readlines():
        tok = line.strip().split(' ')
        trace.append([tok[0], int(tok[1])])

# ==== BASELINE TEST ====
num_correct = HistoryBasedCounterPredictor(5).run(trace)
print('\nInitial test with history=5: Accuracy = {:.2f}%\n'.format(num_correct / len(trace) * 100))

# ==== ACCURACY PER HISTORY LENGTH ====
results = []
for i in range(1, 50):
    start_time = time.time()
    predictor = HistoryBasedCounterPredictor(history_len=i)
    num_correct = predictor.run(trace)
    end_time = time.time()
    acc = num_correct / float(len(trace))
    results.append((acc, end_time - start_time))
    print('i: ' + str(i) + ' --> Accuracy: {:.4f}, Time: {:.4f} sec'.format(acc, results[-1][1]))

# ==== SAVE FINAL RESULT ====
with open('results_counter', 'a') as fileOut:
    fileOut.write(str(results[-1][0] * 100) + '\n')

# ==== STORE FOR PLOTTING ====
counter_accuracies = []
for i in range(1, 50):
    acc, runtime = results[i-1]
    counter_accuracies.append((i, acc, runtime))
