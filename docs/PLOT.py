import random
import time
import matplotlib.pyplot as plt
from collections import deque

# --- Generate Trace ---
def generate_trace(mode='biased', num_entries=200, num_addresses=16):
    trace = []
    history = deque([0]*8, maxlen=8)
    for i in range(num_entries):
        addr = f"{random.randint(0, num_addresses - 1):X}"
        if mode == 'biased':
            taken = 1 if random.random() < 0.9 else 0
        elif mode == 'random':
            taken = random.choice([0, 1])
        elif mode == 'alternating':
            taken = i % 2
        elif mode == 'periodic':
            pattern = [1, 1, 0, 1, 0, 0]
            taken = pattern[i % len(pattern)]
        else:
            raise ValueError("Unknown mode.")
        history.appendleft(taken)
        trace.append([addr, taken])
    return trace

# --- Predictors ---
class Perceptron:
    def __init__(self, N):
        self.N = N
        self.bias = 0
        self.threshold = 2 * N + 14
        self.weights = [0] * N

    def predict(self, history):
        summation = self.bias
        for i in range(self.N):
            summation += history[i] * self.weights[i]
        return (1 if summation >= 0 else -1), summation

    def update(self, prediction, actual, history, summation):
        if prediction != actual or abs(summation) < self.threshold:
            self.bias += actual
            for i in range(self.N):
                self.weights[i] += actual * history[i]

class SaturatingCounterPredictor:
    def __init__(self):
        self.table = {}

    def predict(self, addr):
        if addr not in self.table:
            self.table[addr] = 2
        return 1 if self.table[addr] >= 2 else -1

    def update(self, addr, actual):
        if actual == 1:
            self.table[addr] = min(3, self.table[addr] + 1)
        else:
            self.table[addr] = max(0, self.table[addr] - 1)

class TAGEEntry:
    def __init__(self, tag, counter=2):
        self.tag = tag
        self.counter = counter

class TAGEPredictor:
    def __init__(self, table_sizes=[128]*4, history_lengths=[2, 4, 8, 16]):
        self.table_sizes = table_sizes
        self.history_lengths = history_lengths
        self.tables = [dict() for _ in table_sizes]
        self.global_history = deque([0]*max(history_lengths), maxlen=max(history_lengths))

    def _index_and_tag(self, addr, history_len, size):
        addr_int = int(addr, 16)
        hbits = list(self.global_history)[:history_len]
        hval = int(''.join(str(b) for b in hbits), 2) if hbits else 0
        index = (addr_int ^ hval) % size
        tag = (addr_int ^ (hval << 1)) % 256
        return index, tag

    def predict(self, addr):
        for i in reversed(range(len(self.tables))):
            index, tag = self._index_and_tag(addr, self.history_lengths[i], self.table_sizes[i])
            entry = self.tables[i].get(index)
            if entry and entry.tag == tag:
                return 1 if entry.counter >= 2 else -1, i
        return 1, -1

    def update(self, addr, actual):
        pred, table_used = self.predict(addr)
        actual_val = 1 if actual else -1
        if table_used != -1:
            index, tag = self._index_and_tag(addr, self.history_lengths[table_used], self.table_sizes[table_used])
            entry = self.tables[table_used][index]
            entry.counter = min(3, entry.counter + 1) if actual_val == 1 else max(0, entry.counter - 1)
        else:
            index, tag = self._index_and_tag(addr, self.history_lengths[-1], self.table_sizes[-1])
            self.tables[-1][index] = TAGEEntry(tag, 2 if actual_val == 1 else 1)
        self.global_history.appendleft(1 if actual_val == 1 else 0)

# --- Evaluation Functions ---
def evaluate_predictor(predictor_name, trace, history_len=8):
    correct, total = 0, 0
    history = deque([0]*history_len, maxlen=history_len)
    p_table = {}
    tage = TAGEPredictor()

    if predictor_name == 'perceptron':
        start = time.time()
        for addr, taken in trace:
            idx = hash(addr) % 128
            if idx not in p_table:
                p_table[idx] = Perceptron(history_len)
            p = p_table[idx]
            pred, summ = p.predict(history)
            actual = 1 if taken else -1
            p.update(pred, actual, history, summ)
            correct += (pred == actual)
            history.appendleft(actual)
        end = time.time()
        storage = len(p_table) * (history_len + 1) * 4  # 4 bytes/weight
        return correct/len(trace), (end - start), storage

    elif predictor_name == 'counter':
        predictor = SaturatingCounterPredictor()
        start = time.time()
        for addr, taken in trace:
            pred = predictor.predict(addr)
            actual = 1 if taken else -1
            predictor.update(addr, actual)
            correct += (pred == actual)
        end = time.time()
        storage = len(predictor.table) * 2 // 8  # 2 bits per entry
        return correct/len(trace), (end - start), storage

    elif predictor_name == 'tage':
        start = time.time()
        for addr, taken in trace:
            pred, _ = tage.predict(addr)
            actual = 1 if taken else -1
            tage.update(addr, taken)
            correct += (pred == actual)
        end = time.time()
        storage = sum(len(t) for t in tage.tables) * 3  # 1B tag + 2b counter
        return correct/len(trace), (end - start), storage

# --- Learning Curve: Accuracy over time ---
def learning_curve(predictor_name, trace, window=20):
    curve = []
    for i in range(window, len(trace)+1, window):
        subtrace = trace[:i]
        acc, _, _ = evaluate_predictor(predictor_name, subtrace)
        curve.append(acc)
    return curve

# --- Flip Response: How fast predictor adapts to sudden change ---
def flip_adapt_time(predictor_name, history_len=8):
    trace = [[f"{i:X}", 1 if i < 50 else 0] for i in range(100)]
    acc_window = []
    predictor = SaturatingCounterPredictor() if predictor_name == 'counter' else None
    p_table = {}
    tage = TAGEPredictor()
    history = deque([0]*history_len, maxlen=history_len)

    for i, (addr, taken) in enumerate(trace):
        actual = 1 if taken else -1

        if predictor_name == 'counter':
            pred = predictor.predict(addr)
            predictor.update(addr, actual)
        elif predictor_name == 'perceptron':
            idx = hash(addr) % 128
            if idx not in p_table:
                p_table[idx] = Perceptron(history_len)
            p = p_table[idx]
            pred, summ = p.predict(history)
            p.update(pred, actual, history, summ)
            history.appendleft(actual)
        elif predictor_name == 'tage':
            pred, _ = tage.predict(addr)
            tage.update(addr, taken)

        correct = (pred == actual)
        acc_window.append(correct)

        if i >= 55 and sum(acc_window[-5:]) >= 4:
            return i - 50  # how many instr after flip to recover

    return 50  # worst case, didn't recover

# === Main Comparison ===
predictors = ['counter', 'perceptron', 'tage']
modes = ['biased', 'random', 'alternating', 'periodic']
history_len = 8

for mode in modes:
    trace = generate_trace(mode=mode, num_entries=200)
    print(f"\n=== MODE: {mode.upper()} ===")
    for predictor in predictors:
        acc, latency, storage = evaluate_predictor(predictor, trace, history_len)
        learn = learning_curve(predictor, trace)
        flip = flip_adapt_time(predictor, history_len)
        print(f"{predictor.title():<10} | Acc: {acc:.2f} | Time: {latency*1000:.1f}ms | Storage: {storage}B | Flip Recovery: {flip} instr")
        plt.plot(range(20, 201, 20), learn, label=predictor)

    plt.title(f"Learning Curve - {mode.upper()} trace")
    plt.xlabel("Instructions Seen")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # === Collect data for accuracy and storage plots ===
acc_data = {mode: {} for mode in modes}
storage_data = {mode: {} for mode in modes}

for mode in modes:
    trace = generate_trace(mode=mode, num_entries=200)
    for predictor in predictors:
        acc, _, storage = evaluate_predictor(predictor, trace, history_len)
        acc_data[mode][predictor] = acc
        storage_data[mode][predictor] = storage

# === Plot Accuracy Comparison ===
plt.figure(figsize=(10, 5))
for mode in modes:
    plt.plot(predictors, [acc_data[mode][p] for p in predictors],
             marker='o', label=f"{mode}")
plt.title("Accuracy Comparison across Predictors")
plt.xlabel("Predictor")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Plot Storage Comparison ===
plt.figure(figsize=(10, 5))
for mode in modes:
    plt.plot(predictors, [storage_data[mode][p] for p in predictors],
             marker='s', label=f"{mode}")
plt.title("Storage Comparison across Predictors")
plt.xlabel("Predictor")
plt.ylabel("Estimated Storage (Bytes)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()