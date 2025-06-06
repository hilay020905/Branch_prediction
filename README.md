# 🧠 Branch Prediction Models Comparison

This project compares the performance of three branch prediction models:

- 🔁 **2-bit Saturating Counter**
- 🧮 **Perceptron Model**
- 🧷 **TAGE (Tagged Geometric History Length) Model**

The predictors are evaluated using **trace-driven simulation** and analyzed using Python visualizations.

---

## 📁 Project Structure

### 📁 docs
| 📄 File Name | 📝 Description |
|-------------|----------------|
| `PERCEPTRON_MODEL.py` | 🧮 Implements the perceptron-based branch predictor. |
| `TAGE_MODEL.py` | 🧷 Implements the TAGE branch predictor. |
| `SATURATING_COUNTER.py` | 🔁 Implements a 2-bit saturating counter predictor. |
| `PLOT.py` | 📊 Plots accuracy, learning curve, storage, time, and flip recovery using `matplotlib`. |
| `history_trace.pdf` | 📄 Input trace file for simulation. |

---

## 📊 Metrics Visualized

The following performance metrics are plotted and compared:

- 🎯 **Accuracy** – Correct predictions vs total
- ⏱️ **Time** – Execution time per predictor
- 💾 **Storage** – Memory required (tables/registers)
- 🔄 **Flip Recovery** – Time to recover from a misprediction
- 📈 **Learning Curve** – Accuracy improvement over time

---

## 🌀 Trace Types Explained

| 🔖 Trace Type | 📘 Meaning |
|--------------|------------|
| 🎯 **Biased** | Mostly predictable (e.g., 90% taken or 90% not taken) |
| 🎲 **Random** | 50/50 mix — no pattern |
| 🔁 **Alternating** | Switches between taken and not taken each time |
| 🔄 **Periodic** | Repeats a fixed pattern like TTNTTNTT... |

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install matplotlib
