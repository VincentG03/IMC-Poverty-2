import numpy as np
import pandas as pd

# Step 1: Define exchange matrix
A = np.array([
    [1.0,   1.45, 0.52, 0.72],
    [0.7,   1.0,  0.31, 0.48],
    [1.95,  3.1,  1.0,  1.49],
    [1.34,  1.98, 0.64, 1.0]
])

currency_names = ['Snowballs', "Pizza's", 'Silicon Nuggets', 'SeaShells']
n = A.shape[0]
max_steps = 5
start_currency = 3  # SeaShells index

# Step 2: Init paths matrix
# paths[i][j] = list of paths from i to j
paths = [[[] for _ in range(n)] for _ in range(n)]
for i in range(n):
    for j in range(n):
        paths[i][j] = [[i, j]]

results = []

# Step 3: Build paths using dynamic programming
for step in range(2, max_steps + 1):  # steps = number of trades
    new_paths = [[[] for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for path in paths[i][k]:
                    if len(path) == step:  # only extend paths of correct length
                        new_path = path + [j]
                        new_paths[i][j].append(new_path)

    paths = new_paths

    # Step 4: Check SeaShells → ... → SeaShells cycles
    for path in paths[start_currency][start_currency]:
        if len(path) == step + 1:  # nodes = trades + 1
            val = 1.0
            for a, b in zip(path[:-1], path[1:]):
                val *= A[a][b]
            results.append({
                "path": path,
                "path_names": [currency_names[i] for i in path],
                "value": val,
                "trades": step
            })

# Step 5: Sort and display all valid cycles
df = pd.DataFrame(results)
df_sorted = df.sort_values(by='value', ascending=False)

for idx, row in df_sorted.iterrows():
    print(f"Cycle {idx + 1}:")
    print(" -> ".join(row['path_names']))
    print(f"Return Value: {row['value']:.6f}, Trades: {row['trades']}")
    print()