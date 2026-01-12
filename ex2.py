import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random


G1 = nx.random_regular_graph(3, 100)
G2 = nx.random_regular_graph(5, 100)


G = nx.union(G1, G2, rename=('g1-', 'g2-'))
for edge in G.edges():
    G.add_edge(edge[0], edge[1], weight=1)

print(f"merged graph - nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

random_nodes = random.sample(list(G.nodes()), 2)
print(f"heavy vicinity nodes: {random_nodes}")

for node in random_nodes:
    egonet = nx.ego_graph(G, node, radius=1)
    for u, v in egonet.edges():
        G[u][v]["weight"] += 10


for node in G.nodes():
    egonet = nx.ego_graph(G, node, radius=1)
    
    E_i = egonet.number_of_edges()
    W_i = sum(egonet[u][v].get("weight", 1) for u, v in egonet.edges())
    
    G.nodes[node]["E_i"] = E_i
    G.nodes[node]["W_i"] = W_i


nodes_list = list(G.nodes())
E_values = np.array([G.nodes[node]["E_i"] for node in nodes_list])
W_values = np.array([G.nodes[node]["W_i"] for node in nodes_list])

valid_mask = (E_values > 0) & (W_values > 0)
valid_nodes = [nodes_list[i] for i in range(len(nodes_list)) if valid_mask[i]]
E_valid = E_values[valid_mask]
W_valid = W_values[valid_mask]

log_E = np.log(E_valid).reshape(-1, 1)
log_W = np.log(W_valid)

model = LinearRegression()
model.fit(log_E, log_W)

log_W_pred = model.predict(log_E)
W_pred = np.exp(log_W_pred)


for i, node in enumerate(valid_nodes):
    y_i = W_valid[i]
    pred_i = W_pred[i]
    score_i = (max(y_i, pred_i) / min(y_i, pred_i)) * np.log(abs(y_i - pred_i) + 1)
    G.nodes[node]["anomaly_score"] = score_i

for node in nodes_list:
    if node not in valid_nodes:
        G.nodes[node]["anomaly_score"] = 0


sorted_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n]["anomaly_score"], reverse=True)
top_4_heavy = set(sorted_nodes[:4])

print(f"top 4 heavy vicinity nodes detected: {list(top_4_heavy)}")


node_colors = ['red' if node in top_4_heavy else 'blue' for node in G.nodes()]

plt.figure(figsize=(12, 8))
nx.draw(G, node_color=node_colors, node_size=30, with_labels=False)
plt.title("graph with top 4 heavy vicinity nodes (red)")
plt.show()