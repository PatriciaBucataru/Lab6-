import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

G = nx.Graph()

with open('ca-AstroPh.txt','r') as f:
    for line in f:
        if line.startswith('#'):
            continue 

        parts = line.strip().split()
        if len(parts) >= 2:
            node1, node2 = int(parts[0]), int(parts[1])

            if G.has_edge(node1, node2):
                G[node1][node2]["weight"]+=1

            else:  
                G.add_edge(node1,node2, weight=1)


print(f"Nodes: {G.number_of_nodes()},Edges:{G.number_of_edges()}")

for node in G.nodes():
    egonet = nx.ego_graph(G, node, radius=1)

    N_i = G.degree(node)
    E_i = egonet.number_of_edges()
    W_i = sum(egonet[u][v].get("weight",1) for u, v in egonet.edges())

    adj_matrix = nx.adjacency_matrix(egonet, weight="weight").todense()
    eigenvalues = np.linalg.eigvalsh(adj_matrix)
    lambda_w_i = eigenvalues[-1]

    G.nodes[node]["N_i"] = N_i
    G.nodes[node]["E_i"] = E_i
    G.nodes[node]["W_i"] = W_i
    G.nodes[node]["lambda_w_i"] = lambda_w_i

print("Features extracted for all nodes")   

nodes_list = list(G.nodes())
N_values = np.array([G.nodes[node]["N_i"] for node in nodes_list])
E_values = np.array([G.nodes[node]["E_i"] for node in nodes_list])

valid_mask = (N_values > 0) & ( E_values > 0)
valid_nodes = [nodes_list[i] for i in range(len(nodes_list)) if valid_mask[i]]
N_valid = N_values[valid_mask]
E_valid = E_values[valid_mask]

log_N = np.log(N_valid).reshape(-1,1)
log_E = np.log(E_valid)

model = LinearRegression()
model.fit(log_N, log_E)

log_E_pred = model.predict(log_N)
E_pred = np.exp (log_E_pred)

for i, node in enumerate(valid_nodes):
    y_i=E_valid[i]
    pred_i = E_pred[i]
    score_i = (max(y_i,pred_i)/min(y_i,pred_i)) * np.log(abs(y_i - pred_i)+1)
    G.nodes[node]["anomaly_score"] = score_i


for node in nodes_list:
    if node not in valid_nodes:
        G.nodes[node]["anomaly_score"] = 0

print("Anomaly scores computed")


sorted_nodes = sorted(G.nodes(),key=lambda n:G.nodes[n]["anomaly_score"],reverse=True)
top_10_anomalies = set(sorted_nodes[:10])

print(f"top 10 anomalous nodes: {list(top_10_anomalies)}")
node_colors = ['red' if node in top_10_anomalies else 'blue' for node in G.nodes()]

plt.figure(figsize=(12,8))
nx.draw(G,node_color=node_colors,node_size=50, with_labels=False)
plt.title("graph with top 10 anomalous nodes")
plt.show()

features_for_lof = np.array([[G.nodes[node]["E_i"],G.nodes[node]["N_i"]] for node in nodes_list])

lof = LocalOutlierFactor(n_neighbors=20)
lof_scores = -lof.fit_predict(features_for_lof)
lof_scores = lof.negative_outlier_factor_
lof_scores = -lof_scores

original_scores = np.array([G.nodes[node]["anomaly_score"]for node in nodes_list])
normalized_original = ( original_scores - original_scores.min()) / (original_scores.max() - original_scores.min()+1e-10)

normalized_lof= (lof_scores-lof_scores.min())/(lof_scores.max() - lof_scores.min()+1e-10)

for i, node in enumerate(nodes_list):
    combined_score = normalized_original[i] + normalized_lof[i]
    G.nodes[node]["combined_anomaly_score"] = combined_score

print("combined anomaly scores computed")

sorted_nodes_combined = sorted(G.nodes(),key=lambda n: G.nodes[n]["combined_anomaly_score"],reverse=True)
top_10_combined = set(sorted_nodes_combined[:10])

print(f"top 10 anomalous nodes (combined): {list(top_10_combined)}")

node_colors_combined = ['red' if node in top_10_combined else 'blue' for node in G.nodes()]
plt.figure(figsize=(12,8))
nx.draw(G, node_color=node_colors_combined, node_size=50, with_labels=False)
plt.title("graph with top 10 anomalous nodes = combined score ")
plt.show()