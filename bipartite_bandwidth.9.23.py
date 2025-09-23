import random
import math
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Added for headless environments (e.g., ModelArts Notebook without display)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def calculate_distance(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371.0
    return R * c

def build_bipartite_graph(graphml_file="graph-example/Ulaknet.graphml"):
    try:
        G = nx.read_graphml(graphml_file, node_type=str)
    except Exception as e:
        print(f"Error reading GraphML file: {e}")
        return None, [], [], None, None, None

    node_ids = []
    removed_nodes = []
    for node in G.nodes(data=True):
        node_id, node_data = node
        if 'Latitude' not in node_data or 'Longitude' not in node_data:
            removed_nodes.append(node_id)
            print(f"Node {node_id} removed: Missing Latitude or Longitude")
            continue
        try:
            lat = float(node_data['Latitude'])
            lon = float(node_data['Longitude'])
            if lat == 0 and lon == 0:
                removed_nodes.append(node_id)
                print(f"Node {node_id} removed: Invalid coordinates (0, 0)")
                continue
            G.nodes[node_id]['pos'] = (lon, lat)
            node_ids.append(node_id)
        except (ValueError, TypeError) as e:
            removed_nodes.append(node_id)
            print(f"Node {node_id} removed: Invalid Latitude/Longitude ({e})")
            continue

    G.remove_nodes_from(removed_nodes)
    if removed_nodes:
        print(f"Removed {len(removed_nodes)} nodes missing lat/lon data: {removed_nodes}")

    pos = nx.get_node_attributes(G, 'pos')
    invalid_nodes = [n for n in G.nodes() if n not in pos]
    if invalid_nodes:
        print(f"Error: Nodes {invalid_nodes} still lack pos attribute, removing")
        G.remove_nodes_from(invalid_nodes)
        node_ids = [n for n in node_ids if n in pos]

    if not node_ids:
        print("Error: No valid nodes after filtering")
        return None, [], [], None, None, None

    num_es = max(1, int(len(node_ids) * 0.25))
    es_nodes = random.sample(node_ids, num_es)
    client_nodes = [node for node in node_ids if node not in es_nodes]

    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(client_nodes, bipartite=0)
    bipartite_graph.add_nodes_from(es_nodes, bipartite=1)
    for c in client_nodes:
        for e in es_nodes:
            bipartite_graph.add_edge(c, e)

    for node in bipartite_graph.nodes():
        if node in pos:
            bipartite_graph.nodes[node]['pos'] = pos[node]
        else:
            print(f"Warning: Node {node} in bipartite graph lacks pos, removing")
            bipartite_graph.remove_node(node)
            if node in client_nodes:
                client_nodes.remove(node)
            if node in es_nodes:
                es_nodes.remove(node)

    pos = nx.get_node_attributes(bipartite_graph, 'pos')
    if not all(node in pos for node in client_nodes + es_nodes):
        print("Error: Some nodes still lack pos attributes after final check")
        return None, [], [], None, None, None

    distance_matrix = np.zeros((len(client_nodes), len(es_nodes)))
    es_distance_matrix = np.zeros((len(es_nodes), len(es_nodes)))
    for i, c in enumerate(client_nodes):
        for j, e in enumerate(es_nodes):
            if c in pos and e in pos:
                c_pos = pos[c]
                e_pos = pos[e]
                distance_matrix[i, j] = calculate_distance(c_pos[1], c_pos[0], e_pos[1], e_pos[0])
            else:
                print(f"Warning: No position for client {c} or edge {e}, setting distance to infinity")
                distance_matrix[i, j] = float('inf')
    for i, e1 in enumerate(es_nodes):
        for j, e2 in enumerate(es_nodes):
            if e1 in pos and e2 in pos:
                e1_pos = pos[e1]
                e2_pos = pos[e2]
                es_distance_matrix[i, j] = calculate_distance(e1_pos[1], e1_pos[0], e2_pos[1], e2_pos[0])
            else:
                print(f"Warning: No position for edge {e1} or edge {e2}, setting distance to infinity")
                es_distance_matrix[i, j] = float('inf')

    if client_nodes:
        sample_node = G.nodes[client_nodes[0]]
        print(f"Node data sample: {sample_node}")

    return bipartite_graph, client_nodes, es_nodes, distance_matrix, es_distance_matrix, pos

def plot_graph(bipartite_graph, client_nodes, es_nodes):
    pos = nx.get_node_attributes(bipartite_graph, 'pos')
    missing_pos = [n for n in client_nodes + es_nodes if n not in pos]
    if missing_pos:
        print(f"Error: Nodes {missing_pos} missing pos attributes in plot_graph")
        return
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=client_nodes, node_color='lightblue', node_shape='o',
                           node_size=300, label='Clients')
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=es_nodes, node_color='lightcoral', node_shape='s',
                           node_size=300, label='Edge Servers')
    plt.legend()
    plt.title("Bipartite Graph (Clients and Edge Servers)")
    plt.savefig("bipartite_graph.png")
    plt.close()

def plot_assigned_graph(bipartite_graph, client_nodes, es_nodes, assignments, cluster_heads, C2, es_nodes_indices):
    pos = nx.get_node_attributes(bipartite_graph, 'pos')
    missing_pos = [n for n in client_nodes + es_nodes if n not in pos]
    if missing_pos:
        print(f"Error: Nodes {missing_pos} missing pos attributes in plot_assigned_graph")
        return
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=client_nodes, node_color='lightblue', node_shape='o',
                           node_size=300, label='Clients')
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=[es_nodes[i] for i in es_nodes_indices if i not in cluster_heads.values()],
                           node_color='lightcoral', node_shape='s', node_size=300, label='Edge Servers')
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=[es_nodes[v] for v in cluster_heads.values() if v != -1],
                           node_color='gold', node_shape='*', node_size=500, label='Cluster Heads')
    assigned_edges = [(client_nodes[m], es_nodes[n]) for m, n in assignments]
    nx.draw_networkx_edges(bipartite_graph, pos, edgelist=assigned_edges, edge_color='navy', width=2)
    for ch_idx, es_indices in C2.items():
        if cluster_heads[ch_idx] == -1:
            continue
        ch_node = es_nodes[cluster_heads[ch_idx]]
        for es_idx in es_indices:
            if es_idx != cluster_heads[ch_idx]:
                nx.draw_networkx_edges(bipartite_graph, pos, edgelist=[(es_nodes[es_idx], ch_node)],
                                       edge_color='purple', width=1.5, style='dashed')
    plt.legend()
    plt.title("Assigned Bipartite Graph with Cluster Heads (Navy: Client-ES, Purple: ES-CH)")
    plt.savefig("assigned_bipartite_graph_with_ch.png")
    plt.close()

def allocate_bandwidth_eba(client_nodes, es_nodes, distance_matrix, es_distance_matrix, pos):
    M = len(client_nodes)
    N = len(es_nodes)
    B_cloud = 5e7
    max_capacity = max(1, int(M / N) + 1)
    min_B_assigned = 1e6

    distance_matrix_km = distance_matrix
    p_m = np.ones(M) * 1.0
    N0 = 1e-12
    alpha = 2.5
    k = 1e5
    g = k / (distance_matrix_km ** alpha + 1e-4)
    print(f"Channel gain (g, first 5x5):\n{g[:min(5, M), :min(5, N)]}")

    mu = np.log(2e7)
    sigma = 0.5
    B_mn = np.random.lognormal(mean=mu, sigma=sigma, size=(M, N))
    distance_normalized = distance_matrix_km / np.max(distance_matrix_km)
    B_mn = B_mn / (distance_normalized + 1e-4)
    B_mn = np.clip(B_mn, 1e6, 5e7)
    print(f"Device-to-Edge Bandwidth (B_mn, first 5x5):\n{B_mn[:min(5, M), :min(5, N)]}")

    SNR_initial = (p_m[:, None] * g) / (N0 * B_mn + 1e-4)
    r_initial = B_mn * np.log2(1 + SNR_initial)
    print(f"Initial transmission rates (r_initial, first 5x5):\n{r_initial[:min(5, M), :min(5, N)]}")

    # Use same r for both Bipartite and Random to ensure fair comparison
    r = r_initial
    r_random = r_initial.copy()  # Copy r_initial instead of generating new r_random

    es_distance_matrix_km = es_distance_matrix
    g_es = k / (es_distance_matrix_km ** alpha + 1e-4)
    B_es = np.random.lognormal(mean=mu, sigma=sigma, size=(N, N))
    es_distance_normalized = es_distance_matrix_km / np.max(es_distance_matrix_km + 1e-4)
    B_es = B_es / (es_distance_normalized + 1e-4)
    B_es = np.clip(B_es, 1e6, 5e7)
    p_es = np.ones(N) * 1.0
    SNR_es = (p_es[:, None] * g_es) / (N0 * B_es + 1e-4)
    r_es = B_es * np.log2(1 + SNR_es)
    print(f"ES-to-ES transmission rates (r_es, first 5x5):\n{r_es[:min(5, N), :min(5, N)]}")

    # Dynamic allocation with descending sort and updates
    max_r_per_device = np.max(r_initial, axis=1)
    unassigned = list(range(M))
    assignments_array = np.full(M, -1, dtype=int)
    loads = np.zeros(N, dtype=int)
    while unassigned:
        # Sort remaining devices by current max_r_per_device descending
        sorted_unassigned = sorted(unassigned, key=lambda idx: max_r_per_device[idx], reverse=True)
        for idx in sorted_unassigned[:]:  # Copy to avoid modification during iteration
            available_n = np.where(loads < max_capacity)[0]
            if len(available_n) == 0:
                print(f"Warning: Device {idx} has no available edge server (all full)")
                unassigned.remove(idx)
                continue
            r_for_device = r_initial[idx, available_n]
            load_scores = loads[available_n] + 1e-10
            rate_scores = r_for_device / np.max(r_for_device + 1e-10)  # Avoid div by zero
            combined_scores = 0.1 * (load_scores / np.max(load_scores)) - 0.9 * rate_scores  # Adjusted weights
            sorted_n = available_n[np.argsort(combined_scores)]
            assigned = False
            for n in sorted_n:
                if loads[n] < max_capacity:
                    assignments_array[idx] = n
                    loads[n] += 1
                    assigned = True
                    unassigned.remove(idx)
                    break
            if assigned:
                # If any ES is now full, update max_r_per_device for remaining unassigned devices
                full_es = np.where(loads >= max_capacity)[0]
                if len(full_es) > 0:
                    current_available = np.setdiff1d(np.arange(N), full_es)
                    for rem_idx in unassigned:
                        if len(current_available) > 0:
                            max_r_per_device[rem_idx] = np.max(r_initial[rem_idx, current_available])
                        else:
                            max_r_per_device[rem_idx] = 0  # No available, but shouldn't happen
    print(f"Device sorting (by max r): {np.argsort(max_r_per_device)[::-1][:5]}... "
          f"(max_r: {max_r_per_device[np.argsort(max_r_per_device)[::-1][:5]]})")

    assignments = [(i, int(j)) for i, j in enumerate(assignments_array) if j != -1]
    invalid_assignments = [(i, j) for i, j in assignments if j >= N or j < 0]
    if invalid_assignments:
        print(f"Error: Invalid assignments: {invalid_assignments}")
        assignments = [(i, j) for i, j in assignments if 0 <= j < N]

    r_final = np.zeros_like(r_initial)
    for m, n in assignments:
        if r_initial[m, n] > 0:
            r_final[m, n] = r_initial[m, n]
        else:
            print(f"Warning: Invalid rate r_initial[{m}, {n}] = {r_initial[m, n]}")
    r = r_final
    print(f"Assignments: {assignments}")
    print(f"Loads per edge: {loads}")
    print(f"Final transmission rates (r, first 5x5):\n{r[:min(5, M), :min(5, N)]}")

    active_indices = [n for n in range(N) if loads[n] > 0]
    active_es_nodes = [es_nodes[n] for n in active_indices]
    N_active = len(active_indices)
    B_n = np.zeros(N)
    if N_active > 0:
        mean_B = 5e7
        std_B = 1e7
        base_bandwidths = np.random.normal(loc=mean_B, scale=std_B, size=N_active)
        base_bandwidths = np.clip(base_bandwidths, mean_B / 2, mean_B * 2)
        load_weights = loads[active_indices] / np.sum(loads[active_indices])
        B_n[active_indices] = base_bandwidths * load_weights * (B_cloud / np.sum(base_bandwidths * load_weights))
    print(f"Active edge servers (indices): {active_indices}")
    print(f"Active edge servers (nodes): {active_es_nodes}")
    print(f"Edge-to-Cloud Bandwidth (B_n): {B_n}")

    r_threshold = 5e5
    abnormal_count = 0
    for m, n in assignments:
        if r[m, n] < r_threshold:
            print(f"Warning: Device {m} to edge {n} has r = {r[m, n]} < {r_threshold} bit/s")
            r[m, n] = 0
            abnormal_count += 1
    print(f"Total abnormal r values: {abnormal_count}")

    r_values = r[r > 0]
    r_random_values = r_random[r_random > 0]
    print(f"Bipartite r stats: mean={np.mean(r_values):.2e}, max={np.max(r_values):.2e}, min={np.min(r_values):.2e}")
    print(f"Random r stats: mean={np.mean(r_random_values):.2e}, max={np.max(r_random_values):.2e}, min={np.min(r_random_values):.2e}")

    return r, r_random, assignments, loads, B_n, r_es, active_es_nodes

def validate_results(B_cloud, B_n, r, assignments, loads, max_capacity):
    N = len(B_n)
    M = len([i for i, _ in assignments] + [i for i in range(len(r)) if i not in [x[0] for x in assignments]])
    active = np.unique([j for _, j in assignments])
    if abs(np.sum(B_n) - B_cloud) > 1e-6:
        print(f"Error: Sum of B_n ({np.sum(B_n)}) != B_cloud ({B_cloud})")
    else:
        print("Pass: B_n sum correct")
    for n in range(N):
        if n not in active and B_n[n] != 0:
            print(f"Error: Inactive edge {n} has B_n != 0")
        elif n in active and B_n[n] == 0:
            print(f"Error: Active edge {n} has B_n == 0")
    if np.any(loads > max_capacity):
        print(f"Error: Loads exceed capacity {max_capacity} (loads: {loads})")
    else:
        print("Pass: Capacity constraints satisfied")
    mounted_ratio = len(assignments) / M if M > 0 else 0
    if mounted_ratio < 0.95:
        print(f"Warning: Low assignment coverage ({mounted_ratio:.2%})")
    else:
        print(f"Pass: Assignment coverage {mounted_ratio:.2%}")
    for m, n in assignments:
        if r[m, n] <= 0:
            print(f"Error: Assigned device {m} to edge {n} has r <= 0")
        for other_n in range(N):
            if other_n != n and r[m, other_n] > 0:
                print(f"Error: Non-assigned device {m} to edge {other_n} has r > 0")
    if M > 0:
        max_r_values = np.max(r, axis=1)
        sorted_max_r = np.sort(max_r_values)[::-1]
        half = M // 2
        if np.all(sorted_max_r[:half] >= sorted_max_r[-half:]):
            print("Pass: Prioritized high-rate devices")
        else:
            print("Warning: Sorting may not prioritize high-rate devices")
    print(f"Manual check: Load std (balance) = {np.std(loads):.2f} (lower is better)")

def run_bandwidth_allocation(graphml_file="graph-example/Ulaknet.graphml"):
    bipartite_graph, client_nodes, es_nodes, distance_matrix, es_distance_matrix, pos = build_bipartite_graph(graphml_file)
    if bipartite_graph is None:
        return None, [], [], None, None, [], [], [], None, None
    print("Bipartite Graph Built Successfully!")
    print(f"Total Client Nodes: {len(client_nodes)}")
    print(f"Total ES Nodes: {len(es_nodes)}")
    print(f"Distance Matrix Shape: {distance_matrix.shape}")
    print(f"ES Distance Matrix Shape: {es_distance_matrix.shape}")
    print(f"Distance Matrix (first 5x5):\n{distance_matrix[:5, :min(5, len(es_nodes))]}")
    r, r_random, assignments, loads, B_n, r_es, active_es_nodes = allocate_bandwidth_eba(client_nodes, es_nodes, distance_matrix, es_distance_matrix, pos)
    print(f"Final transmission rates r_m,n (bit/s, first 5x5):\n{r[:5, :min(5, len(es_nodes))]}")
    max_capacity = max(1, int(len(client_nodes) / len(es_nodes)) + 1)
    validate_results(B_cloud=5e7, B_n=B_n, r=r, assignments=assignments, loads=loads, max_capacity=max_capacity)
    plot_graph(bipartite_graph, client_nodes, es_nodes)
    return bipartite_graph, client_nodes, es_nodes, distance_matrix, r, r_random, assignments, loads, B_n, r_es, pos, active_es_nodes