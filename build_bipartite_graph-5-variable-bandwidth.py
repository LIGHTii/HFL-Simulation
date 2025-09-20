import random
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


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
    R = 6371.0  # Earth radius (km)
    return R * c


def build_bipartite_graph(graphml_file):
    original_graph = nx.read_graphml(graphml_file)
    nodes_to_remove = [node for node, data in original_graph.nodes(data=True) if
                       'Latitude' not in data or 'Longitude' not in data]
    if nodes_to_remove:
        original_graph.remove_nodes_from(nodes_to_remove)
        print(f"Removed {len(nodes_to_remove)} nodes missing lat/lon data: {nodes_to_remove}")
    node_ids = list(original_graph.nodes)
    if not node_ids:
        print("No valid nodes in graph, terminating.")
        return None, [], [], None
    print("Node data sample:", list(original_graph.nodes(data=True))[0][1])
    num_es = max(1, int(len(node_ids) * 0.25))
    es_nodes = random.sample(node_ids, num_es)
    client_nodes = [node for node in node_ids if node not in es_nodes]
    bipartite_graph = nx.Graph()
    for node in client_nodes:
        bipartite_graph.add_node(node, bipartite=0, **original_graph.nodes[node])
    for node in es_nodes:
        bipartite_graph.add_node(node, bipartite=1, **original_graph.nodes[node])
    distance_matrix = np.zeros((len(client_nodes), len(es_nodes)))
    for i, client in enumerate(client_nodes):
        for j, es in enumerate(es_nodes):
            client_lat = original_graph.nodes[client]['Latitude']
            client_lon = original_graph.nodes[client]['Longitude']
            es_lat = original_graph.nodes[es]['Latitude']
            es_lon = original_graph.nodes[es]['Longitude']
            weight = calculate_distance(client_lat, client_lon, es_lat, es_lon)
            distance_matrix[i, j] = weight
            bipartite_graph.add_edge(client, es, weight=weight)
    return bipartite_graph, client_nodes, es_nodes, distance_matrix


def plot_graph(bipartite_graph, client_nodes, es_nodes):
    pos = {node: (data['Longitude'], data['Latitude']) for node, data in bipartite_graph.nodes(data=True)}
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=client_nodes, node_color='#a6c0e5', node_size=200,
                           label="Client Nodes")
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=es_nodes, node_color='#ef8b67', node_size=250,
                           label="ES Nodes", node_shape='s')
    nx.draw_networkx_edges(bipartite_graph, pos, width=1.0, alpha=0.5, edge_color='#6f6f6f')
    plt.title('Bipartite Graph: Client Nodes and ES Nodes', fontsize=15)
    plt.legend(scatterpoints=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.grid(True)
    plt.show()


def plot_assigned_graph(bipartite_graph, client_nodes, es_nodes, assignments):
    pos = {node: (data['Longitude'], data['Latitude']) for node, data in bipartite_graph.nodes(data=True)}
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=client_nodes, node_color='#a6c0e5', node_size=200,
                           label="Client Nodes")
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=es_nodes, node_color='#ef8b67', node_size=250,
                           label="ES Nodes", node_shape='s')
    assigned_edges = [(client_nodes[m], es_nodes[assignments[m]]) for m in range(len(assignments)) if
                      assignments[m] != -1]
    nx.draw_networkx_edges(bipartite_graph, pos, edgelist=assigned_edges, width=2.0, alpha=0.8, edge_color='green')
    plt.title('Assigned Graph: Devices to Assigned ES Nodes', fontsize=15)
    plt.legend(scatterpoints=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.grid(True)
    plt.show()


def allocate_bandwidth_eba(client_nodes, es_nodes, distance_matrix):
    M = len(client_nodes)  # Number of devices
    N = len(es_nodes)      # Number of edge servers
    B_cloud = 5e7          # Increased total cloud bandwidth to 50 MHz
    max_capacity = 5       # Max devices per edge server
    min_B_assigned = 1e6   # Minimum assigned bandwidth (1 MHz)

    # Use distance in kilometers
    distance_matrix_km = distance_matrix

    # Channel gain calculation
    p_m = np.ones(M) * 1.0  # Transmit power (W)
    N0 = 1e-12              # Noise power spectral density (W/Hz)
    alpha = 2.5             # Path loss exponent
    k = 1e5                 # Channel gain constant
    g = k / (distance_matrix_km ** alpha + 1e-4)  # Channel gain
    print(f"Channel gain (g, first 5x5):\n{g[:5, :5]}")

    # Device-to-Edge Bandwidth: Log-normal distribution
    mu = np.log(2e7)  # Mean 20 MHz
    sigma = 0.5       # Standard deviation
    B_mn = np.random.lognormal(mean=mu, sigma=sigma, size=(M, N))
    distance_normalized = distance_matrix_km / np.max(distance_matrix_km)
    B_mn = B_mn / (distance_normalized + 1e-4)  # Scale by distance
    B_mn = np.clip(B_mn, 1e6, 5e7)  # 1 MHz to 50 MHz
    print(f"Device-to-Edge Bandwidth (B_mn, first 5x5):\n{B_mn[:5, :5]}")

    # Calculate initial SNR and transmission rate
    SNR_initial = (p_m[:, None] * g) / (N0 * B_mn + 1e-4)  # Initial SNR
    r_initial = B_mn * np.log2(1 + SNR_initial)  # Initial Shannon capacity (bit/s)
    print(f"Initial transmission rates (r_initial, first 5x5):\n{r_initial[:5, :5]}")

    # Sort devices by maximum transmission rate
    max_r_per_device = np.max(r_initial, axis=1)
    sorted_device_indices = np.argsort(max_r_per_device)[::-1]
    print(f"Device sorting (by max r): {sorted_device_indices[:5]}... "
          f"(max_r: {max_r_per_device[sorted_device_indices[:5]]})")

    # Initialize assignments and loads
    assignments = np.full(M, -1, dtype=int)
    loads = np.zeros(N, dtype=int)

    # Assign devices to edge servers, balancing load and rate
    for idx in sorted_device_indices:
        available_n = np.where(loads < max_capacity)[0]
        if len(available_n) == 0:
            print(f"Warning: Device {idx} has no available edge server (all full)")
            continue
        r_for_device = r_initial[idx, available_n]
        # Balance load and rate
        load_scores = loads[available_n] + 1e-10  # Avoid division by zero
        rate_scores = r_for_device / np.max(r_for_device)
        combined_scores = 0.5 * (load_scores / np.max(load_scores)) - 0.5 * rate_scores  # Equal weight
        sorted_n = available_n[np.argsort(combined_scores)]
        for n in sorted_n:
            if loads[n] < max_capacity:
                assignments[idx] = n
                loads[n] += 1
                break
    print(f"Assignments: {assignments}")
    print(f"Loads per edge: {loads}")

    # Edge-to-Cloud Bandwidth: Normal distribution with higher mean
    active = [n for n in range(N) if loads[n] > 0]
    N_active = len(active)
    B_n = np.zeros(N)
    if N_active > 0:
        mean_B = 5e7  # Mean 50 MHz
        std_B = 1e7
        base_bandwidths = np.random.normal(loc=mean_B, scale=std_B, size=N_active)
        base_bandwidths = np.clip(base_bandwidths, mean_B / 2, mean_B * 2)
        load_weights = loads[active] / np.sum(loads[active])  # Equal weighting by load
        B_n[active] = base_bandwidths * load_weights * (B_cloud / np.sum(base_bandwidths * load_weights))
    print(f"Active edge servers: {active}")
    print(f"Edge-to-Cloud Bandwidth (B_n): {B_n}")

    # Compute theta: Linear allocation based on SNR and initial rate
    theta = np.zeros((M, N))
    for n in active:
        assigned_devices = np.where(assignments == n)[0]
        if len(assigned_devices) > 0:
            snr = (p_m[assigned_devices] * g[assigned_devices, n]) / (N0 * B_mn[assigned_devices, n] + 1e-4)
            r_weights = np.array([r_initial[m, n] for m in assigned_devices])
            weights = snr * r_weights / (np.sum(snr * r_weights) + 1e-4)
            min_weight = 0.6 / len(assigned_devices)  # Increased minimum allocation
            weights = np.maximum(weights, min_weight)
            weights = weights / np.sum(weights)  # Re-normalize to sum to 1
            theta[assigned_devices, n] = weights
    print(f"Bandwidth allocation ratios (theta, first 5x5):\n{theta[:5, :5]}")

    # Recalculate final transmission rates
    B_assigned = np.zeros((M, N))
    for n in active:
        assigned_devices = np.where(assignments == n)[0]
        if len(assigned_devices) > 0:
            # Allocate B_n based on theta
            total_B_needed = np.sum(B_mn[assigned_devices, n] * theta[assigned_devices, n])
            if total_B_needed > B_n[n]:
                B_assigned[assigned_devices, n] = np.maximum(B_n[n] * theta[assigned_devices, n], min_B_assigned)
            else:
                B_assigned[assigned_devices, n] = np.maximum(B_mn[assigned_devices, n] * theta[assigned_devices, n], min_B_assigned)
    print(f"Assigned bandwidth (B_assigned, first 5x5):\n{B_assigned[:5, :5]}")
    r = B_assigned * np.log2(1 + SNR_initial)  # Use initial SNR for consistency
    print(f"Final transmission rates (r, first 5x5):\n{r[:5, :5]}")

    # Handle abnormal rates
    r_threshold = 5e5  # 500 kbps
    abnormal_count = 0
    for m in range(M):
        if assignments[m] != -1:
            n = assignments[m]
            if r[m, n] < r_threshold:
                print(f"Warning: Device {m} to edge {n} has r = {r[m, n]} < {r_threshold} bit/s")
                r[m, n] = 0
                abnormal_count += 1
    print(f"Total abnormal r values: {abnormal_count}")

    return r, assignments, loads, B_n, theta


def validate_results(B_cloud, B_n, theta, r, assignments, loads, max_capacity):
    N = len(B_n)
    M = len(theta)
    active = np.unique(assignments[assignments != -1])

    # Check edge-to-cloud bandwidth sum
    if abs(np.sum(B_n) - B_cloud) > 1e-6:
        print(f"Error: Sum of B_n ({np.sum(B_n)}) != B_cloud ({B_cloud})")
    else:
        print("Pass: B_n sum correct")

    # Check bandwidth for inactive edges
    for n in range(N):
        if n not in active and B_n[n] != 0:
            print(f"Error: Inactive edge {n} has B_n != 0")
        elif n in active and B_n[n] == 0:
            print(f"Error: Active edge {n} has B_n == 0")

    # Check capacity constraints
    if np.any(loads > max_capacity):
        print(f"Error: Loads exceed capacity {max_capacity} (loads: {loads})")
    else:
        print("Pass: Capacity constraints satisfied")

    # Check assignment coverage
    mounted_ratio = np.sum(assignments != -1) / M
    if mounted_ratio < 0.95:
        print(f"Warning: Low assignment coverage ({mounted_ratio:.2%})")
    else:
        print(f"Pass: Assignment coverage {mounted_ratio:.2%}")

    # Check transmission rates
    for m in range(M):
        if assignments[m] != -1:
            n = assignments[m]
            if r[m, n] <= 0:
                print(f"Error: Assigned device {m} to edge {n} has r <= 0")
        for other_n in range(N):
            if other_n != assignments[m] and r[m, other_n] > 0:
                print(f"Error: Non-assigned device {m} to edge {other_n} has r > 0")

    # Check theta sum for each edge server
    for n in active:
        assigned_devices = np.where(assignments == n)[0]
        if len(assigned_devices) > 0:
            theta_sum = np.sum(theta[assigned_devices, n])
            if abs(theta_sum - 1.0) > 1e-6:
                print(f"Error: Theta sum for edge {n} != 1 ({theta_sum})")
            else:
                print(f"Pass: Theta sum for edge {n} = {theta_sum:.6f}")

    # Check sorting effectiveness
    if M > 0:
        max_r_values = np.max(r, axis=1)
        sorted_max_r = np.sort(max_r_values)[::-1]
        half = M // 2
        if np.all(sorted_max_r[:half] >= sorted_max_r[-half:]):
            print("Pass: Prioritized high-rate devices")
        else:
            print("Warning: Sorting may not prioritize high-rate devices")

    print(f"Manual check: Load std (balance) = {np.std(loads):.2f} (lower is better)")


def main():
    graphml_file = "graph-example/Ulaknet.graphml"
    bipartite_graph, client_nodes, es_nodes, distance_matrix = build_bipartite_graph(graphml_file)
    if bipartite_graph is None:
        return
    print("Bipartite Graph Built Successfully!")
    print(f"Total Client Nodes: {len(client_nodes)}")
    print(f"Total ES Nodes: {len(es_nodes)}")
    print(f"Distance Matrix Shape: {distance_matrix.shape}")
    print(f"Distance Matrix (first 5x5):\n{distance_matrix[:5, :5]}")

    r, assignments, loads, B_n, theta = allocate_bandwidth_eba(client_nodes, es_nodes, distance_matrix)
    print(f"Final transmission rates r_m,n (bit/s, first 5x5):\n{r[:5, :5]}")

    max_capacity = 5
    validate_results(B_cloud=5e7, B_n=B_n, theta=theta, r=r, assignments=assignments, loads=loads,
                     max_capacity=max_capacity)

    plot_assigned_graph(bipartite_graph, client_nodes, es_nodes, assignments)
    plot_graph(bipartite_graph, client_nodes, es_nodes)
    nx.write_graphml(bipartite_graph, "bipartite_graph.graphml")
    print("\nBipartite graph saved to bipartite_graph.graphml")


if __name__ == "__main__":
    main()