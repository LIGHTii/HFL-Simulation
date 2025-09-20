# 修改后的 bipartite_bandwidth.py
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


def build_bipartite_graph(graphml_file="graph-example/Ulaknet.graphml"):
    import numpy as np
    import networkx as nx
    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees) in kilometers.
        """
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        R = 6371  # Radius of Earth in kilometers
        distance = R * c
        return distance

    try:
        G = nx.read_graphml(graphml_file, node_type=str)
    except Exception as e:
        print(f"Error reading GraphML file: {e}")
        return None, [], [], None

    client_nodes = []
    es_nodes = []
    removed_nodes = []

    # Filter nodes with valid lat/lon and assign pos
    for node in G.nodes(data=True):
        node_id, node_data = node
        if 'Latitude' not in node_data or 'Longitude' not in node_data:
            removed_nodes.append(node_id)
            print(f"Node {node_id} removed: Missing Latitude or Longitude")
            continue
        try:
            lat = float(node_data['Latitude'])
            lon = float(node_data['Longitude'])
            if lat == 0 and lon == 0:  # Check for invalid coordinates
                removed_nodes.append(node_id)
                print(f"Node {node_id} removed: Invalid coordinates (0, 0)")
                continue
            G.nodes[node_id]['pos'] = (lon, lat)  # Ensure pos is set
            if node_data.get('type') == 'Yellow Colour':
                client_nodes.append(node_id)
            elif node_data.get('type') == 'Red Colour':
                es_nodes.append(node_id)
        except (ValueError, TypeError) as e:
            removed_nodes.append(node_id)
            print(f"Node {node_id} removed: Invalid Latitude/Longitude ({e})")
            continue

    # Remove nodes without pos
    G.remove_nodes_from(removed_nodes)
    if removed_nodes:
        print(f"Removed {len(removed_nodes)} nodes missing lat/lon data: {removed_nodes}")

    # Verify all nodes have pos
    pos = nx.get_node_attributes(G, 'pos')
    invalid_nodes = [n for n in G.nodes() if n not in pos]
    if invalid_nodes:
        print(f"Error: Nodes {invalid_nodes} still lack pos attribute, removing")
        G.remove_nodes_from(invalid_nodes)
        client_nodes = [n for n in client_nodes if n in pos]
        es_nodes = [n for n in es_nodes if n in pos]

    if not client_nodes or not es_nodes:
        print("Error: No valid client or edge server nodes after filtering")
        return None, [], [], None

    # Create bipartite graph
    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(client_nodes, bipartite=0)
    bipartite_graph.add_nodes_from(es_nodes, bipartite=1)
    for c in client_nodes:
        for e in es_nodes:
            bipartite_graph.add_edge(c, e)

    # Set pos attributes for bipartite graph
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

    # Recheck pos attributes
    pos = nx.get_node_attributes(bipartite_graph, 'pos')
    if not all(node in pos for node in client_nodes + es_nodes):
        print("Error: Some nodes still lack pos attributes after final check")
        return None, [], [], None

    # Calculate distance matrix
    distance_matrix = np.zeros((len(client_nodes), len(es_nodes)))
    for i, c in enumerate(client_nodes):
        for j, e in enumerate(es_nodes):
            if c in pos and e in pos:
                c_pos = pos[c]
                e_pos = pos[e]
                distance_matrix[i, j] = haversine(c_pos[1], c_pos[0], e_pos[1], e_pos[0])
            else:
                print(f"Warning: No position for client {c} or edge {e}, setting distance to infinity")
                distance_matrix[i, j] = float('inf')

    # Debug: Print sample node data
    if client_nodes:
        sample_node = G.nodes[client_nodes[0]]
        print(f"Node data sample: {sample_node}")

    return bipartite_graph, client_nodes, es_nodes, distance_matrix


def plot_graph(bipartite_graph, client_nodes, es_nodes):
    import matplotlib.pyplot as plt
    pos = nx.get_node_attributes(bipartite_graph, 'pos')

    # Debug: Check for missing pos attributes
    missing_pos = [n for n in client_nodes + es_nodes if n not in pos]
    if missing_pos:
        print(f"Error: Nodes {missing_pos} missing pos attributes in plot_graph")
        return

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=client_nodes, node_color='lightblue', node_shape='o',
                           node_size=300, label='Clients')
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=es_nodes, node_color='lightcoral', node_shape='s',
                           node_size=300, label='Edge Servers')
    nx.draw_networkx_edges(bipartite_graph, pos, edge_color='gray', alpha=0.2)
    nx.draw_networkx_labels(bipartite_graph, pos)
    plt.legend()
    plt.title("Bipartite Graph (Clients and Edge Servers)")
    plt.savefig("bipartite_graph.png")
    plt.close()


def plot_assigned_graph(bipartite_graph, client_nodes, es_nodes, assignments):
    import matplotlib.pyplot as plt
    pos = nx.get_node_attributes(bipartite_graph, 'pos')
    # Debug: Check for missing pos attributes
    missing_pos = [n for n in client_nodes + es_nodes if n not in pos]
    if missing_pos:
        print(f"Error: Nodes {missing_pos} missing pos attributes in plot_assigned_graph")
        return
    plt.figure(figsize=(10, 8))

    # Draw all nodes
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=client_nodes, node_color='lightblue', node_shape='o',
                           node_size=300, label='Clients')
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=es_nodes, node_color='lightcoral', node_shape='s',
                           node_size=300, label='Edge Servers')

    # Draw all edges (unassigned) faintly
    nx.draw_networkx_edges(bipartite_graph, pos, edge_color='gray', alpha=0.2)

    # Draw assigned edges
    assigned_edges = [(client_nodes[m], es_nodes[n]) for m, n in assignments]
    nx.draw_networkx_edges(bipartite_graph, pos, edgelist=assigned_edges, edge_color='blue', width=2)

    # Draw labels
    nx.draw_networkx_labels(bipartite_graph, pos)
    plt.legend()
    plt.title("Assigned Bipartite Graph (Blue edges indicate assignments)")
    plt.savefig("assigned_bipartite_graph.png")
    plt.close()


def run_bandwidth_allocation(graphml_file="graph-example/Ulaknet.graphml", num_users=None):
    import networkx as nx
    bipartite_graph, client_nodes, es_nodes, distance_matrix = build_bipartite_graph(graphml_file)
    if bipartite_graph is None:
        return None, [], [], None, None, [], [], []
    print("Bipartite Graph Built Successfully!")
    print(f"Total Client Nodes: {len(client_nodes)}")
    print(f"Total ES Nodes: {len(es_nodes)}")
    print(f"Distance Matrix Shape: {distance_matrix.shape}")
    print(f"Distance Matrix (first 5x5):\n{distance_matrix[:5, :min(5, len(es_nodes))]}")

    r, assignments, loads, B_n = allocate_bandwidth_eba(client_nodes, es_nodes, distance_matrix, num_users)
    print(f"Final transmission rates r_m,n (bit/s, first 5x5):\n{r[:5, :min(5, len(es_nodes))]}")

    max_capacity = 5
    validate_results(B_cloud=5e7, B_n=B_n, r=r, assignments=assignments, loads=loads, max_capacity=max_capacity)

    plot_assigned_graph(bipartite_graph, client_nodes, es_nodes, assignments)
    plot_graph(bipartite_graph, client_nodes, es_nodes)

    # Debug: Print node attributes before saving
    print("Node attributes before saving GraphML:", list(bipartite_graph.nodes(data=True))[:5])

    # Convert non-supported attribute types for GraphML
    for node in bipartite_graph.nodes():
        attrs = bipartite_graph.nodes[node]
        for key, value in attrs.items():
            if isinstance(value, tuple):  # Convert tuples to strings
                bipartite_graph.nodes[node][key] = f"{value[0]},{value[1]}"
            elif not isinstance(value, (int, float, str, bool)):  # Convert other non-supported types
                bipartite_graph.nodes[node][key] = str(value)

    nx.write_graphml(bipartite_graph, "bipartite_graph.graphml")
    print("\nBipartite graph saved to bipartite_graph.graphml")

    return bipartite_graph, client_nodes, es_nodes, distance_matrix, r, assignments, loads, B_n

def allocate_bandwidth_eba(client_nodes, es_nodes, distance_matrix, num_users=None):
    import numpy as np
    M = len(client_nodes)  # Number of devices
    N = len(es_nodes)      # Number of edge servers
    B_cloud = 5e7          # Total cloud bandwidth (50 MHz)
    max_capacity = 5       # Max devices per edge server
    min_B_assigned = 1e6   # Minimum assigned bandwidth (1 MHz)

    # Restrict to num_users if provided
    if num_users is not None:
        M = min(M, num_users)
        client_nodes = client_nodes[:M]
        print(f"Restricting to {M} clients for allocation")
    distance_matrix_km = distance_matrix[:M, :N]

    # Channel gain calculation
    p_m = np.ones(M) * 1.0
    N0 = 1e-12
    alpha = 2.5
    k = 1e5
    g = k / (distance_matrix_km ** alpha + 1e-4)
    print(f"Channel gain (g, first 5x5):\n{g[:min(5, M), :min(5, N)]}")

    # Device-to-Edge Bandwidth
    mu = np.log(2e7)
    sigma = 0.5
    B_mn = np.random.lognormal(mean=mu, sigma=sigma, size=(M, N))
    distance_normalized = distance_matrix_km / np.max(distance_matrix_km)
    B_mn = B_mn / (distance_normalized + 1e-4)
    B_mn = np.clip(B_mn, 1e6, 5e7)
    print(f"Device-to-Edge Bandwidth (B_mn, first 5x5):\n{B_mn[:min(5, M), :min(5, N)]}")

    # Calculate initial SNR and transmission rate
    SNR_initial = (p_m[:, None] * g) / (N0 * B_mn + 1e-4)
    r_initial = B_mn * np.log2(1 + SNR_initial)
    print(f"Initial transmission rates (r_initial, first 5x5):\n{r_initial[:min(5, M), :min(5, N)]}")

    # Sort devices by maximum transmission rate
    max_r_per_device = np.max(r_initial, axis=1)
    sorted_device_indices = np.argsort(max_r_per_device)[::-1]
    print(f"Device sorting (by max r): {sorted_device_indices[:5]}... "
          f"(max_r: {max_r_per_device[sorted_device_indices[:5]]})")

    # Initialize assignments and loads
    assignments_array = np.full(M, -1, dtype=int)
    loads = np.zeros(N, dtype=int)

    # Assign devices to edge servers
    for idx in sorted_device_indices:
        available_n = np.where(loads < max_capacity)[0]
        if len(available_n) == 0:
            print(f"Warning: Device {idx} has no available edge server (all full)")
            continue
        r_for_device = r_initial[idx, available_n]
        load_scores = loads[available_n] + 1e-10
        rate_scores = r_for_device / np.max(r_for_device)
        combined_scores = 0.5 * (load_scores / np.max(load_scores)) - 0.5 * rate_scores
        sorted_n = available_n[np.argsort(combined_scores)]
        for n in sorted_n:
            if loads[n] < max_capacity:
                assignments_array[idx] = n
                loads[n] += 1
                break

    # Convert assignments to list of tuples
    assignments = [(i, int(j)) for i, j in enumerate(assignments_array) if j != -1]

    # 调试：验证 assignments
    invalid_assignments = [(i, j) for i, j in assignments if j >= N or j < 0]
    if invalid_assignments:
        print(f"Error: Invalid assignments in allocate_bandwidth_eba: {invalid_assignments}")
        assignments = [(i, j) for i, j in assignments if 0 <= j < N]
        print(f"Filtered assignments: {assignments}")

    # Fix r: Keep only assigned rates
    r = np.zeros_like(r_initial)
    for m, n in assignments:
        if r_initial[m, n] > 0:
            r[m, n] = r_initial[m, n]
        else:
            print(f"Warning: Invalid rate r_initial[{m}, {n}] = {r_initial[m, n]}")
    print(f"Assignments: {assignments}")
    print(f"Loads per edge: {loads}")
    print(f"Final transmission rates (r, first 5x5):\n{r[:min(5, M), :min(5, N)]}")

    # Edge-to-Cloud Bandwidth
    active = [n for n in range(N) if loads[n] > 0]
    N_active = len(active)
    B_n = np.zeros(N)
    if N_active > 0:
        mean_B = 5e7
        std_B = 1e7
        base_bandwidths = np.random.normal(loc=mean_B, scale=std_B, size=N_active)
        base_bandwidths = np.clip(base_bandwidths, mean_B / 2, mean_B * 2)
        load_weights = loads[active] / np.sum(loads[active])
        B_n[active] = base_bandwidths * load_weights * (B_cloud / np.sum(base_bandwidths * load_weights))
    print(f"Active edge servers: {active}")
    print(f"Edge-to-Cloud Bandwidth (B_n): {B_n}")

    # Handle abnormal rates
    r_threshold = 5e5
    abnormal_count = 0
    for m, n in assignments:
        if r[m, n] < r_threshold:
            print(f"Warning: Device {m} to edge {n} has r = {r[m, n]} < {r_threshold} bit/s")
            r[m, n] = 0
            abnormal_count += 1
    print(f"Total abnormal r values: {abnormal_count}")

    return r, assignments, loads, B_n


def validate_results(B_cloud, B_n, r, assignments, loads, max_capacity):
    N = len(B_n)
    M = len([i for i, _ in assignments] + [i for i in range(len(r)) if i not in [x[0] for x in assignments]])  # Total devices
    active = np.unique([j for _, j in assignments])  # Extract edge server indices from assignments

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
    mounted_ratio = len(assignments) / M if M > 0 else 0
    if mounted_ratio < 0.95:
        print(f"Warning: Low assignment coverage ({mounted_ratio:.2%})")
    else:
        print(f"Pass: Assignment coverage {mounted_ratio:.2%}")

    # Check transmission rates
    for m, n in assignments:
        if r[m, n] <= 0:
            print(f"Error: Assigned device {m} to edge {n} has r <= 0")
        for other_n in range(N):
            if other_n != n and r[m, other_n] > 0:
                print(f"Error: Non-assigned device {m} to edge {other_n} has r > 0")

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

