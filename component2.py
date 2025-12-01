import osmnx as ox
import networkx as nx
import numpy as np
import time
from collections import deque
import heapq
import geopy.distance
import matplotlib.pyplot as plt

LOCATION = 'Puerto Vallarta, Jalisco, México'
DISTANCE = 5000
NETWORK_TYPE = 'drive'

def download_map_data():
    print("\n" + "="*70)
    print("DOWNLOADING MAP DATA")
    print("="*70)
    
    G = ox.graph_from_address(LOCATION, dist=DISTANCE, network_type=NETWORK_TYPE)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    
    print(f"Nodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")
    return G

def get_distance_between_nodes(G, node1, node2):
    coord1 = (G.nodes[node1]['y'], G.nodes[node1]['x'])
    coord2 = (G.nodes[node2]['y'], G.nodes[node2]['x'])
    return geopy.distance.distance(coord1, coord2).meters

def get_predefined_locations():
    return [
        (20.645221, -105.237188, "Location 1"),
        (20.642388, -105.221028, "Location 2"),
        (20.630880, -105.229097, "Location 3"),
        (20.627529, -105.225031, "Location 4"),
        (20.629460, -105.232496, "Location 5"),
        (20.648714, -105.232193, "Location 6"),
        (20.641103, -105.211800, "Location 7"),
        (20.638832, -105.202757, "Location 8"),
        (20.651781, -105.205792, "Location 9"),
        (20.617872, -105.232678, "Location 10"),
        (20.601398, -105.233589, "Location 11"),
        (20.590206, -105.243360, "Location 12"),
        (20.598160, -105.223696, "Location 13"),
        (20.610317, -105.224363, "Location 14"),
        (20.608897, -105.232375, "Location 15"),
        (20.606909, -105.224485, "Location 16"),
        (20.601909, -105.238080, "Location 17"),
        (20.609976, -105.232193, "Location 18"),
        (20.583900, -105.243299, "Location 19"),
        (20.600887, -105.228308, "Location 20")
    ]

def find_nearest_node(G, lat, lon):
    """Find the nearest graph node to a given coordinate"""
    min_dist = float('inf')
    nearest = None
    
    for node in G.nodes:
        node_lat = G.nodes[node]['y']
        node_lon = G.nodes[node]['x']
        dist = geopy.distance.distance((lat, lon), (node_lat, node_lon)).meters
        
        if dist < min_dist:
            min_dist = dist
            nearest = node
    
    return nearest

def select_node_pairs(G, min_dist, max_dist, num_pairs=5):
    print("\n" + "-"*70)
    print(f"\nSELECTING {num_pairs} NODE PAIRS BETWEEN {min_dist}M AND {max_dist}M.")
    
    # Get predefined locations of puerto vallarta
    locations = get_predefined_locations()
    
    # Map each location to its nearest graph node
    location_nodes = {}
    for lat, lon, name in locations:
        nearest = find_nearest_node(G, lat, lon)
        location_nodes[name] = (nearest, lat, lon)
    
    # Find pairs within distance range
    pairs = []
    location_names = list(location_nodes.keys())
    
    for i in range(len(location_names)):
        for j in range(i + 1, len(location_names)):
            name1 = location_names[i]
            name2 = location_names[j]
            
            node1 = location_nodes[name1][0]
            node2 = location_nodes[name2][0]
            lat1, lon1 = location_nodes[name1][1], location_nodes[name1][2]
            lat2, lon2 = location_nodes[name2][1], location_nodes[name2][2]
            
            # Calculate straight-line distance between original coordinates
            dist = geopy.distance.distance((lat1, lon1), (lat2, lon2)).meters
            
            if min_dist <= dist <= max_dist:
                if nx.has_path(G, node1, node2):
                    pairs.append((node1, node2, dist, name1, name2))
                    print(f"  SELECTED {name1} -- {name2}: {dist:.1f}m")
                    
                    if len(pairs) >= num_pairs:
                        return pairs
    
    if len(pairs) < num_pairs:
        print(f"  Warning: Only found {len(pairs)} valid pairs in this range")
    
    return pairs

def visualize_node_pairs(G, pairs, category_name):
    fig, ax = ox.plot_graph(G, node_size=0, edge_color='#CCCCCC', 
                            edge_linewidth=0.5, bgcolor='white', 
                            show=False, close=False)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, pair in enumerate(pairs):
        node1, node2, dist = pair[0], pair[1], pair[2]
        name1 = pair[3] if len(pair) > 3 else f"N{idx*2}"
        name2 = pair[4] if len(pair) > 4 else f"N{idx*2+1}"
        
        y1, x1 = G.nodes[node1]['y'], G.nodes[node1]['x']
        y2, x2 = G.nodes[node2]['y'], G.nodes[node2]['x']
        
        color = colors[idx % len(colors)]
        
        ax.scatter(x1, y1, c=color, s=200, marker='o', zorder=5, edgecolors='black', linewidths=2)
        ax.scatter(x2, y2, c=color, s=200, marker='s', zorder=5, edgecolors='black', linewidths=2)
        
        ax.text(x1, y1, f" {name1}", fontsize=8, ha='left', va='bottom', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.text(x2, y2, f" {name2}", fontsize=8, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.5, linestyle='--', zorder=3)
    
    plt.title(f'{category_name} - Node Pairs Visualization', fontsize=14, fontweight='bold')
    
    filename = f"node_pairs_{category_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nMap saved as '{filename}'")
    plt.close()

def bfs(G, start, goal):
    queue = deque([start])
    came_from = {start: None}
    nodes_explored = 0
    
    while queue:
        current = queue.popleft()
        nodes_explored += 1
        
        if current == goal:
            break
        
        for neighbor in G.successors(current):
            if neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current
    
    if goal not in came_from:
        return None, nodes_explored
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path, nodes_explored

def dfs(G, start, goal):
    stack = [start]
    came_from = {start: None}
    nodes_explored = 0
    
    while stack:
        current = stack.pop()
        nodes_explored += 1
        
        if current == goal:
            break
        
        for neighbor in G.successors(current):
            if neighbor not in came_from:
                stack.append(neighbor)
                came_from[neighbor] = current
    
    if goal not in came_from:
        return None, nodes_explored
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path, nodes_explored

def ucs(G, start, goal):
    pq = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    nodes_explored = 0
    
    while pq:
        current_cost, current = heapq.heappop(pq)
        nodes_explored += 1
        
        if current == goal:
            break
        
        for neighbor in G.successors(current):
            edge_data = G.get_edge_data(current, neighbor)
            if edge_data:
                edge_cost = edge_data[0].get('length', 1)
                new_cost = cost_so_far[current] + edge_cost
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))
                    came_from[neighbor] = current
    
    if goal not in came_from:
        return None, nodes_explored
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path, nodes_explored

def dfs_limited(G, start, goal, depth_limit):
    stack = [(start, 0)]
    came_from = {start: None}
    nodes_explored = 0
    
    while stack:
        current, depth = stack.pop()
        nodes_explored += 1
        
        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, nodes_explored
        
        if depth < depth_limit:
            for neighbor in G.successors(current):
                if neighbor not in came_from:
                    stack.append((neighbor, depth + 1))
                    came_from[neighbor] = current
    
    return None, nodes_explored

def iddfs(G, start, goal, max_depth=50):
    total_nodes_explored = 0
    
    for depth in range(max_depth):
        result, nodes = dfs_limited(G, start, goal, depth)
        total_nodes_explored += nodes
        
        if result is not None:
            return result, total_nodes_explored
    
    return None, total_nodes_explored

def heuristic(G, node, goal):
    coord1 = (G.nodes[node]['y'], G.nodes[node]['x'])
    coord2 = (G.nodes[goal]['y'], G.nodes[goal]['x'])
    return geopy.distance.distance(coord1, coord2).meters

def astar(G, start, goal):
    pq = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    nodes_explored = 0
    
    while pq:
        _, current = heapq.heappop(pq)
        nodes_explored += 1
        
        if current == goal:
            break
        
        for neighbor in G.successors(current):
            edge_data = G.get_edge_data(current, neighbor)
            if edge_data:
                edge_cost = edge_data[0].get('length', 1)
                new_cost = cost_so_far[current] + edge_cost
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(G, neighbor, goal)
                    heapq.heappush(pq, (priority, neighbor))
                    came_from[neighbor] = current
    
    if goal not in came_from:
        return None, nodes_explored
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path, nodes_explored

def get_path_length(G, path):
    if path is None or len(path) < 2:
        return 0
    
    total_length = 0
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i+1])
        if edge_data:
            total_length += edge_data[0].get('length', 0)
    
    return total_length

def run_experiment(G, pairs, category_name):
    print("\n" + "="*70)
    print(f"TESTING: {category_name}")
    print("="*70)
    
    algorithms = {
        'BFS': bfs,
        'DFS': dfs,
        'UCS': ucs,
        'IDDFS': iddfs,
        'A*': astar
    }
    
    results = {name: [] for name in algorithms}
    
    for idx, pair in enumerate(pairs, 1):
        start, goal, dist = pair[0], pair[1], pair[2]
        name1 = pair[3] if len(pair) > 3 else f"Node {start}"
        name2 = pair[4] if len(pair) > 4 else f"Node {goal}"
        
        print(f"\nPair {idx}: {name1} -> {name2} ({dist:.1f}m)")
        
        for name, algorithm in algorithms.items():
            try:
                start_time = time.time()
                path, nodes_explored = algorithm(G, start, goal)
                elapsed = time.time() - start_time
                
                if path:
                    path_length = get_path_length(G, path)
                    results[name].append({
                        'time': elapsed,
                        'nodes': nodes_explored,
                        'length': path_length,
                        'success': True
                    })
                    print(f"  {name:6s}: {elapsed:.6f}s, {nodes_explored:5d} nodes, {path_length:.1f}m")
                else:
                    results[name].append({'success': False})
                    print(f"  {name:6s}: No path found")
            except Exception as e:
                results[name].append({'success': False})
                print(f"  {name:6s}: Error - {e}")
    
    return results

def analyze_results(results, category_name):
    print(f"\n{category_name} - SUMMARY:")
    print("-" * 60)
    
    for name, data in results.items():
        successful = [d for d in data if d.get('success', False)]
        
        if successful:
            avg_time = np.mean([d['time'] for d in successful])
            avg_nodes = np.mean([d['nodes'] for d in successful])
            avg_length = np.mean([d['length'] for d in successful])
            
            print(f"{name:6s}: {avg_time:.6f}s | {avg_nodes:7.1f} nodes | {avg_length:8.1f}m")
        else:
            print(f"{name:6s}: Failed all tests")

def main():
    print("\nCOMPONENT 2: ROUTE PLANNER - ALGORITHM COMPARISON")
    print("="*70)
    
    G = download_map_data()
    
    short_pairs = select_node_pairs(G, 0, 1000, num_pairs=5)
    visualize_node_pairs(G, short_pairs, "SHORT DISTANCES")
    short_results = run_experiment(G, short_pairs, "SHORT DISTANCES (< 1000m)")
    analyze_results(short_results, "SHORT")
    
    medium_pairs = select_node_pairs(G, 1000, 5000, num_pairs=5)
    visualize_node_pairs(G, medium_pairs, "MEDIUM DISTANCES")
    medium_results = run_experiment(G, medium_pairs, "MEDIUM DISTANCES (1000-5000m)")
    analyze_results(medium_results, "MEDIUM")
    
    long_pairs = select_node_pairs(G, 5000, 15000, num_pairs=5)
    visualize_node_pairs(G, long_pairs, "LONG DISTANCES")
    long_results = run_experiment(G, long_pairs, "LONG DISTANCES (> 5000m)")
    analyze_results(long_results, "LONG")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("A* is recommended because:")
    print("• Finds optimal paths")
    print("• Faster than UCS")
    print("• Explores fewer nodes")
    print("• Scales well")
    
    return G

if __name__ == "__main__":
    np.random.seed(42)
    G = main()
