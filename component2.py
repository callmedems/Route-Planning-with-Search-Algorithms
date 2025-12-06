import osmnx as ox
import networkx as nx
import numpy as np
import time
import geopy.distance
from simpleai.search import SearchProblem, breadth_first, depth_first, uniform_cost, iterative_limited_depth_first, astar

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

class RoutePlanningProblem(SearchProblem):
    def __init__(self, G, start, goal):
        self.G = G
        self.goal_node = goal
        super().__init__(initial_state=start)
        self.nodes_explored = 0
    
    def actions(self, state):
        return list(self.G.successors(state))
    
    def result(self, state, action):
        self.nodes_explored += 1
        return action
    
    def is_goal(self, state):
        return state == self.goal_node
    
    def cost(self, state, action, state2):
        edge_data = self.G.get_edge_data(state, state2)
        if edge_data:
            return edge_data[0].get('length', 1)
        return 1
    
    def heuristic(self, state):
        coord1 = (self.G.nodes[state]['y'], self.G.nodes[state]['x'])
        coord2 = (self.G.nodes[self.goal_node]['y'], self.G.nodes[self.goal_node]['x'])
        return geopy.distance.distance(coord1, coord2).meters

def run_bfs(G, start, goal):
    problem = RoutePlanningProblem(G, start, goal)
    result = breadth_first(problem, graph_search=True)
    if result:
        path = [node for node, action in result.path()]
        return path, problem.nodes_explored
    return None, problem.nodes_explored

def run_dfs(G, start, goal):
    problem = RoutePlanningProblem(G, start, goal)
    result = depth_first(problem, graph_search=True)
    if result:
        path = [node for node, action in result.path()]
        return path, problem.nodes_explored
    return None, problem.nodes_explored

def run_ucs(G, start, goal):
    problem = RoutePlanningProblem(G, start, goal)
    result = uniform_cost(problem, graph_search=True)
    if result:
        path = [node for node, action in result.path()]
        return path, problem.nodes_explored
    return None, problem.nodes_explored

def run_iddfs(G, start, goal):
    problem = RoutePlanningProblem(G, start, goal)
    result = iterative_limited_depth_first(problem, graph_search=True)
    if result:
        path = [node for node, action in result.path()]
        return path, problem.nodes_explored
    return None, problem.nodes_explored

def run_astar(G, start, goal):
    problem = RoutePlanningProblem(G, start, goal)
    result = astar(problem, graph_search=True)
    if result:
        path = [node for node, action in result.path()]
        return path, problem.nodes_explored
    return None, problem.nodes_explored

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
        'BFS': run_bfs,
        'DFS': run_dfs,
        'UCS': run_ucs,
        'IDDFS': run_iddfs,
        'A*': run_astar
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
    short_results = run_experiment(G, short_pairs, "SHORT DISTANCES (< 1000m)")
    analyze_results(short_results, "SHORT")
    
    medium_pairs = select_node_pairs(G, 1000, 5000, num_pairs=5)
    medium_results = run_experiment(G, medium_pairs, "MEDIUM DISTANCES (1000-5000m)")
    analyze_results(medium_results, "MEDIUM")
    
    long_pairs = select_node_pairs(G, 5000, 15000, num_pairs=5)
    long_results = run_experiment(G, long_pairs, "LONG DISTANCES (> 5000m)")
    analyze_results(long_results, "LONG")

    
    return G

if __name__ == "__main__":
    np.random.seed(42)
    G = main()
