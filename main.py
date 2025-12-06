import osmnx as ox
import numpy as np
from component1 import (
    download_map_data as download_map_c1,
    extract_node_data,
    build_kdtree,
    generate_test_locations,
    search_with_kdtree,
    search_brute_force,
    compare_results
)
from component2 import (
    download_map_data as download_map_c2,
    get_predefined_locations,
    find_nearest_node,
    run_bfs,
    run_dfs,
    run_ucs,
    run_iddfs,
    run_astar,
    get_path_length
)
import time

LOCATION = 'Puerto Vallarta, Jalisco, México'
DISTANCE = 5000
NETWORK_TYPE = 'drive'

class RouteMapApp:
    def __init__(self):
        self.G = None
        self.kdtree = None
        self.node_ids = None
        self.node_coords = None
        self.build_time = None
        self.location_nodes = {}
        
    def initialize(self):
        """Initialize the application by downloading map data once"""
        print("\n" + "="*70)
        print("INITIALIZING ROUTE PLANNING APPLICATION")
        print("="*70)
        print("We will use a Puerto Vallarta map.")
        
        self.G = ox.graph_from_address(LOCATION, dist=DISTANCE, network_type=NETWORK_TYPE)
        self.G = ox.add_edge_speeds(self.G)
        self.G = ox.add_edge_travel_times(self.G)
        
        print(f"- Map loaded: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        
        # Build KD-tree for component 1
        print("Building KD-tree index for component 1...")
        self.node_ids = list(self.G.nodes)
        self.node_coords = np.array([[self.G.nodes[node]['y'], self.G.nodes[node]['x']] 
                                     for node in self.node_ids])
        
        points = [tuple(coord) for coord in self.node_coords]
        start_time = time.time()
        from kd_tree import KDTree
        self.kdtree = KDTree(points)
        self.build_time = time.time() - start_time
        
        print(f"- KD-tree built in {self.build_time:.6f} seconds")
        
        # Map predefined locations to nodes for component 2
        print("Mapping predefined locations to graph nodes...")
        locations = get_predefined_locations()
        for lat, lon, name in locations:
            nearest = find_nearest_node(self.G, lat, lon)
            self.location_nodes[name] = (nearest, lat, lon)
        
        print(f"- Mapped {len(self.location_nodes)} locations")

    def component1_demo(self):
        """Run Component 1: Optimized Vertex Search"""
        print("\n" + "="*70)
        print("COMPONENT 1: OPTIMIZED VERTEX SEARCH")
        print("="*70)
        
        test_locations = generate_test_locations()
        
        print("\nWe are currently running a KD-tree search.")
        kdtree_results, kdtree_times = search_with_kdtree(
            self.kdtree, self.node_ids, self.node_coords, test_locations
        )
        
        print("\nNow we are running a brute-force search.")
        bruteforce_results, bruteforce_times = search_brute_force(
            self.node_ids, self.node_coords, test_locations
        )
        
        compare_results(kdtree_results, kdtree_times, bruteforce_results, bruteforce_times, self.build_time)
    
    def component2_demo(self):
        """Run Component 2: Route Planning Algorithm Comparison"""
        print("\n" + "="*70)
        print("COMPONENT 2: ROUTE PLANNING - ALGORITHM COMPARISON")
        print("="*70)
        
        from component2 import select_node_pairs, run_experiment, analyze_results
        
        # Get properly selected pairs based on actual distances
        short_pairs = select_node_pairs(self.G, 0, 1000, num_pairs=5)
        short_results = run_experiment(self.G, short_pairs, "SHORT DISTANCES (< 1000m)")
        analyze_results(short_results, "SHORT")
        
        medium_pairs = select_node_pairs(self.G, 1000, 5000, num_pairs=5)
        medium_results = run_experiment(self.G, medium_pairs, "MEDIUM DISTANCES (1000-5000m)")
        analyze_results(medium_results, "MEDIUM")
        
        long_pairs = select_node_pairs(self.G, 5000, 15000, num_pairs=5)
        long_results = run_experiment(self.G, long_pairs, "LONG DISTANCES (> 5000m)")
        analyze_results(long_results, "LONG")
    
    def search_nearest_vertex(self, lat, lon):
        """Search for nearest vertex to a given coordinate"""
        print(f"\nSearching nearest vertex to ({lat:.6f}, {lon:.6f})...")
        
        query_point = np.array([lat, lon])
        
        # KD-tree search
        start_time = time.time()
        nearest_point, distance = self.kdtree.nearest_neighbor(tuple(query_point))
        kd_time = time.time() - start_time
        
        # Find node ID
        nearest_node = None
        for i, coord in enumerate(self.node_coords):
            if abs(coord[0] - nearest_point[0]) < 1e-9 and abs(coord[1] - nearest_point[1]) < 1e-9:
                nearest_node = self.node_ids[i]
                break
        
        print(f"✓ Found node {nearest_node}")
        print(f"  Distance: {distance:.6f} degrees")
        print(f"  Search time: {kd_time:.9f}s")
        
        return nearest_node
    
    def plan_route(self, loc1_name, loc2_name, algorithm='A*'):
        """Plan a route between two predefined locations"""
        if loc1_name not in self.location_nodes or loc2_name not in self.location_nodes:
            print("Error: Invalid location names")
            return
        
        node1 = self.location_nodes[loc1_name][0]
        node2 = self.location_nodes[loc2_name][0]
        
        print(f"\nPlanning route: {loc1_name} -> {loc2_name}")
        print(f"Algorithm: {algorithm}")
        
        algorithms = {
            'BFS': run_bfs,
            'DFS': run_dfs,
            'UCS': run_ucs,
            'IDDFS': run_iddfs,
            'A*': run_astar
        }
        
        if algorithm not in algorithms:
            print(f"Error: Unknown algorithm '{algorithm}'")
            return
        
        start_time = time.time()
        path, nodes_explored = algorithms[algorithm](self.G, node1, node2)
        elapsed = time.time() - start_time
        
        if path:
            path_length = get_path_length(self.G, path)
            print(f"\n✓ Route found!")
            print(f"  Nodes in path: {len(path)}")
            print(f"  Path length: {path_length:.1f}m")
            print(f"  Nodes explored: {nodes_explored}")
            print(f"  Time: {elapsed:.6f}s")
            return path
        else:
            print("\n✗ No route found")
            return None

def show_menu():
    print("\n" + "="*70)
    print("ROUTE PLANNING APPLICATION - PUERTO VALLARTA")
    print("="*70)
    print("\n1. Run Component 1 (KD-tree Optimized Vertex Search)")
    print("2. Run Component 2 (Route Planning Algorithm Comparison)")
    print("3. Run Both Components")
    print("4. Exit")
    print("\nChoice: ", end="")

def main():
    # Initialize application (load map once)
    app = RouteMapApp()
    app.initialize()
    
    while True:
        show_menu()
        choice = input().strip()
        
        if choice == '1':
            app.component1_demo()
        
        elif choice == '2':
            app.component2_demo()
        
        elif choice == '3':
            app.component1_demo()
            app.component2_demo()
            print("\n" + "="*70)
            print("BOTH COMPONENTS COMPLETED")
            print("="*70)
        
        elif choice == '4':
            print("\n" + "="*70)
            print("Thank you for using Route Planning Application!")
            print("="*70)
            break
        
        else:
            print("\n✗ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
