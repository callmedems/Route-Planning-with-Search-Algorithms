import osmnx as ox
import numpy as np
import time
from kd_tree import KDTree

LOCATION = 'Puerto Vallarta, Jalisco, México'
DISTANCE = 5000
NETWORK_TYPE = 'drive'

def download_map_data():
    print("\n" + "="*70)
    print("DOWNLOADING MAP DATA")
    print("="*70)
    
    G = ox.graph_from_address(LOCATION, dist=DISTANCE, network_type=NETWORK_TYPE)
    
    print(f"Nodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")
    return G

def extract_node_data(G):
    print("\n" + "="*70)
    print("COORDINATE TRANSFORMATION")
    print("="*70)
    
    node_ids = list(G.nodes)
    node_coords = np.array([[G.nodes[node]['y'], G.nodes[node]['x']] 
                            for node in node_ids])
    
    print(f"Extracted {len(node_coords)} node coordinates")
    return node_ids, node_coords

def build_kdtree(node_coords):
    print("\n" + "="*70)
    print("BUILDING KD-TREE")
    print("="*70)
    
    points = [tuple(coord) for coord in node_coords]
    
    start_time = time.time()
    kdtree = KDTree(points)
    build_time = time.time() - start_time
    
    print(f"Build time: {build_time:.6f} seconds")
    return kdtree, build_time

def generate_test_locations():
    print("\n" + "="*70)
    print("TEST LOCATIONS")
    print("="*70)
    
    test_locations = [
        (20.645221, -105.237188),
        (20.642388, -105.221028),
        (20.630880, -105.229097),
        (20.627529, -105.225031),
        (20.629460, -105.232496),
        (20.648714, -105.232193),
        (20.641103, -105.211800),
        (20.638832, -105.202757),
        (20.651781, -105.205792),
        (20.617872, -105.232678),
        (20.601398, -105.233589),
        (20.590206, -105.243360),
        (20.598160, -105.223696),
        (20.610317, -105.224363),
        (20.608897, -105.232375),
        (20.606909, -105.224485),
        (20.601909, -105.238080),
        (20.609976, -105.232193),
        (20.583900, -105.243299),
        (20.600887, -105.228308)
    ]
    
    print(f"Loaded {len(test_locations)} coordinates")
    return test_locations

def nearest_neighbor_kdtree(kdtree, node_ids, node_coords, query_point):
    nearest_point, distance = kdtree.nearest_neighbor(tuple(query_point))
    
    for i, coord in enumerate(node_coords):
        if abs(coord[0] - nearest_point[0]) < 1e-9 and abs(coord[1] - nearest_point[1]) < 1e-9:
            return node_ids[i], distance
    
    return None, distance

def search_with_kdtree(kdtree, node_ids, node_coords, test_locations):
    print("\n" + "="*70)
    print("SEARCHING WITH KD-TREE")
    print("="*70)
    
    results = []
    search_times = []
    
    for i, (lat, lon) in enumerate(test_locations, 1):
        query_point = np.array([lat, lon])
        
        start_time = time.time()
        nearest_node, distance = nearest_neighbor_kdtree(kdtree, node_ids, node_coords, query_point)
        search_time = time.time() - start_time
        
        results.append((nearest_node, distance))
        search_times.append(search_time)
        
        print(f"Location {i:2d}: ({lat:.6f}, {lon:.6f}) → Node {nearest_node}, Time: {search_time:.9f}s")
    
    print(f"\nAverage time: {np.mean(search_times):.9f} seconds")
    return results, search_times

def search_brute_force(node_ids, node_coords, test_locations):
    print("\n" + "="*70)
    print("SEARCHING WITH BRUTE-FORCE")
    print("="*70)
    
    results = []
    search_times = []
    
    for i, (lat, lon) in enumerate(test_locations, 1):
        query_point = np.array([lat, lon])
        
        start_time = time.time()
        
        min_distance = float('inf')
        nearest_node = None
        
        for j, node_id in enumerate(node_ids):
            node_coord = node_coords[j]
            distance = np.linalg.norm(query_point - node_coord)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        search_time = time.time() - start_time
        
        results.append((nearest_node, min_distance))
        search_times.append(search_time)
        
        print(f"Location {i:2d}: ({lat:.6f}, {lon:.6f}) -> Node {nearest_node}, Time: {search_time:.9f}s")
    
    print(f"\nAverage time: {np.mean(search_times):.9f} seconds")
    return results, search_times

def compare_results(kdtree_results, kdtree_times, bruteforce_results, bruteforce_times, build_time):
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    matches = sum(1 for i in range(len(kdtree_results)) 
                  if kdtree_results[i][0] == bruteforce_results[i][0])
    
    print(f"\nMatching results: {matches}/{len(kdtree_results)}")
    
    print(f"\nKD-tree build time: {build_time:.6f} seconds")
    print(f"KD-tree avg search: {np.mean(kdtree_times):.9f} seconds")
    print(f"Brute-force avg search: {np.mean(bruteforce_times):.9f} seconds")

def main():
    print("\nCOMPONENT 1: OPTIMIZED VERTEX SEARCH")
    print("="*70)
    
    G = download_map_data()
    node_ids, node_coords = extract_node_data(G)
    kdtree, build_time = build_kdtree(node_coords)
    test_locations = generate_test_locations()
    
    kdtree_results, kdtree_times = search_with_kdtree(kdtree, node_ids, node_coords, test_locations)
    bruteforce_results, bruteforce_times = search_brute_force(node_ids, node_coords, test_locations)
    
    compare_results(kdtree_results, kdtree_times, bruteforce_results, bruteforce_times, build_time)
    
    return G, kdtree, node_ids, node_coords

if __name__ == "__main__":
    G, kdtree, node_ids, node_coords = main()
