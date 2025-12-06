import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon
import geopandas as gpd
import heapq
import time

LOCATION = 'Puerto Vallarta, Jalisco, Mexico'
DISTANCE = 5000
NETWORK_TYPE = 'drive'


HOSPITAL_COORDS = [
    (20.652430638239863, -105.21505296205477),
    (20.646718491792683, -105.22054975163908),
    (20.648633026847065, -105.22599629558445),
    (20.647108467321445, -105.22018621037452),
    (20.64391395744467, -105.22055861849702),
    (20.639394672578145, -105.2210352250464),
    (20.636632287628046, -105.22758722074346),
    (20.63446354283605, -105.22816657788519),
    (20.638211134502157, -105.23207139339135),
    (20.640078620584912, -105.23209285106327),
    (20.652270395787763, -105.24043737515635),
    (20.62606320543664, -105.2298363630843),
    (20.669772538638313, -105.21033146168664),
    (20.67304199823121, -105.24944250204376)

]

HOSPITAL_NAMES = [
    'Sanatorio Belén',
    'ISSTE Clínica Hospital',
    'Hospital Puerta Del Mar',
    'Hospital Premiere CMQ',
    'Clínica Ángeles',
    'Healthcare by the sea',
    'Hospital Multimédica',
    'Hospital Versalles',
    'IMSS Zona 42',
    'Medical Center',
    'Hospital Joya Marina',
    'Servicio Médico de La Bahía',
    'Hospital Regional de Puerto Vallarta',
    'Hospiten Puerto Vallarta'
]


def generate_voro_points(vor, radius=None):

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]

        if -1 not in region:
            new_regions.append(region)
            continue

        new_region = [v for v in region if v != -1]

        for p2, v1, v2 in all_ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1

            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = (vor.points[p1] + vor.points[p2]) / 2
            direction = 1 if np.dot(midpoint - center, n) > 0 else -1

            far_point = midpoint + direction * radius * n
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        centroid = vs.mean(axis=0)
        new_region = sorted(
            new_region,
            key=lambda v: np.arctan2(new_vertices[v][1] - centroid[1],
                                     new_vertices[v][0] - centroid[0])
        )
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)
class RoutePlanner:
    def __init__(self):
        print("Loading map data...")
    
        self.G = ox.graph_from_address(LOCATION, dist=DISTANCE, network_type=NETWORK_TYPE)
        self.G = ox.add_edge_speeds(self.G)
        self.G = ox.add_edge_travel_times(self.G)
        
        # Grafo proyectado en metros (para Voronoi y A*)
        self.G_proj = ox.project_graph(self.G)
        self.hospital_coords = HOSPITAL_COORDS 
        self.hospital_nodes = []               
        self.voronoi_regions = []
        self.hospital_names = HOSPITAL_NAMES

        self.set_nodes()

    def set_nodes(self):
        print("\nFinding nearest nodes for hospitals: ")
        
        for i, (lat, lon) in enumerate(self.hospital_coords):
            nearest_node = ox.distance.nearest_nodes(self.G, lon, lat)
            self.hospital_nodes.append(nearest_node)
            node_data = self.G.nodes[nearest_node]
            print(f" {[i+1]} {self.hospital_names[i]}: ({lat:.6f}, {lon:.6f}) -> Node {nearest_node} "
                  f"({node_data['y']:.6f}, {node_data['x']:.6f})")
            
        print(f"\nMapped {len(self.hospital_nodes)} hospitals to graph nodes")
    
    #Voronoi in CRS 
    def create_voronoi_partition(self):

        print("\nCreating Voronoi partition area coverage")
        
        points = np.array([
            [self.G_proj.nodes[node]['x'], self.G_proj.nodes[node]['y']]
            for node in self.hospital_nodes
        ])
        
        # Voronoi 
        vor = Voronoi(points)
        regions, vertices = generate_voro_points(vor)
        
        # Bounding box 
        nodes_gdf = ox.graph_to_gdfs(self.G_proj, edges=False)
        minx, miny, maxx, maxy = nodes_gdf.total_bounds
        map_bounds = Polygon([(minx, miny), (maxx, miny),
                              (maxx, maxy), (minx, maxy)])
        
        # Build voronoi regions
        self.voronoi_regions = []
        for region in regions:
            polygon = Polygon(vertices[region])
            bounded_polygon = polygon.intersection(map_bounds)
            if bounded_polygon.is_empty:
                bounded_polygon = polygon  
            self.voronoi_regions.append(bounded_polygon)
        
        print(f"Created {len(self.voronoi_regions)} Voronoi regions")
        return self.voronoi_regions
    
    
    #Región Voronoi para un nodo
    def setRegionNode(self, node_id):
        x = self.G_proj.nodes[node_id]['x']
        y = self.G_proj.nodes[node_id]['y']
        p = Point(x, y)
        
        for i, region in enumerate(self.voronoi_regions):
            if region.contains(p):
                return i
        
        # Si no está en ninguna región (punto muy cercano a límite),
        # usar distancia a los centros Voronoi
        min_dist = float('inf')
        nearest_idx = 0
        for i, h_node in enumerate(self.hospital_nodes):
            hx = self.G_proj.nodes[h_node]['x']
            hy = self.G_proj.nodes[h_node]['y']
            dist = np.hypot(hx - x, hy - y)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx
    def visualize_voronoi(self, figsize=(15, 12)):
        fig, ax = plt.subplots(figsize=figsize)

        # Road network
        ox.plot_graph(self.G_proj, ax=ax, node_size=0, edge_color="lightgray",
                      edge_linewidth=0.5, show=False, close=False)

        # Generate unique colors for each voronoi region
        num_regions = len(self.voronoi_regions)
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_regions))
        
        for i, poly in enumerate(self.voronoi_regions):
            if poly.geom_type == "Polygon":
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.25, color=colors[i], label=f'{self.hospital_names[i]}')
            elif poly.geom_type == "MultiPolygon":
                for sub_poly in poly.geoms:
                    x, y = sub_poly.exterior.xy
                    ax.fill(x, y, alpha=0.25, color=colors[i])

        # Hospitales
        for n in self.hospital_nodes:
            ax.scatter(self.G_proj.nodes[n]["x"],
                       self.G_proj.nodes[n]["y"],
                       s=100, c="red", marker="+")

        plt.title("Hospital area of influence using voronoi")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        return fig, ax
    
    def astar(self, start, goal):

        def heuristic(node, goal_node):
            x1 = self.G_proj.nodes[node]['x']
            y1 = self.G_proj.nodes[node]['y']
            x2 = self.G_proj.nodes[goal_node]['x']
            y2 = self.G_proj.nodes[goal_node]['y']
            return np.hypot(x1 - x2, y1 - y2)
        
        pq = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        nodes_explored = 0
        
        while pq:
            _, current = heapq.heappop(pq)
            nodes_explored += 1
            
            if current == goal:
                break
            
            for neighbor in self.G_proj.successors(current):
                edge_data = self.G_proj.get_edge_data(current, neighbor)
                if edge_data:
                    edge_cost = edge_data[0].get('length', 1)
                    new_cost = cost_so_far[current] + edge_cost
                    
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + heuristic(neighbor, goal)
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
    
    def find_route(self, query_lat, query_lon):

        print("\n" + "="*70)
        print(f"COMPUTING ROUTE FROM ({query_lat:.6f}, {query_lon:.6f})")
        print("="*70)
        
        # Step 1: Find nearest node to query
        query_node = ox.distance.nearest_nodes(self.G, query_lon, query_lat)
        query_node_lat = self.G.nodes[query_node]['y']
        query_node_lon = self.G.nodes[query_node]['x']
        

        print(f"\n  Query coordinates: ({query_lat:.6f}, {query_lon:.6f})")
        print(f"  Nearest node: {query_node} at ({query_node_lat:.6f}, {query_node_lon:.6f})")
        
        # Step 2: Determine hospital using Voronoi
        
        hospital_idx = self.setRegionNode(query_node)
        hospital_node = self.hospital_nodes[hospital_idx]
        
        hospital_lat = self.hospital_coords[hospital_idx][0]
        hospital_lon = self.hospital_coords[hospital_idx][1]
        hospital_name = self.hospital_names[hospital_idx]
        print(f"\n Assigned to HOSPITAL: {hospital_name}")
        print(f" Hospital location: ({hospital_lat:.6f}, {hospital_lon:.6f})")

        
        # Step 3: Calculate route with A*
        print(f"\n CALCULATING ROUTE ...")
        start_time = time.time()
        route, nodes_explored = self.astar(query_node, hospital_node)
        astar_time = time.time() - start_time
        
        if route is None:
            print("\nNo path found!")
            return None, hospital_idx, None, query_node
        
        # Calculate route length
        route_length = 0
        for i in range(len(route) - 1):
            edge_data = self.G_proj.get_edge_data(route[i], route[i+1])
            if edge_data:
                route_length += edge_data[0].get('length', 0)
                
        print(f" Route distance: {route_length:.2f} meters ({route_length/1000:.2f} km)")
        print(f" Route found in: {astar_time:.6f} seconds")
        print("="*70)
        
        return route, hospital_idx, route_length, query_node

    
    def visualize_route(self, route, query_lat, query_lon, hospital_idx, query_node, 
                       save_path='hospital_route.png'):
        
        print(f"\nGenerating visualization...")
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # 1. Plot street network
        ox.plot_graph(self.G, ax=ax, node_size=0, edge_color='lightgray',
                      edge_linewidth=0.5, show=False, close=False)
        
        # 2. Plot Voronoi regions (convert to WGS84)
        gdf_vor = gpd.GeoDataFrame(geometry=self.voronoi_regions, 
                                    crs=self.G_proj.graph['crs'])
        gdf_vor = gdf_vor.to_crs(self.G.graph['crs'])
        
        num_regions = len(self.voronoi_regions)
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_regions))
        
        for i, region in enumerate(gdf_vor.geometry):
            alpha = 0.35 if i == hospital_idx else 0.15  # Highlight selected region
            if region.geom_type == "Polygon":
                x, y = region.exterior.xy
                ax.fill(x, y, alpha=alpha, color=colors[i], edgecolor='gray', linewidth=0.5)
            elif region.geom_type == "MultiPolygon":
                for poly in region.geoms:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, alpha=alpha, color=colors[i], edgecolor='gray', linewidth=0.5)
        
        # 3. Plot the route
        if route:
            ox.plot_graph_route(self.G, route, ax=ax, route_color='blue',
                                route_linewidth=4, node_size=0, show=False, close=False)
        
        # 4. Plot all hospitals
        for i, node in enumerate(self.hospital_nodes):
            h_x = self.G.nodes[node]['x']
            h_y = self.G.nodes[node]['y']
            
            if i == hospital_idx:
                # Highlight destination hospital
                ax.scatter([h_x], [h_y], c='red', s=250, marker='*',
                          edgecolors='darkred', linewidths=3, zorder=10,
                          label=f'DESTINATION: {self.hospital_names[i]}')
            else:
                # Other hospitals
                ax.scatter([h_x], [h_y], c='lightcoral', s=150, marker='+',
                          linewidths=2, zorder=8)
            
            # Label each hospital
            ax.text(h_x, h_y, f'  H{i+1}', fontsize=9, fontweight='bold',
                   color='darkred' if i == hospital_idx else 'gray',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='yellow' if i == hospital_idx else 'white',
                            alpha=0.8), zorder=9)
        
        # 5. Plot query location (at the route start node)
        if route:
            start_x = self.G.nodes[query_node]['x']
            start_y = self.G.nodes[query_node]['y']
            ax.scatter([start_x], [start_y], c='blue', s=100,
                      marker='o', edgecolors='blue', linewidths=3,
                      label='QUERY LOCATION', zorder=11)
        
        # Title and legend
        ax.set_title(f'Route to Hospital {self.hospital_names[i]}\n'
                    f'From ({query_lat:.5f}, {query_lon:.5f})',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        plt.tight_layout()
        
        # Save
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
        plt.show()
        
        return fig, ax


def main():

    planner = RoutePlanner()
    planner.create_voronoi_partition()
    planner.create_voronoi_partition()
    planner.visualize_voronoi(figsize=(16,14))
    plt.savefig('voronoi_partition.png', dpi=150, bbox_inches='tight')
    plt.close()
    

    #Start program
    print("\n\n" + "="*70)
    print("Find the nearest hospital from your location")
    print("="*70)
    print("\nEnter your own coordinates! (Format: latitude, longitude)")
    print("Example: 20.647, -105.220")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("\nEnter coordinates (lat, lon) or 'quit': ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n Exiting program. Goodbye!")
                break
            
            # Parse coordinates
            parts = user_input.split(',')
            if len(parts) != 2:
                print("Error: Please enter exactly 2 values (latitude, longitude)")
                continue
            
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            
            # Validate coordinates (Puerto Vallarta approximate bounds)
            if not (20.55 <= lat <= 20.75 and -105.30 <= lon <= -105.15):
                print("Warning: Coordinates outside Puerto Vallarta area")
                print("Continuing anyway...")
            
            # Find route
            route, hospital_idx, route_length, query_node = planner.find_route(lat, lon)
            planner.visualize_route(route, lat, lon, hospital_idx, query_node, save_path='route.png')
       
        except Exception as e:
            print(f"Something went wrong {e}")


if __name__ == "__main__":
    main()