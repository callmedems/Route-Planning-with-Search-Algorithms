import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, Polygon
import geopandas as gpd

# Configuration
LOCATION = 'Puerto Vallarta, Jalisco, Mexico'
DISTANCE = 5000
NETWORK_TYPE = 'drive'

# Hospital locations (lat, lon)
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
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Convierte las regiones infinitas del Voronoi en polígonos finitos.
    Receta estándar de SciPy.
    """
    import numpy as np

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
class HospitalRoutePlanner:
    def __init__(self, location, distance, network_type):
        """Initialize the route planner with map data."""
        print("Loading map data...")
        # Grafo en WGS84
        self.G = ox.graph_from_place(location, network_type=network_type)
        # Grafo proyectado en metros (para Voronoi y A*)
        self.G_proj = ox.project_graph(self.G)

        print(f"Loaded graph with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
        
        self.hospital_coords = HOSPITAL_COORDS  # lista de (lat, lon)
        self.hospital_nodes = []               # nodos de la red
        self.voronoi_regions = []              # polígonos Voronoi (en CRS proyectado)
    
    # ---------------------------------------------------------
    # 1) Hospitales -> nodos
    # ---------------------------------------------------------
    def find_nearest_nodes(self):
        """Find the nearest graph node for each hospital."""
        print("\nFinding nearest nodes for hospitals...")
        self.hospital_nodes = []
        
        for i, (lat, lon) in enumerate(self.hospital_coords):
            nearest_node = ox.distance.nearest_nodes(self.G, lon, lat)
            self.hospital_nodes.append(nearest_node)
            node_data = self.G.nodes[nearest_node]
            print(f"Hospital {i+1}: ({lat:.6f}, {lon:.6f}) -> Node {nearest_node} "
                  f"({node_data['y']:.6f}, {node_data['x']:.6f})")
        
        return self.hospital_nodes
    
    # ---------------------------------------------------------
    # 2) Voronoi en CRS proyectado
    # ---------------------------------------------------------
    def create_voronoi_partition(self):
        """
        Generate Voronoi partition based on hospital locations.
        Voronoi se hace en el grafo proyectado (metros) y se recorta
        al bounding box del grafo.
        """
        print("\nCreating Voronoi partition (projected)...")
        
        if not self.hospital_nodes:
            raise RuntimeError("Call find_nearest_nodes() before create_voronoi_partition().")
        
        # Coordenadas (x, y) en CRS proyectado
        points = np.array([
            [self.G_proj.nodes[node]['x'], self.G_proj.nodes[node]['y']]
            for node in self.hospital_nodes
        ])
        
        # Voronoi crudo
        vor = Voronoi(points)
        
        # Regiones finitas
        regions, vertices = voronoi_finite_polygons_2d(vor)
        
        # Bounding box del grafo proyectado
        nodes_gdf = ox.graph_to_gdfs(self.G_proj, edges=False)
        minx, miny, maxx, maxy = nodes_gdf.total_bounds
        map_bounds = Polygon([(minx, miny), (maxx, miny),
                              (maxx, maxy), (minx, maxy)])
        
        # Construir regiones Voronoi recortadas
        self.voronoi_regions = []
        for region in regions:
            polygon = Polygon(vertices[region])
            bounded_polygon = polygon.intersection(map_bounds)
            if bounded_polygon.is_empty:
                bounded_polygon = polygon  # fallback mínimo
            self.voronoi_regions.append(bounded_polygon)
        
        print(f"Created {len(self.voronoi_regions)} Voronoi regions")
        return self.voronoi_regions
    
    # ---------------------------------------------------------
    # 3) Región Voronoi para un nodo
    # ---------------------------------------------------------
    def _get_region_for_node(self, node_id):
        """
        Given a node in the projected graph, return the index of the Voronoi
        region (and therefore hospital) it belongs to.
        """
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
        
        print(f"  Node {node_id} not in any region, using fallback (distance-based).")
        return nearest_idx
    
    # ---------------------------------------------------------
    # 4) Función pública: encontrar hospital usando Voronoi
    # ---------------------------------------------------------
    def find_nearest_hospital(self, query_lat, query_lon):
        """
        Dado (lat, lon), determina:
        - el nodo más cercano en la red
        - la región Voronoi a la que pertenece
        - el hospital (nodo) asociado a esa región
        """
        # Nodo más cercano en la red (en coordenadas geográficas)
        query_node = ox.distance.nearest_nodes(self.G, query_lon, query_lat)
        print(f"  Query location (lat={query_lat:.6f}, lon={query_lon:.6f})")
        print(f"  Nearest road node: {query_node}")
        
        # Región Voronoi para ese nodo (el nodo está en la red proyectada)
        region_idx = self._get_region_for_node(query_node)
        hospital_node = self.hospital_nodes[region_idx]
        
        print(f"  Voronoi region: {region_idx + 1}")
        print(f"  Assigned hospital: Hospital {region_idx + 1} (Node {hospital_node})")
        
        return region_idx, hospital_node, query_node
    
    # ---------------------------------------------------------
    # 5) Ruta con A* hasta el hospital de la región
    # ---------------------------------------------------------
    def compute_route(self, query_lat, query_lon):
        """
        Compute the route from a query location to the nearest hospital 
        (according to Voronoi region) using A* on the projected graph.
        """
        print(f"\n{'='*60}")
        print(f"Computing route from ({query_lat:.6f}, {query_lon:.6f})")
        print(f"{'='*60}")
        
        # Usar Voronoi para decidir el hospital
        hospital_idx, hospital_node, query_node = self.find_nearest_hospital(query_lat, query_lon)
        
        # A* en grafo proyectado
        def heuristic(u, v):
            x1, y1 = self.G_proj.nodes[u]['x'], self.G_proj.nodes[u]['y']
            x2, y2 = self.G_proj.nodes[v]['x'], self.G_proj.nodes[v]['y']
            return np.hypot(x1 - x2, y1 - y2)
        
        try:
            route = nx.astar_path(self.G_proj, query_node, hospital_node,
                                  heuristic=heuristic, weight='length')
            route_length = nx.path_weight(self.G_proj, route, weight='length')
            
            print(f"\nRoute found successfully!")
            print(f"  - Number of nodes: {len(route)}")
            print(f"  - Distance: {route_length:.2f} meters ({route_length/1000:.2f} km)")
            print(f"  - Segments: {len(route) - 1}")
            return route, hospital_idx, route_length
        except nx.NetworkXNoPath:
            print("\nNo path found between query location and hospital!")
            return None, hospital_idx, None
    
    # ---------------------------------------------------------
    # 6) Visualización Voronoi
    # ---------------------------------------------------------
    def visualize_voronoi(self, figsize=(15, 12)):
        fig, ax = plt.subplots(figsize=figsize)

        # Red vial proyectada
        ox.plot_graph(self.G_proj, ax=ax, node_size=0, edge_color="lightgray",
                      edge_linewidth=0.5, show=False, close=False)

        # Generate unique colors for each Voronoi region
        num_regions = len(self.voronoi_regions)
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_regions))
        
        # Polígonos Voronoi with unique colors
        for i, poly in enumerate(self.voronoi_regions):
            if poly.geom_type == "Polygon":
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.25, color=colors[i], label=f'Hospital {i+1}')
            elif poly.geom_type == "MultiPolygon":
                for sub_poly in poly.geoms:
                    x, y = sub_poly.exterior.xy
                    ax.fill(x, y, alpha=0.25, color=colors[i])

        # Hospitales
        for n in self.hospital_nodes:
            ax.scatter(self.G_proj.nodes[n]["x"],
                       self.G_proj.nodes[n]["y"],
                       s=100, c="red", marker="+")

        plt.title("Voronoi Partition (Hospitals)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        return fig, ax
    
    # ---------------------------------------------------------
    # 7) Visualización de ruta (en G original)
    # ---------------------------------------------------------
    def visualize_route(self, route, query_lat, query_lon, hospital_idx, figsize=(15, 12)):
        """Visualize a specific route on the map."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the street network (lat/lon)
        ox.plot_graph(self.G, ax=ax, node_size=0, edge_color='lightgray', 
                      edge_linewidth=0.5, show=False, close=False)
        
        # Plot the route (hay que mapear nodos proyectados a nodos en G,
        # pero usamos los mismos ids, así que sirve igual)
        if route:
            ox.plot_graph_route(self.G, route, ax=ax, route_color='blue', 
                                route_linewidth=3, node_size=0, show=False, close=False)
        
        # All hospitals
        hospital_xs = [self.G.nodes[node]['x'] for node in self.hospital_nodes]
        hospital_ys = [self.G.nodes[node]['y'] for node in self.hospital_nodes]
        ax.scatter(hospital_xs, hospital_ys, c='lightcoral', s=150, marker='+', 
                   linewidths=2, zorder=4)
        
        # Destination hospital
        dest_x = self.G.nodes[self.hospital_nodes[hospital_idx]]['x']
        dest_y = self.G.nodes[self.hospital_nodes[hospital_idx]]['y']
        ax.scatter([dest_x], [dest_y], c='red', s=300, marker='*', 
                   label=f'Destination: Hospital {hospital_idx+1}', zorder=5)
        
        # Query location
        ax.scatter([query_lon], [query_lat], c='green', s=200, marker='o', 
                   label='Query Location', zorder=5)
        
        ax.set_title('Route to Nearest Hospital (A* Algorithm)', fontsize=16)
        ax.legend()
        plt.tight_layout()
        return fig, ax

# Main execution
if __name__ == "__main__":
    # Initialize planner
    planner = HospitalRoutePlanner(LOCATION, DISTANCE, NETWORK_TYPE)
    
    # Step 1: Find nearest nodes for hospitals
    hospital_nodes = planner.find_nearest_nodes()
    
    # Step 2: Create Voronoi partition
    voronoi_regions = planner.create_voronoi_partition()
    
    # Visualize Voronoi partition
    print("\nVisualizing Voronoi partition...")
    fig1, ax1 = planner.visualize_voronoi()
    plt.savefig('voronoi_partition.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 3: Example - Find route from a query location
    # Example query location (somewhere in Puerto Vallarta)
    query_lat, query_lon = 20.635, -105.225
    
    route, hospital_idx, route_length = planner.compute_route(query_lat, query_lon)
    
    if route:
        print("\nVisualizing route...")
        fig2, ax2 = planner.visualize_route(route, query_lat, query_lon, hospital_idx)
        plt.savefig('route_to_hospital.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Interactive example function
    def find_route_to_hospital(lat, lon):
        """Helper function to find route from any location."""
        route, hosp_idx, length = planner.compute_route(lat, lon)
        if route:
            planner.visualize_route(route, lat, lon, hosp_idx)
            plt.show()
            return route, length
        return None, None
    
    print("\n" + "="*60)
    print("Route planning system ready!")
    print("Use find_route_to_hospital(lat, lon) to find routes from any location")
   

