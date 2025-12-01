class KDNode:
    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, points, depth=0):
        self.k = len(points[0]) if len(points) > 0 else 2
        self.root = self._build_tree(points, depth)
    
    def _build_tree(self, points, depth):
        if len(points) == 0:
            return None
        
        axis = depth % self.k
        points = sorted(points, key=lambda x: x[axis])
        median = len(points) // 2
        
        return KDNode(
            point=points[median],
            axis=axis,
            left=self._build_tree(points[:median], depth + 1),
            right=self._build_tree(points[median + 1:], depth + 1)
        )
    
    def nearest_neighbor(self, query_point):
        if self.root is None:
            return None, float('inf')
        
        best = [None, float('inf')]
        self._nearest_helper(self.root, query_point, best)
        return best[0], best[1]
    
    def _nearest_helper(self, node, query_point, best):
        if node is None:
            return
        
        point = node.point
        dist = self._distance(point, query_point)
        
        if dist < best[1]:
            best[0] = point
            best[1] = dist
        
        axis = node.axis
        diff = query_point[axis] - point[axis]
        
        if diff < 0:
            near_subtree = node.left
            far_subtree = node.right
        else:
            near_subtree = node.right
            far_subtree = node.left
        
        self._nearest_helper(near_subtree, query_point, best)
        
        if abs(diff) < best[1]:
            self._nearest_helper(far_subtree, query_point, best)
    
    def _distance(self, point1, point2):
        return sum((point1[i] - point2[i]) ** 2 for i in range(self.k)) ** 0.5