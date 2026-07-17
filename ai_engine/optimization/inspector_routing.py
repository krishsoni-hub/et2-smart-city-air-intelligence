import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger("Inspector-Deployment-Optimizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class InspectorDeploymentOptimizer:
    """
    Optimizes the deployment routing of municipal inspectors to forecasted 
    pollution hotspots, maximizing enforcement coverage while minimizing travel time.
    Solves a greedy variant of the constrained Vehicle Routing Problem (VRP).
    """
    def __init__(self, num_inspectors: int = 5, max_stops_per_inspector: int = 8):
        self.num_inspectors = num_inspectors
        self.max_stops = max_stops_per_inspector

    def optimize_routes(self, forecasted_hotspots: List[Dict[str, Any]], hq_location: Tuple[float, float]) -> Dict[str, Any]:
        """
        forecasted_hotspots: list of dicts with keys ['grid_id', 'lat', 'lon', 'severity_score']
        hq_location: (lat, lon) of municipal headquarters.
        Returns optimized route assignments for each inspector.
        """
        if not forecasted_hotspots:
            logger.info("No hotspots detected. No deployment needed.")
            return {"routes": []}

        logger.info(f"Optimizing deployment for {self.num_inspectors} inspectors across {len(forecasted_hotspots)} hotspots.")
        
        # Sort hotspots by severity (highest first)
        hotspots = sorted(forecasted_hotspots, key=lambda x: x['severity_score'], reverse=True)
        
        # We can only visit at most (num_inspectors * max_stops) hotspots
        visit_limit = self.num_inspectors * self.max_stops
        target_hotspots = hotspots[:visit_limit]
        
        # Build coordinates matrix: Index 0 is HQ, 1..N are hotspots
        coords = [hq_location] + [(h['lat'], h['lon']) for h in target_hotspots]
        coords_arr = np.array(coords)
        
        # Distance matrix (Euclidean approximation for speed, Haversine in strict prod)
        dist_matrix = squareform(pdist(coords_arr, metric='euclidean'))
        
        routes = {f"Inspector_{i+1}": [] for i in range(self.num_inspectors)}
        unvisited = set(range(1, len(coords)))
        
        # Greedy heuristic assignment
        for inspector_id in routes.keys():
            current_loc_idx = 0 # Start at HQ
            stops = 0
            
            while stops < self.max_stops and unvisited:
                # Find nearest unvisited hotspot
                nearest_idx = None
                min_dist = float('inf')
                
                for candidate_idx in unvisited:
                    d = dist_matrix[current_loc_idx][candidate_idx]
                    if d < min_dist:
                        min_dist = d
                        nearest_idx = candidate_idx
                
                if nearest_idx is not None:
                    # Assign to route
                    hotspot_data = target_hotspots[nearest_idx - 1] # -1 because HQ is index 0
                    routes[inspector_id].append({
                        "grid_id": hotspot_data["grid_id"],
                        "severity_score": hotspot_data["severity_score"],
                        "distance_from_prev": min_dist
                    })
                    unvisited.remove(nearest_idx)
                    current_loc_idx = nearest_idx
                    stops += 1
                else:
                    break

        logger.info(f"Routing optimization complete. Total unvisited critical hotspots: {len(unvisited)}")
        return {"routes": routes, "unvisited_hotspot_count": len(unvisited)}

if __name__ == "__main__":
    optimizer = InspectorDeploymentOptimizer(num_inspectors=2, max_stops_per_inspector=3)
    hq = (28.6139, 77.2090) # New Delhi HQ
    hotspots_mock = [
        {"grid_id": f"GRID-{i}", "lat": hq[0] + np.random.normal(0, 0.05), "lon": hq[1] + np.random.normal(0, 0.05), "severity_score": np.random.uniform(50, 200)}
        for i in range(10)
    ]
    
    plan = optimizer.optimize_routes(hotspots_mock, hq)
    import json
    logger.info(f"Deployment Plan:\n{json.dumps(plan, indent=2)}")
