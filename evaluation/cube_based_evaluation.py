import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# Function to define the vertices of a cube centered at a given point with different sizes
def create_cube(center, size):
    x, y, z = center
    size_x, size_y, size_z = size
    vertices = np.array([
        [x - size_x, y - size_y, z - size_z],  # Vertex 1
        [x + size_x, y - size_y, z - size_z],  # Vertex 2
        [x + size_x, y + size_y, z - size_z],  # Vertex 3
        [x - size_x, y + size_y, z - size_z],  # Vertex 4
        [x - size_x, y - size_y, z + size_z],  # Vertex 5
        [x + size_x, y - size_y, z + size_z],  # Vertex 6
        [x + size_x, y + size_y, z + size_z],  # Vertex 7
        [x - size_x, y + size_y, z + size_z]   # Vertex 8
    ])
    return vertices

# Function to define the edges of a cube
def create_edges():
    return np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ])


class CubeBasedEvaluation:
    """Evaluation of test case values by means of evaluation points including labels 
    (groups) and pass/fail information by comparison with a reference GT cube."""
    
    def __init__(self, 
            gt_cube_sizes: np.ndarray, 
            gt_sub_cube_centers: np.ndarray, 
            gt_pass_fail: list) -> None:
        
        """
        :param gt_cube_sizes: Ground truth cube sizes
        :type gt_cube_sizes: np.ndarray
        :param gt_sub_cube_centers: Ground truth sub-cube centers
        :type gt_sub_cube_centers: np.ndarray
        :param gt_pass_fail: Ground truth pass/fail labels
        :type gt_pass_fail: list
        """
        
        self.gt_cube_sizes = gt_cube_sizes
        self.gt_sub_cube_centers = gt_sub_cube_centers
        self.gt_pass_fail = gt_pass_fail


    def _analyze_list(self, entries):
        # Check if all entries are the same
        if all(entry == entries[0] for entry in entries):
            return True, None, None  # All entries are the same

        # Count occurrences of each entry
        counter = Counter(entries)
        
        # Find the most common entry
        most_common_entry, most_common_count = counter.most_common(1)[0]
        
        # Find all indices of the most common entry
        indices = [index for index, entry in enumerate(entries) if entry == most_common_entry]
        
        return False, most_common_entry, indices
    

    def _find_points_in_sub_cubes(self, 
                                  points: np.ndarray, 
                                  groups: list, 
                                  pass_fail: list) -> dict:
        """
        For every point/group value pair, find the respective sub-cube.

        :param points: Evaluation points.
        :type points: np.ndarray
        :param groups: Information about the different labels (inherently the test case values) for each evaluation point.
        :type groups: list
        :param pass_fail: Information about pass/fail for each evaluation point based on the test case.
        :type pass_fail: list
        :return: Sub-cube with points dictionary.
        :rtype: dict
        """

        sub_cubes_with_points = {}
    
        for point, group, single_pass_fail in zip(points, groups, pass_fail):
            
            # Determine to which sub-cube center the point belongs to
            original_cube_center = None
            for idx, center in enumerate(self.gt_sub_cube_centers):
                size = self.gt_cube_sizes[idx]
                if np.all(np.abs(point - center) < size):
                    original_cube_center = center
                    sub_cube_index = idx
                    break
            
            if original_cube_center is None:
                print(f"Point {point} does not belong to any sub-cube.")
                continue
            
            # If the sub-cube index is not in the dictionary, initialize it
            if sub_cube_index not in sub_cubes_with_points:
                sub_cubes_with_points[sub_cube_index] = {
                    'group': [group],
                    'pass_fail': [single_pass_fail],
                    'points': [point],
                    'sub_cube_center': original_cube_center
                }
            else:
                sub_cubes_with_points[sub_cube_index]['group'].append(group)
                sub_cubes_with_points[sub_cube_index]['points'].append(point)
                sub_cubes_with_points[sub_cube_index]['pass_fail'].append(single_pass_fail)

        
        return sub_cubes_with_points
    

    def evaluate_gt_cubes_based_on_eval_points(self, 
                                               points: np.ndarray, 
                                               groups: list, 
                                               pass_fail: list) -> dict:
        """
        Evaluate ground truth cubes based on evaluation points.

        :param points: Evaluation points.
        :type points: np.ndarray
        :param groups: Information about the different labels (inherently the test case values) for each evaluation point.
        :type groups: list
        :param pass_fail: Information about pass/fail for each evaluation point based on the test case.
        :type pass_fail: list
        :return: _description_
        :rtype: dict
        """

        # Find which sub-cubes contain points
        sub_cubes_with_points = self._find_points_in_sub_cubes(points=points,
                                                               groups=groups,
                                                               pass_fail=pass_fail)

        # Output the results
        for sub_cube_index, data in sub_cubes_with_points.items():
            print(f"Sub-cube {sub_cube_index} contains points from group {data['group']} (original cube center: {data['sub_cube_center']}): {data['points']}")
            # Example usage
            all_same, most_common, indices = self._analyze_list(data['group'])

            if not all_same:
                updated_points = [data['points'][i] for i in indices]
                sub_cubes_with_points[sub_cube_index]['points'] = updated_points
                sub_cubes_with_points[sub_cube_index]['group'] = [data['group'][i] for i in indices]

        print('after update -------------')
        for sub_cube_index, data in sub_cubes_with_points.items():
            print(f"Sub-cube {sub_cube_index} contains points from group {data['group']} (original cube center: {data['sub_cube_center']}): {data['points']}")

        print("Number of unique sub-cubes:", len(sub_cubes_with_points))

        return sub_cubes_with_points
    

    def assign_all_sub_cubes(self, 
                             sub_cubes_with_points: dict, 
                             points: np.ndarray, 
                             groups: list, 
                             pass_fail: list) -> dict:
        """
        Making sure each GT sub-cube got assigned at least one evaluation point.

        :param sub_cubes_with_points: Dict containing all information regarding the evaluation data mapped to each GT sub-cube.
        :type sub_cubes_with_points: dict
        :param points: Evaluation points.
        :type points: np.ndarray
        :param groups: Information about the different labels (inherently the test case values) for each evaluation point.
        :type groups: list
        :param pass_fail: Information about pass/fail for each evaluation point based on the test case.
        :type pass_fail: list
        :return: Updated sub_cubes_with_points dict.
        :rtype: dict
        """

        for idx, sub_cube_center in enumerate(self.gt_sub_cube_centers):

            if idx not in sub_cubes_with_points:
                # Find closest point to sub-cube center
                closest_point_idx = np.argmin(np.linalg.norm(points - sub_cube_center, axis=1))
                closest_point = points[closest_point_idx]
                closest_group = groups[closest_point_idx]
                closest_pass_fail = pass_fail[closest_point_idx]
                sub_cubes_with_points[idx] = {'sub_cube_center': sub_cube_center, 'points': [closest_point
                ], 'groups': [closest_group], 'pass_fail': [closest_pass_fail]}
            else:
                print(f"Sub-cube {idx} already has points assigned")
                continue

        print("Number of unique sub-cubes:", len(sub_cubes_with_points))
        print("Number of total sub cubes:", len(self.gt_sub_cube_centers))

        if len(sub_cubes_with_points) != len(self.gt_sub_cube_centers):
            print("Warning: Not all sub-cubes have points assigned")

        return sub_cubes_with_points


    def plot_cubes(self, 
                 sub_cubes_with_points: dict, 
                 points: np.ndarray):
        """
        Plotting sub-cubes and points.

        :param sub_cubes_with_points: Dict containing all information regarding the evaluation data mapped to each GT sub-cube.
        :type sub_cubes_with_points: dict
        :param points: Evaluation points.
        :type points: np.ndarray
        """

        fig = plt.figure(figsize=(10, 8))  # Increase figure size for better visualization
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.gt_sub_cube_centers[:,0], self.gt_sub_cube_centers[:,1], self.gt_sub_cube_centers[:,2])

        # Plot each small cube
        for idx, sub_cube_center in enumerate(self.gt_sub_cube_centers):
            center = (sub_cube_center[0], sub_cube_center[1], sub_cube_center[2])
            size = self.gt_cube_sizes[idx]
            vertices = create_cube(center, size)
            edges = create_edges()

            if idx in sub_cubes_with_points:
                color = 'black'
                alpha = 1.0
            else:
                color = 'b'
                alpha = 0.1

            # Plot the edges of the cube
            for edge in edges:
                ax.plot(vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], c=color, alpha=alpha)

        ax.scatter(points [:,0], points[:,1], points[:,2], c='black')

        plt.show()


    def calculate_comparison_metric(self, 
                             sub_cubes_with_points: dict) -> float:
        """
        Calculate overlap between pass and fail decision (GT and evaluation).

        :param sub_cubes_with_points: Dict containing all information regarding the evaluation data mapped to each GT sub-cube.
        :type sub_cubes_with_points: dict
        :return: Overlap between pass and fail decision in percent.
        :rtype: float
        """
        
        comparison = np.zeros(len(self.gt_pass_fail))

        for idx, sub_cube_center in enumerate(self.gt_sub_cube_centers):
            gt_pass_fail = self.gt_pass_fail[idx]
            eval_pass_fail = sub_cubes_with_points[idx]['pass_fail'][0]
            
            comparison[idx] = True if gt_pass_fail == eval_pass_fail else False

        comparison_true = list(filter(lambda x: x == 1, comparison))
        len_comparison_true = len(comparison_true)

        return (len_comparison_true/len(self.gt_pass_fail))*100


if __name__ == '__main__':

    # Define the sizes of each small cube for each dimension
    cube_sizes = np.array([
        [0.5, 0.5, 1.5],  # Size for sub-cube 1
        [0.5, 0.5, 1.5],  # Size for sub-cube 2
        [0.5, 0.5, 1.5],  # Size for sub-cube 3
        [0.5, 0.5, 1.5],  # Size for sub-cube 4
        [0.5, 0.5, 1.5],  # Size for sub-cube 5
        [0.5, 0.5, 1.5],  # Size for sub-cube 6
        [0.5, 0.5, 1.5],  # Size for sub-cube 7
        [0.5, 0.5, 1.5]   # Size for sub-cube 8
    ])

    # Define the centers of the sub-cubes
    sub_cube_centers = np.array([
        [0, 0, 0],
        [0, 0, 3],
        [0, 1, 0],
        [0, 1, 3],
        [1, 0, 0],
        [1, 0, 3],
        [1, 1, 0],
        [1, 1, 3]
    ])

    gt_pass_fail = [True, False, True, False, True, True, False, False]


    # Example points and their corresponding groups
    points = np.array([
        [0.1, 0.1, 0.1],
        [0.6, 0.6, 0.6],
        [1.2, 1.2, 1.2],
        [1.5, 1.5, 1.5],
        [2.0, 2.0, 2.0],
        [0.4, 0.4, 0.4],
        [1.1, 1.1, 1.1],
        [1.4, 1.4, 1.4],
        [0.3, 0.3, 0.3],
    ])

    # Corresponding groups for each point
    groups = ['A', 'A', 'B', 'B', 'C', 'A', 'A', 'A', 'A']  # Group 'A' has a conflict with point from group 'B'

    # Pass/Fail information
    pass_fail = [True, True, True, True, False, True, True, True, True]


    comparison_gt_evaluation = CubeBasedEvaluation(gt_cube_sizes=cube_sizes,
                                                   gt_sub_cube_centers=sub_cube_centers,
                                                   gt_pass_fail=gt_pass_fail)
    sub_cubes_with_points = comparison_gt_evaluation.evaluate_gt_cubes_based_on_eval_points(points=points,
                                                                                            groups=groups,
                                                                                            pass_fail=pass_fail)
    
    sub_cubes_with_points = comparison_gt_evaluation.assign_all_sub_cubes(sub_cubes_with_points=sub_cubes_with_points,
                                                                          points=points,
                                                                          groups=groups,
                                                                          pass_fail=pass_fail)
    
    #comparison_gt_evaluation.plot_cubes(sub_cubes_with_points=sub_cubes_with_points,
    #                                  points=points)
    
    metric = comparison_gt_evaluation.calculate_comparison_metric(sub_cubes_with_points=sub_cubes_with_points)
    print(metric)