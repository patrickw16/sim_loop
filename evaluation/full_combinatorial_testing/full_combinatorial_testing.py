import numpy as np
from evaluation.cube_based_evaluation import CubeBasedEvaluation


if __name__ == '__main__':

    #Load GT data
    gt_cubes_sizes = np.load('evaluation/gt_analysis/data/gt_cube_sizes.npy')
    gt_pass_fail = np.load('evaluation/gt_analysis/data/gt_pass_fail.npy')
    gt_sub_cube_centers = np.load('evaluation/gt_analysis/data/gt_sub_cube_centers.npy')
    
    #Evaluation data
    s_delta_range = np.arange(10,62,12)
    v_delta_range = np.arange(10,42,12)
    ego_max_dec_range = np.arange(3,8,2)

    # Create a meshgrid
    grid1, grid2, grid3 = np.meshgrid(s_delta_range, v_delta_range, ego_max_dec_range, indexing='ij')

    points = np.vstack([grid1.ravel(), grid2.ravel(), grid3.ravel()]).T
    groups = np.arange(0,len(points), 1)
    pass_fail = list()

    for test_case in points:
        idx = np.where(np.all(gt_sub_cube_centers == test_case, axis=1))[0]
        pass_fail.append(gt_pass_fail[idx][0])

    
    #Compare

    comparison_gt_evaluation = CubeBasedEvaluation(gt_cube_sizes=gt_cubes_sizes,
                                                   gt_sub_cube_centers=gt_sub_cube_centers,
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
    print(f"Metric:{metric}")
    print(f"Number of test cases:{len(points)}")