import numpy as np
from evaluation.cube_based_evaluation import CubeBasedEvaluation
from odd_coverage.coverage_param_lvl.coverage_param_lvl import CoverageParamLvl



if __name__ == '__main__':
        
    #Load GT data:
    gt_cubes_sizes = np.load('evaluation/gt_analysis/data/gt_cube_sizes.npy')
    gt_pass_fail = np.load('evaluation/gt_analysis/data/gt_pass_fail.npy')
    gt_sub_cube_centers = np.load('evaluation/gt_analysis/data/gt_sub_cube_centers.npy')
    
    #Generate evaluation points:
    my_coverage_param = CoverageParamLvl(
                        number_of_test_values=50,
                        trace_epsilon=0.0075,
                        epsilon_buffer=0.005,
                        number_of_attempts=1,
                        combined_points_data_path='./evaluation/variance_bounded_testing_method/data/prior_points_s_delta_v_delta_ego_max_dec.npy')
            
    param_values, cov_contribution, updated_cluster_traces, adapted_kmeans_labels, re_sampled_scaled, scaler, weighted_within_variance = my_coverage_param.get_optimised_values(plot_path='/home/patrick_w/odd_coverage/var_poc_3d_example.svg')

    points = scaler.inverse_transform(re_sampled_scaled)

    pass_fail = [True] * len(adapted_kmeans_labels)

    #Compare

    comparison_gt_evaluation = CubeBasedEvaluation(gt_cube_sizes=gt_cubes_sizes,
                                                   gt_sub_cube_centers=gt_sub_cube_centers,
                                                   gt_pass_fail=gt_pass_fail)
    sub_cubes_with_points = comparison_gt_evaluation.evaluate_gt_cubes_based_on_eval_points(points=points,
                                                                                            groups=adapted_kmeans_labels,
                                                                                            pass_fail=pass_fail)
    
    sub_cubes_with_points = comparison_gt_evaluation.assign_all_sub_cubes(sub_cubes_with_points=sub_cubes_with_points,
                                                                          points=points,
                                                                          groups=adapted_kmeans_labels,
                                                                          pass_fail=pass_fail)
    
    #comparison_gt_evaluation.plot_cubes(sub_cubes_with_points=sub_cubes_with_points,
    #                                  points=points)
    
    metric = comparison_gt_evaluation.calculate_comparison_metric(sub_cubes_with_points=sub_cubes_with_points)
    print(metric)

