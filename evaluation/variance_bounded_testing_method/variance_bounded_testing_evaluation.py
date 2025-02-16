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

    print(param_values)

    points = scaler.inverse_transform(re_sampled_scaled)

    # Step 1: take the param_values and generate the test case file for simulation
        # --> focus on test case amount between 8-100 (less makes no sense, and more as well)

    # Parameter distribution cannot be used, as already concrete test cases are defined (with the explicit values)
    # Take the param values and generate the individual scenarios (changing s_delta and v_delta)
    # New files for run_distribution needs to be created --> as input at least the amount of test cases and the folder of the scenarios
    # Then, using the index (refering to the param values), the saved param values need be loaded and then the ego max dec needs to be determined (based on the index)
    # For colab usage this needs to be combined in a Juypter notebook...
    # -------------------------

    # Step 2: run the simulation and get the results (pass/fail info)
    # Step 3 (optional): place additional evaluation points depending on the shape/position of the logistic regression plane compared to the orignal
    # Step 4: load the GT results and perform the comparison/evaluation

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

