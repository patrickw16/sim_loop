import numpy as np
import pandas as pd
from evaluation.cube_based_evaluation import CubeBasedEvaluation, fail_ratio



if __name__ == '__main__':
        
    #Load GT data:
    gt_cubes_sizes = np.load('evaluation/gt_analysis/data/gt_cube_sizes.npy')
    gt_pass_fail = np.load('evaluation/gt_analysis/data/gt_pass_fail.npy')
    gt_sub_cube_centers = np.load('evaluation/gt_analysis/data/gt_sub_cube_centers.npy')
    
    # Order the sim log to align the order of the simulation runs with the param values (test case values)
    df_sim = pd.read_csv('evaluation/variance_bounded_testing_method/data/logs/compiled_log_data.csv')
    df_sim['FileNumber'] = df_sim['Filename'].str.extract(r'(\d+)').astype(int) # Extract the first value from the Filename
    result = df_sim[['FileNumber', 'CollisionDetected']]
    ordered_sim_logs = result.sort_values(by='FileNumber')
    ordered_sim_logs.reset_index(drop=True, inplace=True)
    pass_fail = ordered_sim_logs['CollisionDetected'].to_numpy()
    
    points = np.load('evaluation/variance_bounded_testing_method/data/points.npy')
    groups = np.load('evaluation/variance_bounded_testing_method/data/adapted_kmeans_labels.npy').astype(int)
    param_values = np.load('evaluation/variance_bounded_testing_method/data/param_values.npy')

    print(groups)

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
    gt_fail_ratio = fail_ratio(gt_pass_fail)
    eval_fail_ratio = fail_ratio(np.array(pass_fail))
    fail_ratio_difference = (fail_ratio(gt_pass_fail)-fail_ratio(np.array(pass_fail)))*100
    abs_fail_ratio_difference = np.abs(fail_ratio_difference)
    number_of_test_cases = len(points)

    evaluation_dict[combi_idx] = {
        'metric': metric,
        'fail_ratio_difference': fail_ratio_difference,
        'absolute_fail_ratio_difference': abs_fail_ratio_difference,
        'number_of_test_cases': number_of_test_cases,
        'gt_fail_ratio': gt_fail_ratio,
        'eval_fail_ratio': eval_fail_ratio,
        's_delta_range': s_delta_range,
        'v_delta_range': v_delta_range,
        'ego_max_dec_range': ego_max_dec_range,
        'points': points,
        'pass_fail': pass_fail
    }

    df = pd.DataFrame.from_dict(evaluation_dict, orient='index')    
    print(df)

    df.to_csv('evaluation/variance_bounded_testing_method/data/variance_bounded_testing_method_evaluation.csv', index=False) 


