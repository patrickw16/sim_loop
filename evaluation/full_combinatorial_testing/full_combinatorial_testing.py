import numpy as np
import pandas as pd
from evaluation.cube_based_evaluation import CubeBasedEvaluation, fail_ratio


if __name__ == '__main__':

    #Load GT data
    gt_cubes_sizes = np.load('evaluation/gt_analysis/data/gt_cube_sizes.npy')
    gt_pass_fail = np.load('evaluation/gt_analysis/data/gt_pass_fail.npy')
    gt_sub_cube_centers = np.load('evaluation/gt_analysis/data/gt_sub_cube_centers.npy')
    

    evaluation_dict = dict()
    i_j_combi = [(i, j) for i in np.arange(2,25,2) for j in np.arange(1,3,1)]

    for combi_idx, combi in enumerate(i_j_combi):
        i, j = combi
        #Evaluation data
        s_delta_range = np.arange(10,62,i)
        v_delta_range = np.arange(10,42,i)
        ego_max_dec_range = np.arange(3,8,j)

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

    df.to_csv('evaluation/full_combinatorial_testing/data/full_combinatorial_testing_evaluation.csv', index=False) 