from eval.skill_metrics import summarize


def test_retries_stay_in_action_denominator_and_missing_geometry_is_not_zero():
    result=summarize([{'trial_id':'retry','actual_success':True,'extra':{
        'execution_records':[{'result':{'action':{'skill':'GRASP'},'success':ok,
        'error_code':'NONE' if ok else 'GRASP_MISSED','recovery_attempted':False}} for ok in (False,True)]}},
        {'trial_id':'legacy','actual_success':True,'extra':{}}])
    assert result['actual_successes']==2
    assert result['actions']['GRASP']['success_rate']==.5
    assert result['completed_without_retry_or_recovery']==0
    assert result['trials_with_execution_telemetry']==1
    assert result['mean_center_xy_error_m'] is None


def test_failed_trial_placement_error_still_counts():
    result=summarize([{'trial_id':'failed','actual_success':False,
        'extra':{'placement_geometry':{'center_xy_error_m':.3}}},
        {'trial_id':'passed','actual_success':True,
        'extra':{'placement_geometry':{'center_xy_error_m':.02}}}])
    assert result['placement_geometry_samples']==2
    assert result['mean_center_xy_error_m']==.16 and result['max_center_xy_error_m']==.3
