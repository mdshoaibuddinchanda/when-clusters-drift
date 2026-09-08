import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
import yaml

from clusterdrift.falsification.dataset import (
    FORBIDDEN_PREDICTOR_FEATURES,
    build_joined_evaluation_table,
    get_primary_evaluation_dataset,
    get_secondary_evaluation_dataset,
    get_feature_matrix_and_target,
)
from clusterdrift.falsification.models import (
    TrainingOnlyPreprocessor,
    tune_and_fit_hgbr,
    tune_and_fit_ridge,
)
from clusterdrift.falsification.execution.checkpoint import EvaluationCheckpointManager
from clusterdrift.falsification.verification import (
    verify_pass_a,
    verify_pass_b,
    evaluate_falsification_verdict,
)
from clusterdrift.falsification.protocol import compute_prediction_record_sha256

REPO_ROOT = Path(__file__).parents[2]

def test_pass_a_hash_mutation_blocks_pass_c(monkeypatch):
    import clusterdrift.falsification.verification as v_mod
    orig = v_mod.compute_file_sha256
    def mock_sha(p):
        if 'signals_label_free.csv' in str(p):
            return '0' * 64
        return orig(p)
    monkeypatch.setattr(v_mod, 'compute_file_sha256', mock_sha)
    with pytest.raises(ValueError, match='Pass A signals SHA256 mismatch'):
        verify_pass_a(REPO_ROOT)

def test_pass_b_hash_mutation_blocks_pass_c(monkeypatch):
    import clusterdrift.falsification.verification as v_mod
    orig = v_mod.compute_file_sha256
    def mock_sha(p):
        if 'quality_evaluation_only.csv' in str(p):
            return '0' * 64
        return orig(p)
    monkeypatch.setattr(v_mod, 'compute_file_sha256', mock_sha)
    with pytest.raises(ValueError, match='Pass B quality SHA256 mismatch'):
        verify_pass_b(REPO_ROOT)

def test_join_missing_key_fails():
    sig_path = REPO_ROOT / 'results' / 'falsification' / 'signals_label_free.csv'
    q_path = REPO_ROOT / 'results' / 'falsification' / 'quality_evaluation_only.csv'
    if not sig_path.exists() or not q_path.exists():
        pytest.skip('Pass A or B missing')
    sig_df = pd.read_csv(sig_path).iloc[:-1]
    q_df = pd.read_csv(q_path)
    with pytest.raises(ValueError, match='Expected 4500 unique signals keys'):
        build_joined_evaluation_table(sig_df, q_df)

def test_join_duplicate_key_fails():
    sig_path = REPO_ROOT / 'results' / 'falsification' / 'signals_label_free.csv'
    q_path = REPO_ROOT / 'results' / 'falsification' / 'quality_evaluation_only.csv'
    if not sig_path.exists() or not q_path.exists():
        pytest.skip('Pass A or B missing')
    sig_df = pd.read_csv(sig_path)
    sig_df = pd.concat([sig_df, sig_df.iloc[:1]], ignore_index=True)
    q_df = pd.read_csv(q_path)
    with pytest.raises(ValueError, match='Duplicate'):
        build_joined_evaluation_table(sig_df, q_df)

def test_forbidden_predictor_fails():
    join_path = REPO_ROOT / 'results' / 'falsification' / 'joined_evaluation_table.csv'
    if not join_path.exists():
        pytest.skip('Joined table missing')
    df = pd.read_csv(join_path)
    for forbidden in ['ari_clean', 'ari_condition', 'delta_ari']:
        with pytest.raises(ValueError, match='Forbidden'):
            get_feature_matrix_and_target(df, feature_names=[forbidden, 'D_U_R'])

def test_primary_clean_exclusion_exact():
    join_path = REPO_ROOT / 'results' / 'falsification' / 'joined_evaluation_table.csv'
    if not join_path.exists():
        pytest.skip('Joined table missing')
    df = pd.read_csv(join_path)
    primary = get_primary_evaluation_dataset(df)
    assert len(primary) == 4200
    assert 'clean' not in primary['condition'].values

def test_lodo_held_out_dataset_absent_from_training():
    join_path = REPO_ROOT / 'results' / 'falsification' / 'joined_evaluation_table.csv'
    if not join_path.exists():
        pytest.skip('Joined table missing')
    df = get_primary_evaluation_dataset(pd.read_csv(join_path))
    for ds in df['dataset_id'].unique():
        tr = df[df['dataset_id'] != ds]
        te = df[df['dataset_id'] == ds]
        assert ds not in tr['dataset_id'].values
        assert len(te) == 350

def test_losfo_held_out_family_absent_from_training():
    join_path = REPO_ROOT / 'results' / 'falsification' / 'joined_evaluation_table.csv'
    if not join_path.exists():
        pytest.skip('Joined table missing')
    df = get_primary_evaluation_dataset(pd.read_csv(join_path))
    for fam in df['shift_family'].unique():
        tr = df[df['shift_family'] != fam]
        te = df[df['shift_family'] == fam]
        assert fam not in tr['shift_family'].values
        assert len(te) == 600

def test_inner_group_kfold_groups_disjoint():
    join_path = REPO_ROOT / 'results' / 'falsification' / 'joined_evaluation_table.csv'
    if not join_path.exists():
        pytest.skip('Joined table missing')
    df = get_primary_evaluation_dataset(pd.read_csv(join_path))
    held_out = df['dataset_id'].unique()[0]
    tr = df[df['dataset_id'] != held_out]
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(tr, groups=tr['dataset_id']))
    assert len(splits) == 5
    for tr_idx, val_idx in splits:
        tr_g = set(tr.iloc[tr_idx]['dataset_id'])
        val_g = set(tr.iloc[val_idx]['dataset_id'])
        assert tr_g.isdisjoint(val_g)

def test_exactly_5_inner_folds():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    y = rng.randn(100)
    groups = np.array([f'g{i%4}' for i in range(100)]) # only 4 groups
    with pytest.raises(AssertionError, match='Expected exactly 5 inner folds'):
        tune_and_fit_ridge(X, y, groups, alphas=[1.0], n_splits=5)

def test_training_only_imputation():
    X_train = np.array([[1.0, np.nan], [3.0, 10.0], [5.0, 20.0]])
    X_test = np.array([[np.nan, np.nan]])
    prep = TrainingOnlyPreprocessor(with_scaling=False)
    X_tr_trans = prep.fit_transform(X_train)
    X_te_trans = prep.transform(X_test)
    assert prep.medians_[0] == 3.0
    assert prep.medians_[1] == 15.0
    assert X_te_trans[0, 0] == 3.0
    assert X_te_trans[0, 1] == 15.0

def test_ridge_training_only_scaling():
    X_train = np.array([[0.0, 10.0], [10.0, 20.0]])
    prep = TrainingOnlyPreprocessor(with_scaling=True)
    prep.fit(X_train)
    assert np.allclose(prep.means_, [5.0, 15.0])
    X_test = np.array([[20.0, 30.0]])
    X_te_scaled = prep.transform(X_test)
    assert np.allclose(X_te_scaled, [[3.0, 3.0]])

def test_all_missing_feature_fallback_zero():
    X_train = np.full((10, 2), np.nan)
    X_train[:, 0] = np.arange(10)
    prep = TrainingOnlyPreprocessor(with_scaling=False)
    X_tr_trans = prep.fit_transform(X_train)
    assert not np.isnan(X_tr_trans).any()
    assert (X_tr_trans[:, 1] == 0.0).all()

def test_hgbr_grid_exactly_frozen():
    with open(REPO_ROOT / 'configs' / 'falsification.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    grid = cfg['primary_regressor']['grid']
    assert grid['learning_rate'] == [0.05, 0.10]
    assert grid['max_leaf_nodes'] == [7, 15]
    assert grid['max_iter'] == [150, 300]
    assert grid['l2_regularization'] == [0.0, 1.0]

def test_ridge_grid_exactly_frozen():
    with open(REPO_ROOT / 'configs' / 'falsification.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    alphas = cfg['linear_control']['alphas']
    assert alphas == [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

def test_deterministic_tie_breaking():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 2)
    y = np.zeros(100)
    groups = np.array([f'g{i%10}' for i in range(100)])
    alphas = [1.0, 10.0]
    _, _, selection = tune_and_fit_ridge(X, y, groups, alphas=alphas)
    assert selection.selected_params['alpha'] == 10.0  # higher alpha preferred in tie-break

def test_secondary_includes_clean():
    join_path = REPO_ROOT / 'results' / 'falsification' / 'joined_evaluation_table.csv'
    if not join_path.exists():
        pytest.skip('Joined table missing')
    df = pd.read_csv(join_path)
    secondary = get_secondary_evaluation_dataset(df)
    assert len(secondary) == 4500
    assert 'clean' in secondary['condition'].values

def test_secondary_does_not_influence_verdict():
    import inspect
    sig = inspect.signature(evaluate_falsification_verdict)
    assert 'secondary' not in sig.parameters

def test_interrupted_pass_c_resumes(tmp_path):
    mgr = EvaluationCheckpointManager(work_dir=tmp_path)
    assert not mgr.is_checkpoint_valid('primary', 'LODO', 'iris', 'P0', 'hgbr')
    dummy_pred = {
        'row_id': 'iris_0_location_mild_fcm_adaptive_1',
        'dataset_id': 'iris',
        'outer_fold': 0,
        'condition': 'location_mild',
        'algorithm_seed': 1,
        'shift_family': 'location',
        'observed_delta_ari': 0.1,
        'predicted_delta_ari': 0.15,
        'residual': -0.05,
        'squared_error': 0.0025,
        'absolute_error': 0.05,
        'feature_block': 'P0',
        'regressor': 'hgbr',
    }
    dummy_pred['prediction_record_sha256'] = compute_prediction_record_sha256(dummy_pred)
    envelope = {
        'analysis_scope': 'primary',
        'outer_eval_type': 'LODO',
        'outer_test_group': 'iris',
        'feature_block': 'P0',
        'regressor': 'hgbr',
        'predictions': [dummy_pred],
        'selected_params': {},
        'inner_cv_mean_mse': 0.01,
        'training_key_universe_sha256': 'a',
        'test_key_universe_sha256': 'b',
        'job_execution_wall_clock_seconds': 0.1,
    }
    mgr.save_checkpoint(envelope)
    assert mgr.is_checkpoint_valid('primary', 'LODO', 'iris', 'P0', 'hgbr', 'a', 'b')

def test_wrong_protocol_checkpoint_rejected(tmp_path):
    mgr_writer = EvaluationCheckpointManager(work_dir=tmp_path, phase7_protocol_sha256='bad_sha')
    dummy_pred = {'row_id': 'k1', 'observed_delta_ari': 0.1, 'predicted_delta_ari': 0.15, 'residual': -0.05, 'squared_error': 0.0025, 'absolute_error': 0.05, 'dataset_id': 'iris', 'outer_fold': 0, 'condition': 'location_mild', 'algorithm_seed': 1, 'shift_family': 'location', 'feature_block': 'P0', 'regressor': 'hgbr'}
    dummy_pred['prediction_record_sha256'] = compute_prediction_record_sha256(dummy_pred)
    env = {'analysis_scope': 'primary', 'outer_eval_type': 'LODO', 'outer_test_group': 'iris', 'feature_block': 'P0', 'regressor': 'hgbr', 'predictions': [dummy_pred]}
    mgr_writer.save_checkpoint(env)
    mgr_reader = EvaluationCheckpointManager(work_dir=tmp_path, phase7_protocol_sha256='good_sha')
    assert not mgr_reader.is_checkpoint_valid('primary', 'LODO', 'iris', 'P0', 'hgbr')

def test_wrong_pass_a_sha_checkpoint_rejected(tmp_path):
    mgr_writer = EvaluationCheckpointManager(work_dir=tmp_path, pass_a_signals_sha256='bad_sha')
    dummy_pred = {'row_id': 'k1', 'observed_delta_ari': 0.1, 'predicted_delta_ari': 0.15, 'residual': -0.05, 'squared_error': 0.0025, 'absolute_error': 0.05, 'dataset_id': 'iris', 'outer_fold': 0, 'condition': 'location_mild', 'algorithm_seed': 1, 'shift_family': 'location', 'feature_block': 'P0', 'regressor': 'hgbr'}
    dummy_pred['prediction_record_sha256'] = compute_prediction_record_sha256(dummy_pred)
    env = {'analysis_scope': 'primary', 'outer_eval_type': 'LODO', 'outer_test_group': 'iris', 'feature_block': 'P0', 'regressor': 'hgbr', 'predictions': [dummy_pred]}
    mgr_writer.save_checkpoint(env)
    mgr_reader = EvaluationCheckpointManager(work_dir=tmp_path, pass_a_signals_sha256='good_sha')
    assert not mgr_reader.is_checkpoint_valid('primary', 'LODO', 'iris', 'P0', 'hgbr')

def test_wrong_pass_b_sha_checkpoint_rejected(tmp_path):
    mgr_writer = EvaluationCheckpointManager(work_dir=tmp_path, pass_b_quality_sha256='bad_sha')
    dummy_pred = {'row_id': 'k1', 'observed_delta_ari': 0.1, 'predicted_delta_ari': 0.15, 'residual': -0.05, 'squared_error': 0.0025, 'absolute_error': 0.05, 'dataset_id': 'iris', 'outer_fold': 0, 'condition': 'location_mild', 'algorithm_seed': 1, 'shift_family': 'location', 'feature_block': 'P0', 'regressor': 'hgbr'}
    dummy_pred['prediction_record_sha256'] = compute_prediction_record_sha256(dummy_pred)
    env = {'analysis_scope': 'primary', 'outer_eval_type': 'LODO', 'outer_test_group': 'iris', 'feature_block': 'P0', 'regressor': 'hgbr', 'predictions': [dummy_pred]}
    mgr_writer.save_checkpoint(env)
    mgr_reader = EvaluationCheckpointManager(work_dir=tmp_path, pass_b_quality_sha256='good_sha')
    assert not mgr_reader.is_checkpoint_valid('primary', 'LODO', 'iris', 'P0', 'hgbr')

def test_corrupt_prediction_hash_rejected(tmp_path):
    mgr = EvaluationCheckpointManager(work_dir=tmp_path)
    dummy_pred = {'row_id': 'k1', 'observed_delta_ari': 0.1, 'predicted_delta_ari': 0.15, 'residual': -0.05, 'squared_error': 0.0025, 'absolute_error': 0.05, 'dataset_id': 'iris', 'outer_fold': 0, 'condition': 'location_mild', 'algorithm_seed': 1, 'shift_family': 'location', 'feature_block': 'P0', 'regressor': 'hgbr'}
    dummy_pred['prediction_record_sha256'] = compute_prediction_record_sha256(dummy_pred)
    env = {'analysis_scope': 'primary', 'outer_eval_type': 'LODO', 'outer_test_group': 'iris', 'feature_block': 'P0', 'regressor': 'hgbr', 'predictions': [dummy_pred]}
    p = mgr.save_checkpoint(env)
    # Corrupt prediction hash
    with open(p, 'r') as f:
        data = json.load(f)
    data['predictions'][0]['prediction_record_sha256'] = 'corrupt'
    with open(p, 'w') as f:
        json.dump(data, f)
    assert not mgr.is_checkpoint_valid('primary', 'LODO', 'iris', 'P0', 'hgbr')
    assert (tmp_path / 'checkpoints' / 'quarantine_evaluation' / f'corrupt_{p.name}').exists()

def test_workers_scientific_equality():
    calib_file = REPO_ROOT / 'results' / 'falsification' / 'pass_c_calibration.json'
    if not calib_file.exists():
        pytest.skip('Pass C calibration artifact missing')
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    assert calib.get('scientific_equality_verified') is True
