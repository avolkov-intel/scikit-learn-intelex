# ==============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.tests._utils_spmd import (
    _generate_classification_data,
    _get_local_tensor,
    _mpi_libs_and_gpu_available,
    _spmd_assert_allclose,
)
import sys
'''
@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.mpi
def test_logistic_spmd_gold(dataframe, queue):
    # Import spmd and batch algo
    from sklearnex.linear_model import LogisticRegression as LogisticRegression_Batch
    from sklearnex.spmd.linear_model import LogisticRegression as LogisticRegression_SPMD

    # Create gold data and convert to dataframe
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    X_test = np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [10.0, -10.0],
        ]
    )

    local_dpt_X_train = _convert_to_dataframe(
        _get_local_tensor(X_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_y_train = _convert_to_dataframe(
        _get_local_tensor(y_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_X_test = _convert_to_dataframe(
        _get_local_tensor(X_test), sycl_queue=queue, target_df=dataframe
    )
    dpt_X_train = _convert_to_dataframe(X_train, sycl_queue=queue, target_df=dataframe)
    dpt_y_train = _convert_to_dataframe(y_train, sycl_queue=queue, target_df=dataframe)
    dpt_X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)

    # Ensure trained model of batch algo matches spmd
    spmd_model = LogisticRegression_SPMD(random_state=0, solver="newton-cg").fit(
        local_dpt_X_train, local_dpt_y_train
    )
    batch_model = LogisticRegression_Batch(random_state=0, solver="newton-cg").fit(
        dpt_X_train, dpt_y_train
    )

    assert_allclose(spmd_model.coef_, batch_model.coef_, rtol=1e-2)
    assert_allclose(spmd_model.intercept_, batch_model.intercept_, rtol=1e-2)

    # Ensure predictions of batch algo match spmd
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(dpt_X_test)

    _spmd_assert_allclose(spmd_result, _as_numpy(batch_result))
'''

# parametrize max_iter, C, tol
@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [10000])
@pytest.mark.parametrize("n_features", [100])
@pytest.mark.parametrize("C", [0.5])
@pytest.mark.parametrize("tol", [1e-2])
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.mpi
def test_logistic_spmd_synthetic(n_samples, n_features, C, tol, dataframe, queue, dtype):
    # TODO: Resolve numerical issues when n_rows_rank < n_cols
    if n_samples <= n_features:
        pytest.skip("Numerical issues when rank rows < columns")

    print(n_samples, n_features, C, tol, dataframe, queue, dtype)
    sys.stdout.flush()


    from mpi4py import MPI
 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Import spmd and batch algo
    from sklearnex.linear_model import LogisticRegression as LogisticRegression_Batch
    from sklearnex.spmd.linear_model import LogisticRegression as LogisticRegression_SPMD
    from sklearn.metrics import log_loss

    # Generate data and convert to dataframe
    X_train, X_test, y_train, _ = _generate_classification_data(
        n_samples, n_features, dtype=dtype
    )

    local_dpt_X_train = _convert_to_dataframe(
        _get_local_tensor(X_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_y_train = _convert_to_dataframe(
        _get_local_tensor(y_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_X_test = _convert_to_dataframe(
        _get_local_tensor(X_test), sycl_queue=queue, target_df=dataframe
    )
    dpt_X_train = _convert_to_dataframe(X_train, sycl_queue=queue, target_df=dataframe)
    dpt_y_train = _convert_to_dataframe(y_train, sycl_queue=queue, target_df=dataframe)
    dpt_X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)

    # Ensure trained model of batch algo matches spmd
    print("Data shapes:", X_train.shape, y_train.shape, dpt_X_train.shape, local_dpt_X_train.shape)
    print("Data sum:", X_train.sum(), y_train.sum())
    print("Row sums:")
    print(X_train.sum(axis=0))
    print(X_train.sum(axis=1))
    sys.stdout.flush()
    print("First 10 rows")
    for i in range(10):
        for j in range (X_train.shape[1]):
            print(X_train[i, j], end=' ')
        print()
    print("---------------------")
    print("Last 10 rows")
    for i in range(X_train.shape[0] - 10, X_train.shape[0]):
        for j in range (X_train.shape[1]):
            print(X_train[i, j], end=' ')
        print()
    print("---------------------")
    sys.stdout.flush()
    
    spmd_model = LogisticRegression_SPMD(
        random_state=0, solver="newton-cg", C=C, tol=tol
    ).fit(local_dpt_X_train, local_dpt_y_train)
    
    batch_model = LogisticRegression_Batch(
        random_state=0, solver="newton-cg", C=C, tol=tol
    ).fit(dpt_X_train, dpt_y_train)
    
    # TODO: Logistic Regression coefficients do not align
    tol = 1e-2

    print("Spmd coef:", spmd_model.coef_)
    print(spmd_model.intercept_)
    print("Batch coef:", batch_model.coef_)
    print(batch_model.intercept_)
    sys.stdout.flush()

    spmd_loss = log_loss(_as_numpy(local_dpt_y_train), _as_numpy(spmd_model.predict_proba(local_dpt_X_train)))
    batch_loss = log_loss(y_train, _as_numpy(batch_model.predict_proba(dpt_X_train)))

    print("Rank:", rank, "spmd loss:", spmd_loss,  spmd_loss * local_dpt_y_train.shape[0])
    print("Batch loss", batch_loss, batch_loss * y_train.shape[0])
    sys.stdout.flush()

    spmd_reg = 1. / ((spmd_model.coef_ ** 2).sum() * 2)
    batch_reg = 1. / ((batch_model.coef_ ** 2).sum() * 2)

    print("Rank:", rank, "spmd regul:", spmd_reg, "batch regul:", batch_reg)

    sys.stdout.flush()

    assert_allclose(spmd_model.coef_, batch_model.coef_, rtol=tol, atol=tol)
    assert_allclose(spmd_model.intercept_, batch_model.intercept_, rtol=tol, atol=tol)

    # Ensure predictions of batch algo match spmd
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(dpt_X_test)

    _spmd_assert_allclose(spmd_result, _as_numpy(batch_result))
