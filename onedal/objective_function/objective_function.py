# ===============================================================================
# Copyright 2023 Intel Corporation
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
# ===============================================================================

from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod

import numpy as np
from numbers import Number

from ..common._policy import _get_policy

from ..datatypes._data_conversion import (
    from_table,
    to_table,
    _convert_to_supported,
    _convert_to_dataframe,
)
from onedal import _backend


class BaseObjectiveFunction(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, algorithm, objective_function_method):
        self.algorithm = algorithm
        self.func = objective_function_method

    @staticmethod
    def get_all_result_options():
        return ["value", "gradient", "hessian"]

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _get_result_options(self, options):
        if options == "all":
            options = self.get_all_result_options()
        if isinstance(options, list):
            options = "|".join(options)
        assert isinstance(options, str)
        return options

    def _get_onedal_params(
        self, options, L1=0.0, L2=0.0, intercept=True, dtype=np.float32
    ):
        options = self._get_result_options(options)
        return {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": self.algorithm,
            "result_option": options,
            "l1_coef": L1,
            "l2_coef": L2,
            "intercept": intercept,
        }

    def _compute(
            self,
            X,
            y,
            coef,
            options,
            l2_reg_strength=0.0,
            fit_intercept=True,
            queue=None):

        # in python interface intercept-coef is at the end of the array
        # in onedal interface coef array always has the size p + 1
        # coef[0] is considered as intercept-coef or ignored if
        # fit_intercept=False

        coef = coef.reshape(-1)
        if fit_intercept:
            coef = np.hstack([coef[-1], coef[:-1]])
        else:
            coef = np.hstack([0.0, coef])

        y = y.astype(np.int32)

        policy = self._get_policy(queue, X, y, coef)
        X_loc, y_loc, coef_loc = _convert_to_dataframe(policy, X, y, coef)
        X_loc, y_loc, coef_loc = _convert_to_supported(policy, X_loc, y_loc, coef_loc)

        ftype = X_loc.dtype

        X_table, coef_table, y_table = to_table(X_loc, coef_loc, y_loc)

        params = self._get_onedal_params(
            options, L2=l2_reg_strength, intercept=fit_intercept, dtype=ftype
        )

        result = self.func(policy, params, X_table, coef_table, y_table)

        options = self._get_result_options(options)
        options = options.split("|")

        res = {opt: getattr(result, opt) for opt in options}

        """To align with LinearModelClass we need to return
        the sum of losses over the data, while primitives from oneDAL
        return the mean value of loss.
        That is why we multiply the outputs by the size of the dataset
        """
        n = X.shape[0]
        return {k: from_table(v).ravel() * n for k, v in res.items()}


class LogisticLoss(BaseObjectiveFunction):
    def __init__(
        self, *, algorithm="by_default", fit_intercept=True, **kwargs
    ):
        self.fit_intercept = fit_intercept
        super().__init__(algorithm, _backend.objective_function.compute.logloss)

    def change_gradient_format(self, grad, coef, l2_reg_strength):
        """Change the gradient layout

        onedal compute_gradient function returns:
        [dL / dw_0, dL / dw_1, ..., dL / dw_p] if fit_intercept=True
        [0.0, dL / dw_1, ..., dL / dw_p] if fit_intercept=False
        gradient function from sklearn.linear_model._linear_loss returns:
        [dL / dw_1, ..., dL / dw_p, dL / dw_0] if fit_intercept=True
        [dL / dw_1, ..., dL / dw_p] if fit_intercept=False

        so to align with python interface format should be changed
        """

        if self.fit_intercept:
            grad = np.hstack([grad[1:] + coef[:-1] * l2_reg_strength, grad[0]])
        else:
            grad = grad[1:] + coef * l2_reg_strength
        return grad

    def change_hessian_format(self, hess, coef, l2_reg_strength):
        """Change the hessian layout
        onedal compute_hessian function returns matrix H of size (p+1)*(p+1)
        H_i,j = dL / (dw_i * d_w_j)
        if fit_intercept=False: H_0,i = H_0,i = 0.0

        hessian function from sklearn.linear_model._linear_loss returns:

        if fit_intercept=True
        matrix of size (p + 1) * (p + 1)
        H_i,j = dL / (dw_(i+1) * dw_(j+1)) for 0 <= i,j < p
        H_i,p = H_p,i = dL / (dw_(i+1) dw_0)

        if fit_intercept=False
        matrix of size p * p
        H_i,j = dL / (dw_(i+1) * dw_(j+1))

        so to align with python interface format should be changed
        """
        num_params = coef.shape[0]
        if not self.fit_intercept:
            num_params += 1
        hess = hess.reshape(num_params, num_params)
        if self.fit_intercept:
            hess_w = hess[1:, 1:] + np.diag([l2_reg_strength] * (num_params - 1))
            hess_with_bias_row = np.vstack([hess_w, hess[0, 1:]])
            hess_bias_col = np.hstack([hess[0, 1:], hess[0][0]]).reshape(-1, 1)
            hess = np.hstack([hess_with_bias_row, hess_bias_col])
        else:
            hess = hess[1:, 1:] + np.diag([l2_reg_strength] * (num_params - 1))
        return hess

    def __calculate_regularization(self, coef, l2_reg_strength):
        if self.fit_intercept:
            return 0.5 * (coef[:-1] ** 2).sum() * l2_reg_strength
        else:
            return 0.5 * (coef**2).sum() * l2_reg_strength

    def __validate_hyperparameters(self, sample_weight, n_threads, raw_prediction):
        if (sample_weight is not None) and \
                (not np.allclose(sample_weight,
                                 np.array([1]).astype(dtype=sample_weight.dtype))):
            raise Exception("sample_weigth parameter is not supported")
        if (n_threads != 1):
            raise Exception("multithreading is not supported")
        if (raw_prediction is not None):
            raise Exception("raw_prediction parameter is not supported")

    def loss(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
        queue=None
    ):
        self.__validate_hyperparameters(sample_weight, n_threads, raw_prediction)
        value = super()._compute(X, y, coef, "value", 0.0,
                                 self.fit_intercept, queue)["value"]
        if l2_reg_strength > 0:
            value += self.__calculate_regularization(coef, l2_reg_strength)
        return value

    def loss_gradient(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
        queue=None
    ):
        self.__validate_hyperparameters(sample_weight, n_threads, raw_prediction)
        res = super()._compute(
            X, y, coef, ["value", "gradient"], 0.0, self.fit_intercept, queue
        )
        value, grad = res["value"], res["gradient"]
        if l2_reg_strength > 0:
            value += self.__calculate_regularization(coef, l2_reg_strength)
        return (value, self.change_gradient_format(grad, coef, l2_reg_strength))

    def gradient(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
        queue=None
    ):
        self.__validate_hyperparameters(sample_weight, n_threads, raw_prediction)
        grad = super()._compute(X, y, coef, "gradient", 0.0, self.fit_intercept, queue)[
            "gradient"
        ]
        return self.change_gradient_format(grad, coef, l2_reg_strength)

    def gradient_hessian(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
        queue=None
    ):
        self.__validate_hyperparameters(sample_weight, n_threads, raw_prediction)
        res = super()._compute(
            X, y, coef, ["gradient", "hessian"], 0.0, self.fit_intercept, queue
        )
        grad = self.change_gradient_format(res["gradient"], coef, l2_reg_strength)
        hess = self.change_hessian_format(res["hessian"], coef, l2_reg_strength)
        flag = np.sum(res["hessian"] <= 0.0) * 2 >= res["hessian"].shape[0]
        return (grad, hess, flag)

    def gradient_hessian_product(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
        queue=None
    ):
        self.__validate_hyperparameters(sample_weight, n_threads, raw_prediction)
        res = super()._compute(
            X, y, coef, ["gradient", "hessian"], 0.0, self.fit_intercept, queue
        )
        grad = self.change_gradient_format(res["gradient"], coef, l2_reg_strength)
        hess = self.change_hessian_format(res["hessian"], coef, l2_reg_strength)

        def hessp(s):
            ret = np.empty_like(s)
            n_feat = hess.shape[0]
            ret[:n_feat] = hess @ s[:n_feat]
            return ret

        return grad, hessp