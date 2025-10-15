# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:12:20 2023

@author: adelrioc

说明：
- 修复了 pySOT 的 opt_prob 类型要求：现在通过 PySOTProblem 包装原始 f，
  使其实现 OptimizationProblem 接口，避免 "opt_prob must implement OptimizationProblem" 报错。
- 统一使用 problem.eval(...) 评估目标函数。
- has_x0=True 时若原始 f 无 x0，则自动使用边界中心作为起点，确保不报错。
- 假设你的 poap SerialController 已按注释修改为在调用时使用 self.objective.eval(...)。
"""

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import DYCORSStrategy, SRBFStrategy, SOPStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import OptimizationProblem
from poap.controller import SerialController
import numpy as np


class PySOTProblem(OptimizationProblem):
    """
    将用户自定义的 f 包装为 pySOT 可识别的问题对象。
    要求：
      - f 需要能通过 f.fun_test(x) 计算目标（若没有 fun_test，则尝试 f(x) 可调用）
      - 若提供 f.x0 作为初始点，会被使用（否则可选用边界中心点）
      
    """
    

    def __init__(self, f, dim: int, bounds: np.ndarray):
        """
        Parameters
        ----------
        f : object
            用户的目标函数对象。优先调用 f.fun_test(x)，否则尝试 f(x)。
            x 期望为 shape (n_dim,) 或 (1, n_dim) 的数组。
        dim : int
            维度
        bounds : (dim, 2) ndarray
            下界、上界
        """
        assert isinstance(bounds, np.ndarray) and bounds.shape == (dim, 2), \
            "bounds 必须是 shape=(dim,2) 的 numpy 数组"
        self._f = f
        self.dim = dim
        self.lb = bounds[:, 0].astype(float)
        self.ub = bounds[:, 1].astype(float)
        self.int_var = np.array([], dtype=int)          # 全连续变量
        self.cont_var = np.arange(0, dim, dtype=int)

        # 可选：起始点
        self.x0 = None
        if hasattr(f, "x0") and f.x0 is not None:
            x0 = np.asarray(f.x0)
            # 可能是列表/单点/多点，这里取第一个点并拍平成 1D
            if x0.ndim == 2:
                # 如果是多行候选点，取第一行
                if x0.shape[1] == self.dim:
                    self.x0 = x0[0]
            elif x0.ndim == 1:
                if x0.size == self.dim:
                    self.x0 = x0
            # 如果不符合预期，就保持 None，不报错

    def _call_user_fun(self, x: np.ndarray) -> float:
        """调用用户提供的目标函数"""
        # 优先 f.fun_test(x)
        if hasattr(self._f, "fun_test") and callable(self._f.fun_test):
            return float(self._f.fun_test(x))
        # 其次尝试 f(x)
        if callable(self._f):
            return float(self._f(x))
        raise TypeError("目标函数对象既没有 fun_test 可用，也不可直接调用。")

    def eval(self, x):
        """
        Parameters
        ----------
        x : array-like
            shape (dim,) 或 (1, dim) 或 (n, dim)；这里只处理单点评估。

        Returns
        -------
        float
        """
        xx = np.asarray(x, dtype=float)
        if xx.ndim == 2:
            if xx.shape[0] != 1:
                # 仅支持单点评估；传多点将取第一行
                xx = xx[0]
            else:
                xx = xx.reshape(-1)
        elif xx.ndim == 1:
            pass
        else:
            xx = xx.reshape(-1)

        if xx.size != self.dim:
            raise ValueError(f"x 维度不匹配：期望 {self.dim}，得到 {xx.size}")

        # 保证在边界内（可根据需要裁剪）
        xx = np.minimum(np.maximum(xx, self.lb), self.ub)
        return self._call_user_fun(xx)
    
    def __call__(self, x):
        """让对象可调用，直接转到 eval()"""
        return self.eval(x)


# -------------------------
# 工具：随机搜索（用于示例/起点）
# -------------------------
def Random_searchDYCORS(problem: PySOTProblem, n_p: int, bounds_rs: np.ndarray, iter_rs: int):
    """
    简单随机搜索，返回 (f_best, x_best)
    """
    localx = np.zeros((n_p, iter_rs))   # 采样点
    localval = np.zeros((iter_rs))      # 对应函数值

    bounds_range = bounds_rs[:, 1] - bounds_rs[:, 0]
    bounds_bias = bounds_rs[:, 0]

    for sample_i in range(iter_rs):
        x_trial = np.random.uniform(0, 1, n_p) * bounds_range + bounds_bias
        localx[:, sample_i] = x_trial
        localval[sample_i] = problem.eval(x_trial)

    minindex = np.argmin(localval)
    f_b = localval[minindex]
    x_b = localx[:, minindex]
    return f_b, x_b


# -------------------------
# DYCORS
# -------------------------
def DYCORS(f, x_dim, bounds, iter_tot, has_x0=False):
    """
    DYCORS 优化
    """
    bounds = np.asarray(bounds, dtype=float)
    problem = PySOTProblem(f, x_dim, bounds)

    rbf = RBFInterpolant(
        dim=x_dim, lb=bounds[:, 0], ub=bounds[:, 1],
        kernel=CubicKernel(), tail=LinearTail(x_dim)
    )

    # controller 期望 objective 带有 .eval（你已在 SerialController 源码做了改动）
    controller = SerialController(objective=problem)

    if has_x0:
        # 起点：优先来自 f.x0，否则采用边界中心
        if problem.x0 is None:
            x_start = np.array([[-3.5, 4.0]])
        else:
            x_start = problem.x0.reshape(1, -1)

        f_start = np.array(problem.eval(x_start)).reshape((1, 1))
        iter_ = max(int(iter_tot) - 1, 1)

        slhd_minus1 = SymmetricLatinHypercube(dim=x_dim, num_pts=2 * (x_dim + 1) - 1)

        controller.strategy = DYCORSStrategy(
            max_evals=iter_,
            opt_prob=problem,
            asynchronous=False,
            exp_design=slhd_minus1,
            extra_points=np.array(x_start),
            extra_vals=np.array(f_start),
            surrogate=rbf,
            num_cand=100 * x_dim,
            batch_size=1
        )
        result = controller.run()
        return result, None, None, None

    else:
        slhd = SymmetricLatinHypercube(dim=x_dim, num_pts=2 * (x_dim + 1))

        controller.strategy = DYCORSStrategy(
            max_evals=int(iter_tot),
            opt_prob=problem,
            asynchronous=False,
            exp_design=slhd,
            surrogate=rbf,
            num_cand=100 * x_dim,
            batch_size=1
        )
        result = controller.run()
        return result, None, None, None


# -------------------------
# SRBF
# -------------------------
def SRBF(f, x_dim, bounds, iter_tot, has_x0=False):
    """
    SRBF 优化
    """
    bounds = np.asarray(bounds, dtype=float)
    problem = PySOTProblem(f, x_dim, bounds)

    rbf = RBFInterpolant(
        dim=x_dim, lb=bounds[:, 0], ub=bounds[:, 1],
        kernel=CubicKernel(), tail=LinearTail(x_dim)
    )

    controller = SerialController(objective=problem)

    if has_x0:
        if problem.x0 is None:
            x_start = np.array([[-3.5, 4.0]])
        else:
            x_start = problem.x0.reshape(1, -1)

        f_start = np.array(problem.eval(x_start)).reshape((1, 1))
        iter_ = max(int(iter_tot) - 1, 1)

        slhd_minus1 = SymmetricLatinHypercube(dim=x_dim, num_pts=2 * (x_dim + 1) - 1)

        controller.strategy = SRBFStrategy(
            max_evals=iter_,
            opt_prob=problem,
            asynchronous=False,
            exp_design=slhd_minus1,
            extra_points=np.array(x_start),
            extra_vals=np.array(f_start),
            surrogate=rbf,
            num_cand=100 * x_dim,
            batch_size=1
        )
        result = controller.run()
        return result, None, None, None

    else:
        slhd = SymmetricLatinHypercube(dim=x_dim, num_pts=2 * (x_dim + 1))

        controller.strategy = SRBFStrategy(
            max_evals=int(iter_tot),
            opt_prob=problem,
            asynchronous=False,
            exp_design=slhd,
            surrogate=rbf,
            num_cand=100 * x_dim,
            batch_size=1
        )
        result = controller.run()
        return result, None, None, None


# -------------------------
# SOPStrategy
# -------------------------
def opt_SOP(f, x_dim, bounds, iter_tot, has_x0=False):
    """
    SOPStrategy 优化
    """
    bounds = np.asarray(bounds, dtype=float)
    problem = PySOTProblem(f, x_dim, bounds)

    rbf = RBFInterpolant(
        dim=x_dim, lb=bounds[:, 0], ub=bounds[:, 1],
        kernel=CubicKernel(), tail=LinearTail(x_dim)
    )

    controller = SerialController(objective=problem)

    if has_x0:
        if problem.x0 is None:
            x_start = ((bounds[:, 0] + bounds[:, 1]) / 2.0).reshape(1, -1)
        else:
            x_start = problem.x0.reshape(1, -1)

        f_start = np.array(problem.eval(x_start)).reshape((1, 1))
        iter_ = max(int(iter_tot) - 1, 1)

        slhd_minus1 = SymmetricLatinHypercube(dim=x_dim, num_pts=2 * (x_dim + 1) - 1)

        controller.strategy = SOPStrategy(
            max_evals=iter_,
            opt_prob=problem,
            asynchronous=False,
            exp_design=slhd_minus1,
            extra_points=np.array(x_start),
            extra_vals=np.array(f_start),
            surrogate=rbf,
            num_cand=100 * x_dim,
            batch_size=1
        )
        result = controller.run()
        return result, None, None, None

    else:
        slhd = SymmetricLatinHypercube(dim=x_dim, num_pts=2 * (x_dim + 1))

        controller.strategy = SOPStrategy(
            max_evals=int(iter_tot),
            opt_prob=problem,
            asynchronous=False,
            exp_design=slhd,
            surrogate=rbf,
            num_cand=100 * x_dim,
            batch_size=1
        )
        result = controller.run()
        return result, None, None, None
    
# ---------- 你的 NDCT 基础方法 ----------
def ndctN_basis(X, K_list):
    X = np.atleast_2d(X)
    n, d = X.shape
    grids = [np.arange(K+1) for K in K_list]
    multi_idx = list(product(*grids))
    A = np.empty((n, len(multi_idx)))
    for k, idx in enumerate(multi_idx):
        Phi = np.ones(n)
        for j, ij in enumerate(idx):
            Phi *= np.cos(np.pi * ij * X[:, j])
        A[:, k] = Phi
    return A, multi_idx

def ndctN_fit(X, y, K_list, lam=0.0):
    A, multi_idx = ndctN_basis(X, K_list)
    if lam > 0:
        ATA = A.T @ A
        ATy = A.T @ y
        c = np.linalg.solve(ATA + lam*np.eye(ATA.shape[0]), ATy)
    else:
        c = np.linalg.lstsq(A, y, rcond=None)[0]
    return c, multi_idx

def ndctN_predict(X, coeffs, multi_idx):
    X = np.atleast_2d(X)
    Z = np.zeros(X.shape[0])
    for coef, idx in zip(coeffs, multi_idx):
        Phi = np.ones(X.shape[0])
        for j, ij in enumerate(idx):
            Phi *= np.cos(np.pi * ij * X[:, j])
        Z += coef * Phi
    return Z

from pySOT.surrogate import Surrogate
import numpy as np
from itertools import product

class NDCTSurrogate(Surrogate):
    def __init__(self, dim, lb, ub, K_list=None, lam=1e-6):
        super().__init__(dim, lb, ub)
        self.K_list = K_list if K_list is not None else [5]*dim
        self.lam = lam
        self.X = []
        self.y = []
        self.coeffs = None
        self.multi_idx = None

    def fit(self):
        if len(self.X) == 0:
            return
        X = np.array(self.X)
        y = np.array(self.y)
        X_scaled = (X - self.lb) / (self.ub - self.lb)
        self.coeffs, self.multi_idx = ndctN_fit(X_scaled, y, self.K_list, self.lam)

    def predict(self, X):
        X = np.atleast_2d(X)
        if self.coeffs is None or self.multi_idx is None:
            # 还没 fit 时返回零预测
            return np.zeros((X.shape[0], 1))
        X_scaled = (X - self.lb) / (self.ub - self.lb)
        z = ndctN_predict(X_scaled, self.coeffs, self.multi_idx)
        return z.reshape(-1, 1)   # 确保 (n,1)

    def predict_deriv(self, X):
        # 暂时不实现导数，返回零矩阵
        X = np.atleast_2d(X)
        return np.zeros((X.shape[0], self.dim))

    def evals(self):
        return len(self.y)


def pso_optimize_with_surrogate(surrogate, bounds, n_particles=20, iters=100, w=1, c1=2, c2=2):
    dim = surrogate.dim
    lb, ub = bounds[:, 0], bounds[:, 1]

    # 初始化粒子群
    X = np.random.rand(n_particles, dim) * (ub - lb) + lb
    V = np.zeros_like(X)

    # 固定第一个粒子在 (-3.5, 4.0) （仅当 dim >= 2）
    if dim >= 2:
        X[0] = np.array([-3.5, 4.0])

    # surrogate 预测
    y = surrogate.predict(X).ravel()
    pbest = X.copy()
    pbest_val = y.copy()
    gbest = X[np.argmin(y)]
    gbest_val = np.min(y)

    # PSO 迭代
    for _ in range(iters):
        r1, r2 = np.random.rand(n_particles, dim), np.random.rand(n_particles, dim)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lb, ub)

        y = surrogate.predict(X).ravel()

        # 更新个体最优
        update = y < pbest_val
        pbest[update] = X[update]
        pbest_val[update] = y[update]

        # 更新全局最优
        if np.min(y) < gbest_val:
            gbest = X[np.argmin(y)]
            gbest_val = np.min(y)

    return gbest, gbest_val


# ---------- 主函数 ----------
def PSO_NDCT(f, x_dim, bounds, iter_tot=100, init_samples=20, K_list=None, lam=1e-6, has_x0=False):
    """
    使用 NDCT surrogate + PSO 进行优化
    f: 目标函数 (可为普通函数 or Test_function 对象)
    x_dim: 维度
    bounds: [(lb1, ub1), (lb2, ub2), ...]
    iter_tot: PSO 迭代次数
    init_samples: 初始真实采样点数量
    """
    bounds = np.asarray(bounds, dtype=float)

    # surrogate 初始化
    surrogate = NDCTSurrogate(
        dim=x_dim,
        lb=bounds[:, 0],
        ub=bounds[:, 1],
        K_list=[6,6],
        lam=lam
    )

    # 初始采样
    X_init = np.random.rand(init_samples, x_dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    # 固定第一个初始点 (-3.5, 4.0)（仅当 dim >= 2）
    if x_dim >= 2:
        X_init[0] = np.array([-3.5, 4.0])

    # 真实函数评估
    if hasattr(f, "fun_test"):
        y_init = np.array([f.fun_test(x) for x in X_init]).reshape(-1, 1)
    else:
        y_init = np.array([f(x) for x in X_init]).reshape(-1, 1)

    surrogate.X = X_init
    surrogate.y = y_init
    surrogate.fit()

    # surrogate 上运行 PSO
    gbest, gbest_val = pso_optimize_with_surrogate(
        surrogate, bounds, n_particles=20, iters=iter_tot
    )

    # 真实函数评估 gbest
    if hasattr(f, "fun_test"):
        f_real = f.fun_test(gbest)
    else:
        f_real = f(gbest)

    # ---- 保持 ML4CE 接口 ----
    a = gbest           # 最优解向量
    b = f_real          # 最优函数值
    team_names = "PSO_NDCT"
    cids = None
    return a, b, team_names, cids

#------------------------------------------
def new_pso_optimize_with_surrogate(
    surrogate,
    bounds,
    n_particles=30,
    iters=100,
    # —— 动态参数（与 PSO_new 对齐思路）——
    w_min=0.4, w_max=0.9,
    c1=1.5, c2=1.5,
    # —— 动态速度增强 —— 
    beta_center=0.3,
    sigma_pert0=0.02,
    anneal_pert=True,
    # —— DE 混合 —— 
    de_interval=10,
    de_rate=0.25,
    de_F=0.5,
    de_CR=0.9,
    # —— DYCORS 风格稀疏更新 —— 
    dycors_min=0.1,
    # —— 速度与边界 —— 
    vmax_scale=0.5,   # vmax = scale * (ub-lb)
    # —— 探索项（默认关闭；NDCT 无 sigma 时不生效）——
    gamma=0.0,
    # —— 约束（默认无）——
    constraint_ueq=None,
    penalty_coeff=1e6,
    # —— 起点（可选）——
    x0=None,
):
    dim = surrogate.dim
    lb, ub = bounds[:, 0].astype(float), bounds[:, 1].astype(float)

    # 初始化群体
    X = np.random.uniform(lb, ub, size=(n_particles, dim))
    if x0 is not None and len(x0) == dim:
        X[0] = np.clip(np.asarray(x0, dtype=float), lb, ub)
    elif dim >= 2:
        # 你的习惯起点：(-3.5, 4.0)
        X[0] = np.clip(np.array([-3.5, 4.0] + [0.0]*(dim-2), dtype=float), lb, ub)

    v_high = (ub - lb)
    vmax = vmax_scale * v_high
    V = np.random.uniform(-v_high, v_high, size=(n_particles, dim))

    # 约束：归一化为函数列表
    if constraint_ueq is None:
        constraint_list = []
    elif isinstance(constraint_ueq, (list, tuple)):
        constraint_list = list(constraint_ueq)
    else:
        constraint_list = [constraint_ueq]

    # 评估 surrogate 目标 + 罚项（批量）
    def penalty_vals(Xin):
        if not constraint_list:
            return np.zeros((Xin.shape[0], 1))
        pen = 0.0
        for g in constraint_list:
            cvals = np.asarray(g(Xin)).reshape(Xin.shape[0], -1)
            pen += np.maximum(0.0, cvals).sum(axis=1, keepdims=True)
        return penalty_coeff * pen

    y = surrogate.predict(X).reshape(-1, 1)
    y = y + penalty_vals(X)

    # pbest/gbest
    pbest_X = X.copy()
    pbest_y = y.copy()
    g_idx = int(np.argmin(pbest_y))
    gbest = pbest_X[g_idx].copy()
    gbest_val = float(pbest_y[g_idx, 0])

    # 动态参数状态
    w = w_max
    cp, cg = c1, c2
    sigma_pert = sigma_pert0

    for t in range(iters):
        # —— 动态参数：与 PSO_new 近似的余弦/对数调度 —— #
        ratio = t / max(1, iters - 1)
        w = w_min + (w_max - w_min) * np.cos(0.5 * np.pi * ratio)
        z = (1.0 - np.log1p(ratio * (np.e - 1.0))) * (0.5 * np.pi)
        cp = 2.0 * np.sin(z)
        cg = 2.0 * np.cos(z)
        sigma_pert = (sigma_pert0 * (1 - ratio)) if anneal_pert else sigma_pert0

        # —— 标准速度项 —— #
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        V_std = (w * V
                 + cp * r1 * (pbest_X - X)
                 + cg * r2 * (gbest - X))

        # —— 吸引“群心附近粒子” —— #
        center = np.mean(X, axis=0)
        nearest_idx = np.argmin(np.linalg.norm(X - center, axis=1))
        center_particle = X[nearest_idx]
        attract_center = beta_center * (center_particle - X)

        # —— 退火高斯扰动 —— #
        perturb = np.random.normal(0.0, sigma_pert, size=V.shape)

        V = V_std + attract_center + perturb
        # 速度裁剪
        V = np.clip(V, -vmax, vmax)

        # —— DYCORS：随机维度更新掩码 —— #
        perturb_prob = max(dycors_min, 1.0 - ratio)
        mask = (np.random.rand(*X.shape) < perturb_prob)

        # 位置更新 + 边界裁剪
        X_new = X + V * mask
        X_new = np.clip(X_new, lb, ub)

        # surrogate 预测
        y_new = surrogate.predict(X_new).reshape(-1, 1)
        # 探索项：仅当提供 sigma 时启用
        if gamma > 0.0 and hasattr(surrogate, "predict_uncertainty"):
            try:
                sigma = surrogate.predict_uncertainty(X_new).reshape(-1, 1)
                y_new = y_new - gamma * sigma
            except Exception:
                pass

        # 约束罚项
        y_new = y_new + penalty_vals(X_new)

        # 更新个体最优
        improve = (y_new < pbest_y)
        pbest_X = np.where(improve, X_new, pbest_X)
        pbest_y = np.where(improve, y_new, pbest_y)

        # 更新全局最优（以 pbest 为准，避免错配）
        g_idx = int(np.argmin(pbest_y))
        if float(pbest_y[g_idx, 0]) < gbest_val:
            gbest = pbest_X[g_idx].copy()
            gbest_val = float(pbest_y[g_idx, 0])

        # 替换当前状态
        X, y = X_new, y_new

        # —— DE 混合（可选） —— #
        if (de_interval > 0) and ((t + 1) % de_interval == 0) and (de_rate > 0.0):
            m = max(1, int(n_particles * de_rate))
            idx_target = np.random.choice(n_particles, size=m, replace=False)
            for i in idx_target:
                idxs = np.random.choice([j for j in range(n_particles) if j != i], size=3, replace=False)
                x1, x2, x3 = X[idxs]
                mutant = x1 + de_F * (x2 - x3)
                cross = np.random.rand(dim) < de_CR
                if not np.any(cross):
                    cross[np.random.randint(0, dim)] = True
                trial = np.where(cross, mutant, X[i])
                trial = np.clip(trial, lb, ub)

                # 评估 trial
                f_trial = surrogate.predict(trial.reshape(1, -1)).reshape(1, 1)
                f_trial = f_trial + penalty_vals(trial.reshape(1, -1))
                if (gamma > 0.0) and hasattr(surrogate, "predict_uncertainty"):
                    try:
                        sig_t = surrogate.predict_uncertainty(trial.reshape(1, -1)).reshape(1, 1)
                        f_trial = f_trial - gamma * sig_t
                    except Exception:
                        pass

                if float(f_trial[0, 0]) < float(y[i, 0]):
                    X[i] = trial
                    y[i, 0] = float(f_trial[0, 0])
                    # 维护 pbest
                    if y[i, 0] < pbest_y[i, 0]:
                        pbest_y[i, 0] = y[i, 0]
                        pbest_X[i] = trial
            # 用 pbest 更新 gbest
            g_idx = int(np.argmin(pbest_y))
            if float(pbest_y[g_idx, 0]) < gbest_val:
                gbest = pbest_X[g_idx].copy()
                gbest_val = float(pbest_y[g_idx, 0])

    return gbest, gbest_val



def new_PSO_NDCT(f, x_dim, bounds, iter_tot=100, init_samples=20, K_list=None, lam=1e-6, has_x0=False):
    """
    使用 NDCT surrogate + 改进型 PSO 进行优化
    f: 目标函数 (可为普通函数 or Test_function 对象)
    x_dim: 维度
    bounds: [(lb1, ub1), (lb2, ub2), ...]
    iter_tot: PSO 迭代次数
    init_samples: 初始真实采样点数量
    """
    bounds = np.asarray(bounds, dtype=float)

    # surrogate 初始化
    surrogate = NDCTSurrogate(
        dim=x_dim,
        lb=bounds[:, 0],
        ub=bounds[:, 1],
        K_list=[6,6],
        lam=lam
    )

    # 初始采样
    X_init = np.random.rand(init_samples, x_dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    # 固定第一个初始点 (-3.5, 4.0)（仅当 dim >= 2）
    if x_dim >= 2:
        X_init[0] = np.array([-3.5, 4.0])

    # 真实函数评估
    if hasattr(f, "fun_test"):
        y_init = np.array([f.fun_test(x) for x in X_init]).reshape(-1, 1)
    else:
        y_init = np.array([f(x) for x in X_init]).reshape(-1, 1)

    surrogate.X = X_init
    surrogate.y = y_init
    surrogate.fit()

    # surrogate 上运行 改进版 PSO
    gbest, gbest_val = new_pso_optimize_with_surrogate(
        surrogate,
        bounds,
        n_particles=10,
        iters=iter_tot,
        w_max=2,
        w_min=0.4,
        c1=3,
        c2=2,
        gamma=0
    )

    # 真实函数评估 gbest
    if hasattr(f, "fun_test"):
        f_real = f.fun_test(gbest)
    else:
        f_real = f(gbest)

    # ---- 保持 ML4CE 接口 ----
    a = gbest           # 最优解向量
    b = f_real          # 最优函数值
    team_names = "PSO_NDCT"
    cids = None
    return a, b, team_names, cids