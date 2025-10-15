"""
pso_new.py — 强化版 PSO（动态参数 + 动态速度 + DE 混合 + 约束）
- 修复：update_gbest 使用 pbest_x 对齐 gbest_y
- 新增：支持不等式约束 constraint_ueq
"""
from __future__ import annotations
import numpy as np

__all__ = ["PSO_new"]

class PSO_new:
    def __init__(self, func, n_dim, pop=40, max_iter=150,
                 lb=-1e5, ub=1e5,
                 # ① 动态参数调度
                 w_min=0.4, w_max=0.9,
                 c1=0.5, c2=0.5,
                 # ② 动态速度更新
                 beta_center=0.3,
                 sigma_pert=0.01,
                 anneal_pert=True,
                 # ③ DE 混合
                 de_interval=10,
                 de_rate=0.25,
                 de_F=0.5,
                 de_CR=0.9,
                 # ④ 约束
                 constraint_ueq=None,     # 不等式约束函数 (或列表)
                 penalty_coeff=1e6,       # 惩罚系数
                 verbose=False):
        """
        强化版 PSO
        func: 接受形如 (n_dim,) 的单个粒子，返回标量；也支持对 (pop,n_dim) 批量。
        constraint_ueq: 函数或函数列表，要求 g(x) <= 0 才可行
        """
        self.func = func
        self.n_dim = n_dim
        self.pop = pop
        self.max_iter = max_iter
        self.w_min, self.w_max = w_min, w_max
        self.cp, self.cg = c1, c2
        self.beta_center = beta_center
        self.sigma_pert0 = sigma_pert
        self.anneal_pert = anneal_pert
        self.de_interval = max(1, int(de_interval))
        self.de_rate = float(np.clip(de_rate, 0.0, 1.0))
        self.de_F = de_F
        self.de_CR = de_CR
        self.verbose = verbose

        # 约束
        if constraint_ueq is None:
            self.constraint_ueq = []
        elif isinstance(constraint_ueq, (list, tuple)):
            self.constraint_ueq = list(constraint_ueq)
        else:
            self.constraint_ueq = [constraint_ueq]
        self.penalty_coeff = penalty_coeff

        # 边界
        self.lb, self.ub = np.array(lb) * np.ones(n_dim), np.array(ub) * np.ones(n_dim)
        assert len(self.lb) == len(self.ub) == n_dim
        assert np.all(self.ub > self.lb)

        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(pop, n_dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(pop, n_dim))

        self.Y = self.cal_y()
        self.pbest_x = self.X.copy()
        self.pbest_y = np.full((pop, 1), np.inf)
        self.update_pbest()
        self.gbest_x = self.pbest_x[0, :].copy()
        self.gbest_y = np.inf
        self.update_gbest()

        self.gbest_y_hist = []

        # 记录
        self.record_mode = False
        self.record_value = {"X": [], "V": [], "Y": [], "w": [], "c1": [], "c2": [], "gbest_x": [], "gbest_y": []}

    # —— 约束惩罚 —— #
    def _apply_constraints(self, Xin, val):
        penalty = 0.0
        for g in self.constraint_ueq:
            cval = g(Xin)
            cval = np.asarray(cval)
            penalty += np.sum(np.maximum(0.0, cval))
        return val + self.penalty_coeff * penalty

    # —— 辅助：稳健评估单粒子 —— #
    def _eval_scalar(self, x1d):
        val = self.func(np.array(x1d))
        val = float(np.ravel(val)[0]) if np.ndim(val) > 0 else float(val)
        return self._apply_constraints(x1d, val)

    # ① 动态参数调度
    def _update_hyperparams(self, iter_num):
        ratio = iter_num / max(1, self.max_iter - 1)
        self.w = self.w_min + (self.w_max - self.w_min) * np.cos(0.5 * np.pi * ratio)
        z = (1.0 - np.log1p(ratio * (np.e - 1.0))) * (0.5 * np.pi)
        self.cp = 2.0 * np.sin(z)
        self.cg = 2.0 * np.cos(z)
        self.sigma_pert = self.sigma_pert0 * (1 - ratio) if self.anneal_pert else self.sigma_pert0

    # —— 稳健 cal_y —— #
    def cal_y(self):
        try:
            y = self.func(self.X)   # 批量
            y = np.asarray(y).reshape(-1, 1)
            if y.shape[0] != self.pop:
                raise ValueError
            if self.constraint_ueq:
                penalties = []
                for g in self.constraint_ueq:
                    cvals = g(self.X)
                    cvals = np.asarray(cvals).reshape(-1)
                    penalties.append(np.maximum(0.0, cvals))
                penalties = np.sum(penalties, axis=0).reshape(-1, 1)
                y = y + self.penalty_coeff * penalties
            return y
        except Exception:
            y = np.apply_along_axis(self._eval_scalar, 1, self.X)
            return y.reshape(-1, 1)

    # ② 动态速度更新
    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        V_std = (self.w * self.V
                 + self.cp * r1 * (self.pbest_x - self.X)
                 + self.cg * r2 * (self.gbest_x - self.X))

        center = np.mean(self.X, axis=0)
        nearest_idx = np.argmin(np.linalg.norm(self.X - center, axis=1))
        center_particle = self.X[nearest_idx]
        attract_center = self.beta_center * (center_particle - self.X)

        perturb = np.random.normal(0.0, self.sigma_pert, size=self.V.shape)

        self.V = V_std + attract_center + perturb

    def update_X(self):
        self.X += self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    def update_pbest(self):
        better_mask = self.Y < self.pbest_y
        self.pbest_x = np.where(better_mask, self.X, self.pbest_x)
        self.pbest_y = np.where(better_mask, self.Y, self.pbest_y)

    def update_gbest(self):
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.pbest_x[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    # ③ DE 混合
    def _de_hybrid(self):
        if self.de_rate <= 0.0:
            return
        m = max(1, int(self.pop * self.de_rate))
        idx_target = np.random.choice(self.pop, size=m, replace=False)
        X = self.X
        pop, dim = X.shape
        for i in idx_target:
            idxs = np.random.choice([j for j in range(pop) if j != i], size=3, replace=False)
            x1, x2, x3 = X[idxs]
            mutant = x1 + self.de_F * (x2 - x3)
            cross_points = np.random.rand(dim) < self.de_CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, X[i])
            trial = np.clip(trial, self.lb, self.ub)
            f_trial = self._eval_scalar(trial)
            if f_trial < self.Y[i, 0]:
                self.X[i] = trial
                self.Y[i, 0] = f_trial
                if f_trial < self.pbest_y[i, 0]:
                    self.pbest_y[i, 0] = f_trial
                    self.pbest_x[i] = trial
        self.update_gbest()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value["X"].append(self.X.copy())
        self.record_value["V"].append(self.V.copy())
        self.record_value["Y"].append(self.Y.copy())
        self.record_value["w"].append(self.w)
        self.record_value["c1"].append(self.cp)
        self.record_value["c2"].append(self.cg)
        self.record_value["gbest_x"].append(self.gbest_x.copy())
        self.record_value["gbest_y"].append(float(self.gbest_y))

    def run(self, max_iter=None, precision=None, N=20, stop_mode="fitness"):
        self.max_iter = max_iter or self.max_iter
        consec = 0
        for iter_num in range(self.max_iter):
            self._update_hyperparams(iter_num)
            self.update_V()
            self.update_X()
            self.Y = self.cal_y()
            self.update_pbest()
            self.update_gbest()

            if self.de_interval > 0 and ((iter_num + 1) % self.de_interval == 0):
                self._de_hybrid()

            self.recorder()

            if self.verbose:
                print(f"Iter: {iter_num:4d} | w={self.w:.4f} c1={self.cp:.4f} c2={self.cg:.4f} "
                      f"| Best fit: {float(self.gbest_y):.6g} at {self.gbest_x}")

            self.gbest_y_hist.append(self.gbest_y)

            if precision is not None:
                conds = []
                if stop_mode in ("fitness", "both"):
                    spread_y = float(np.max(self.Y) - np.min(self.Y))
                    conds.append(spread_y < precision)
                if stop_mode in ("position", "both"):
                    spread_x = np.max(self.X, axis=0) - np.min(self.X, axis=0)
                    spread_x_inf = float(np.max(spread_x))
                    conds.append(spread_x_inf < precision)
                satisfied = all(conds) if conds else False
                if satisfied:
                    consec += 1
                    if consec > N:
                        if self.verbose:
                            msg = f"Converged at iter {iter_num}"
                            if 'spread_y' in locals(): msg += f", fitness spread={spread_y:.3e}"
                            if 'spread_x_inf' in locals(): msg += f", position spread={spread_x_inf:.3e}"
                            print(msg)
                        break
                else:
                    consec = 0

        return self.gbest_x, self.gbest_y
