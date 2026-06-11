"""Combinatorial Purged Cross-Validation(López de Prado, AFML ch.12)。

時系列を N 個の連続ブロックに分け、k 個をテストに取る全組み合わせ
C(N,k) について「学習ブロックで最良のコンフィグを選び、テスト
ブロックで評価」を繰り返す。これにより:

- 各ブロックは C(N-1, k-1) 回テストに登場し、完全な OOS パスを
  φ = C(N-1, k-1) 本再構成できる → OOS シャープの「分布」が得られる
- 選ばれたコンフィグの OOS 順位から PBO(バックテスト過学習確率)を推定
- 全試行のシャープ分散から Deflated Sharpe Ratio を計算

本戦略はパラメータを日次で「学習」しない純因果パイプラインなので、
各コンフィグの全期間バックテストを 1 回走らせてから日次リターンを
ブロック単位でスライスすればリークはない(t 日のウェイトは t 日まで
の情報のみに依存)。purge/embargo はブロック境界の前後 embargo_days
を学習側から落とすことで、平滑化ポジションを通じた情報の滲みを断つ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from finbot.backtest.engine import run_backtest
from finbot.backtest.grid import DEFAULT_GRID
from finbot.backtest.metrics import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from finbot.config import BotConfig


@dataclass
class CPCVResult:
    path_sharpes: list[float]   # 再構成した OOS パスごとの年率シャープ(φ 本)
    pbo: float                  # Probability of Backtest Overfitting
    dsr: float                  # 全期間最良コンフィグの Deflated Sharpe Ratio
    psr: float                  # 同コンフィグの PSR(ベンチマーク SR=0)
    best_params: dict           # 全期間シャープ最良のコンフィグ
    combos: list[dict]          # 組み合わせごとの詳細


def _daily_sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    if len(x) < 3 or sd == 0 or np.isnan(sd):
        return 0.0
    return float(x.mean() / sd)


def run_cpcv(
    prices: pd.DataFrame,
    cfg: BotConfig,
    n_groups: int = 10,
    k_test: int = 2,
    grid: tuple[dict, ...] = DEFAULT_GRID,
    embargo_days: int = 5,
    carry: pd.DataFrame | None = None,
) -> CPCVResult:
    # 各コンフィグの全期間バックテスト(因果的なので 1 回で良い)
    active_rets = []
    for params in grid:
        res = run_backtest(prices, cfg.with_overrides(**params), carry=carry)
        active_rets.append(res.returns.iloc[cfg.warmup + 1 :].to_numpy())
    rets = np.column_stack(active_rets)  # T_active × n_cfg
    t_len, n_cfg = rets.shape

    if t_len < n_groups * 40:
        raise ValueError(
            f"データ不足: CPCV にはウォームアップ後 {n_groups * 40} 日以上必要"
        )

    bounds = np.linspace(0, t_len, n_groups + 1, dtype=int)
    n_paths = math.comb(n_groups - 1, k_test - 1)
    # path_slots[p][g] = パス p におけるブロック g の OOS リターン
    path_slots: list[list[np.ndarray | None]] = [
        [None] * n_groups for _ in range(n_paths)
    ]
    test_count = [0] * n_groups

    combos_out: list[dict] = []
    omegas: list[float] = []

    for combo in combinations(range(n_groups), k_test):
        train_mask = np.ones(t_len, dtype=bool)
        for g in combo:
            lo = max(0, bounds[g] - embargo_days)       # purge(テスト直前)
            hi = min(t_len, bounds[g + 1] + embargo_days)  # embargo(テスト直後)
            train_mask[lo:hi] = False
        test_idx = np.concatenate([np.arange(bounds[g], bounds[g + 1]) for g in combo])

        train_sr = [_daily_sharpe(rets[train_mask, j]) for j in range(n_cfg)]
        best_j = int(np.argmax(train_sr))
        oos_sr = [_daily_sharpe(rets[test_idx, j]) for j in range(n_cfg)]

        # 選択コンフィグの OOS 相対順位 ω ∈ (0,1)。ω <= 0.5 が「過学習」事象
        rank = sum(1 for s in oos_sr if oos_sr[best_j] >= s)
        omega = rank / (n_cfg + 1.0)
        omegas.append(omega)

        for g in combo:
            path_slots[test_count[g]][g] = rets[bounds[g] : bounds[g + 1], best_j]
            test_count[g] += 1

        combos_out.append(
            {
                "test_groups": combo,
                "params": grid[best_j],
                "train_sharpe_daily": train_sr[best_j],
                "oos_sharpe_ann": oos_sr[best_j] * np.sqrt(cfg.trading_days),
                "omega": omega,
            }
        )

    path_sharpes = []
    for p in range(n_paths):
        path = np.concatenate([blk for blk in path_slots[p] if blk is not None])
        path_sharpes.append(
            sharpe_ratio(pd.Series(path), cfg.rf_rate, cfg.trading_days)
        )

    pbo = float(np.mean([w <= 0.5 for w in omegas]))

    # DSR: 「グリッド全体から全期間シャープ最良を選んだ」という選択イベントを
    # 試行数 n_cfg と試行間分散でデフレートする
    full_daily = [_daily_sharpe(rets[:, j]) for j in range(n_cfg)]
    best_full = int(np.argmax(full_daily))
    best_series = pd.Series(rets[:, best_full])
    dsr = deflated_sharpe_ratio(best_series, full_daily)
    psr = probabilistic_sharpe_ratio(best_series, 0.0)

    return CPCVResult(
        path_sharpes=path_sharpes,
        pbo=pbo,
        dsr=dsr,
        psr=psr,
        best_params=grid[best_full],
        combos=combos_out,
    )
