"""相関構造・ボラティリティレジーム付きの合成市場データ。

実データAPIに接続できない環境での開発・検証用。資産クラスごとに
現実的な水準のドリフト/ボラ/相関を持たせ、平穏期と荒れ相場を
マルコフ連鎖で切り替えることでトレンドとショックの両方を再現する。
シード固定で完全に再現可能。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finbot.config import DEFAULT_UNIVERSE
from finbot.data.base import DataSource

# 資産クラスごとの (年率ドリフト, 年率ボラ)
_ASSET_PARAMS: dict[str, tuple[float, float]] = {
    "EQ_US": (0.07, 0.16),
    "EQ_INTL": (0.055, 0.17),
    "EQ_EM": (0.06, 0.22),
    "BOND_GOV": (0.025, 0.07),
    "BOND_CORP": (0.035, 0.08),
    "GOLD": (0.04, 0.15),
    "COMMOD": (0.03, 0.18),
    "REIT": (0.06, 0.19),
}

# ざっくりした資産クラス間相関(対称行列の下三角を定義)
_CORR = np.array(
    [
        # EQ_US EQ_IN EQ_EM B_GOV B_CRP GOLD  CMD   REIT
        [1.00, 0.85, 0.75, -0.20, 0.20, 0.05, 0.30, 0.70],
        [0.85, 1.00, 0.80, -0.15, 0.25, 0.10, 0.35, 0.65],
        [0.75, 0.80, 1.00, -0.10, 0.30, 0.15, 0.45, 0.60],
        [-0.20, -0.15, -0.10, 1.00, 0.60, 0.25, -0.10, 0.10],
        [0.20, 0.25, 0.30, 0.60, 1.00, 0.15, 0.10, 0.30],
        [0.05, 0.10, 0.15, 0.25, 0.15, 1.00, 0.35, 0.10],
        [0.30, 0.35, 0.45, -0.10, 0.10, 0.35, 1.00, 0.25],
        [0.70, 0.65, 0.60, 0.10, 0.30, 0.10, 0.25, 1.00],
    ]
)


class SyntheticSource(DataSource):
    def __init__(
        self,
        universe: tuple[str, ...] = DEFAULT_UNIVERSE,
        days: int = 2520,  # 約10年
        seed: int = 42,
        start: str = "2016-01-04",
    ):
        unknown = set(universe) - set(_ASSET_PARAMS)
        if unknown:
            raise ValueError(f"未定義の資産: {unknown}")
        self.universe = tuple(universe)
        self.days = days
        self.seed = seed
        self.start = start

        self._prices: pd.DataFrame | None = None
        self._carry: pd.DataFrame | None = None

    def _generate(self) -> None:
        rng = np.random.default_rng(self.seed)
        names = list(self.universe)
        idx = [list(_ASSET_PARAMS).index(n) for n in names]
        n = len(names)

        mu = np.array([_ASSET_PARAMS[a][0] for a in names]) / 252.0
        sigma = np.array([_ASSET_PARAMS[a][1] for a in names]) / np.sqrt(252.0)
        corr = _CORR[np.ix_(idx, idx)]
        chol = np.linalg.cholesky(corr)

        # 2状態マルコフ連鎖: 平穏(vol×1)と荒れ相場(vol×2.2, ドリフト反転気味)
        p_calm_to_storm, p_storm_to_calm = 0.02, 0.10
        regime = np.zeros(self.days, dtype=int)
        state = 0
        for t in range(self.days):
            regime[t] = state
            p = p_calm_to_storm if state == 0 else 1 - p_storm_to_calm
            state = 1 if rng.random() < p else 0

        z = rng.standard_normal((self.days, n)) @ chol.T
        vol_mult = np.where(regime == 1, 2.2, 1.0)[:, None]
        drift_mult = np.where(regime == 1, -1.5, 1.0)[:, None]
        rets = mu * drift_mult + sigma * vol_mult * z

        dates = pd.bdate_range(self.start, periods=self.days)
        self._prices = self.validate(
            pd.DataFrame(
                100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=names
            )
        )

        # 観測可能な年率キャリー: 真の期待リターンを部分的に反映するノイズ付き
        # 観測値(AR(1))。実データでの配当利回り・ロールイールドに相当する。
        true_carry = mu * drift_mult * 252.0
        ar = np.zeros((self.days, n))
        eps = rng.normal(0.0, 0.01, (self.days, n))
        phi = 0.97
        for t in range(1, self.days):
            ar[t] = phi * ar[t - 1] + eps[t]
        self._carry = pd.DataFrame(
            0.5 * true_carry + ar, index=dates, columns=names
        )

    def load(self) -> pd.DataFrame:
        if self._prices is None:
            self._generate()
        return self._prices

    def load_carry(self) -> pd.DataFrame:
        if self._carry is None:
            self._generate()
        return self._carry
