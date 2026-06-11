"""検証用パラメータグリッド。

グリッドは意図的に小さい: 設計自由度の大半は文献の標準値に固定し
(docs/DESIGN_V2.md §7)、ここで「探索」するのはスタイル配分と
目標ボラの粗い感度だけ。グリッドを増やすと DSR のデフレーションで
罰される(試行数 N がそのまま帰無分布を押し上げる)。
"""

from __future__ import annotations

DEFAULT_GRID: tuple[dict, ...] = (
    {},                                          # 基準: トレンド60/キャリー40
    {"trend_weight": 1.0, "carry_weight": 0.0},  # トレンド単独
    {"trend_weight": 0.5, "carry_weight": 0.5},
    {"target_vol": 0.08},
    {"target_vol": 0.12},
)
