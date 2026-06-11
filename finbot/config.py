"""ボット全体の設定(設計方針 v2: docs/DESIGN_V2.md)。

パラメータは「最良値の探索」ではなく文献の標準値に固定してある
(Baz et al. 2015 のトレンド構成、Carver のフォーキャスト規約、
Barra 流のボラ/相関分離推定)。自由度を増やすと多重検定の
デフレーション(DSR)で罰されることに注意。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

DEFAULT_UNIVERSE: tuple[str, ...] = (
    "EQ_US",      # 米国株式
    "EQ_INTL",    # 先進国株式(米国除く)
    "EQ_EM",      # 新興国株式
    "BOND_GOV",   # 国債(長期)
    "BOND_CORP",  # 社債
    "GOLD",       # 金
    "COMMOD",     # コモディティ
    "REIT",       # 不動産
)


@dataclass(frozen=True)
class BotConfig:
    # ユニバース
    universe: Sequence[str] = field(default_factory=lambda: DEFAULT_UNIVERSE)

    # --- シグナル: トレンド(Baz et al. 2015 EWMAC) ---
    # タイムスケールペア (S, L)。HL = log(0.5)/log(1-1/n) で EWMA の halflife に変換。
    trend_pairs: Sequence[tuple[int, int]] = ((8, 24), (16, 48), (32, 96))
    trend_price_vol_window: int = 63    # 1段目正規化: 価格の rolling std 窓
    trend_signal_vol_window: int = 252  # 2段目正規化: シグナルの rolling std 窓

    # --- シグナル: キャリー ---
    carry_smooth_span: int = 90   # キャリーの EWMA 平滑化(ターンオーバー削減)

    # --- フォーキャスト統合層(Carver 規約) ---
    forecast_abs_target: float = 10.0   # E|forecast| の目標
    forecast_cap: float = 20.0          # フォーキャストの上限(±)
    forecast_scale_burn: int = 63       # スケーラ推定の最低観測日数(因果的 expanding)
    trend_weight: float = 0.6           # トレンド:キャリー = 60:40
    carry_weight: float = 0.4
    fdm: float = 1.15                   # フォーキャスト分散乗数(低相関 2 ルールの標準値)

    # --- リスクモデル(ボラと相関の分離推定) ---
    vol_halflife: int = 20          # 短期 EWMA ボラの halflife(営業日)
    vol_blend: float = 0.7          # 短期ボラの重み(残りは長期アンカー)
    vol_long_min_periods: int = 252  # 長期アンカーの最低観測日数
    corr_halflife: int = 200        # 相関 EWMA の halflife(ボラより遅く推定)
    corr_shrinkage: float = 0.2     # 定相関ターゲットへの線形シュリンク係数
    target_vol: float = 0.10        # ポートフォリオ目標ボラ(年率)
    max_leverage: float = 2.0       # グロスレバレッジ上限
    max_weight: float = 0.30        # 1資産あたり |weight| 上限

    # --- 執行・コスト ---
    cost_bps: float = 2.0          # 片道取引コスト(bps)。流動性の高いETF水準
    buffer_frac: float = 0.10      # ポジションバッファ半幅(平均ポジション比)
    weight_smooth_span: int = 5    # 目標の部分調整(EWMA 平滑化 ≒ Garleanu-Pedersen)

    # --- 評価 ---
    rf_rate: float = 0.0         # 年率無リスク金利(シャープ計算用)
    trading_days: int = 252

    # ウォームアップ(シグナル成立に必要な最低履歴日数)
    @property
    def warmup(self) -> int:
        return (
            self.trend_price_vol_window
            + self.trend_signal_vol_window
            + self.forecast_scale_burn
            + 10
        )

    def with_overrides(self, **kwargs) -> "BotConfig":
        return replace(self, **kwargs)
