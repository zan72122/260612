"""ボット全体の設定。

戦略パラメータは「予測精度」ではなく「リスク管理」に寄せてあり、
過学習を避けるため自由度を意図的に小さくしている。
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

    # シグナル: 時系列モメンタムのルックバック(営業日)
    momentum_lookbacks: Sequence[int] = (21, 63, 126, 252)
    # クロスセクショナル・モメンタムのルックバック
    xs_lookback: int = 126
    # 時系列モメンタムとクロスセクショナルの混合比(ts の重み)
    ts_weight: float = 0.7

    # リスクモデル
    vol_span: int = 60          # 資産別 EWMA ボラの span(営業日)
    cov_span: int = 120         # EWMA 共分散の span
    target_vol: float = 0.10    # ポートフォリオ目標ボラ(年率)
    max_leverage: float = 1.5   # グロスレバレッジ上限
    max_weight: float = 0.30    # 1資産あたり |weight| 上限

    # ドローダウン・ブレーキ
    dd_threshold: float = 0.08  # この深さを超えるDDでエクスポージャ縮小開始
    dd_full_cut: float = 0.20   # この深さで最小エクスポージャに到達
    dd_min_exposure: float = 0.25

    # 執行・コスト
    cost_bps: float = 5.0          # 片道取引コスト(bps)
    rebalance_band: float = 0.05   # 目標との最大乖離がこれ以下なら取引しない
    weight_smooth_span: int = 5    # 目標ウェイトの EWMA 平滑化(gp_enabled=False 時のみ)

    # リバランス・トランチング(タイミング運の除去)
    # ポートフォリオを n_tranches 本に分割し、各トランシェは n_tranches 日
    # 周期の異なる位相でのみリバランスする。平均化により「リバランス日の
    # 選び方」という運要素を除去し、リターン期待値を変えずに分散を下げる。
    n_tranches: int = 5

    # レンジベース・ボラ推定(Yang-Zhang)
    # OHLC が利用可能なとき、終値リターンより統計効率が数倍高い
    # Yang-Zhang 推定量で資産別ボラと共分散の対角成分を置き換える。
    use_range_vol: bool = True

    # Gârleanu-Pedersen 部分調整
    # gp_trade_rate は 1 日あたりの「目標へ近づく割合」(取引速度)。
    # コストが高いほど小さく、アルファ減衰が速いほど大きくすべき値。
    # エイム(目標)側では減衰の遅いシグナルを 1/(1 + φ/a) で過大評価する。
    gp_enabled: bool = True
    gp_trade_rate: float = 0.25

    # 評価
    rf_rate: float = 0.0         # 年率無リスク金利(シャープ計算用)
    trading_days: int = 252

    # ウォームアップ(シグナル成立に必要な最低履歴日数)
    @property
    def warmup(self) -> int:
        return max(max(self.momentum_lookbacks), self.xs_lookback, self.cov_span) + 5

    def with_overrides(self, **kwargs) -> "BotConfig":
        return replace(self, **kwargs)
