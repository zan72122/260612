"""目標ウェイト計算パイプライン。

価格履歴 → シグナル × リスクパリティ → ボラターゲット → 制約、の順に
日次の目標ウェイト系列を生成する。バックテストとペーパートレードの
両方がこの同一関数を使うため、検証した挙動がそのまま運用される。

t 行のウェイトは t 日終値までの情報のみで計算される(因果的)。
翌日リターンへの適用(1日シフト)はバックテストエンジン側の責務。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finbot.config import BotConfig
from finbot.risk.covariance import ewma_covariance
from finbot.risk.vol_target import vol_target_leverage
from finbot.signals.ensemble import combined_signal


def compute_target_weights(prices: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    """日次目標ウェイト(index: 日付, columns: 資産)を返す。

    - リスクパリティ: 逆 EWMA ボラ加重で各資産のリスク寄与を平準化
    - シグナルティルト: モメンタムアンサンブル [-1,1] を乗算
    - ボラターゲット: EWMA 共分散による予測ボラを目標値に正規化
    - 制約: 資産別 |w| <= max_weight, グロス <= max_leverage
    """
    rets = prices.pct_change()
    signal = combined_signal(prices, cfg)

    ann_vol = rets.ewm(span=cfg.vol_span, min_periods=cfg.vol_span // 2).std() * np.sqrt(
        cfg.trading_days
    )
    inv_vol = 1.0 / ann_vol.where(ann_vol > 1e-8)
    rp = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    raw = (rp * signal).fillna(0.0)

    cov = ewma_covariance(rets, span=cfg.cov_span)
    raw_np = raw.to_numpy()
    weights = np.zeros_like(raw_np)
    for t in range(len(raw)):
        w = raw_np[t]
        lev = vol_target_leverage(
            w, cov[t], cfg.target_vol, cfg.max_leverage, cfg.trading_days
        )
        weights[t] = np.clip(w * lev, -cfg.max_weight, cfg.max_weight)

    out = pd.DataFrame(weights, index=prices.index, columns=prices.columns)
    # EWMA 平滑化: 日々の目標の揺れを均し、ターンオーバー(=コスト)を削減する。
    # 過去方向の平均なので因果性は保たれる。
    if cfg.weight_smooth_span > 1:
        out = out.ewm(span=cfg.weight_smooth_span).mean()
    # ウォームアップ期間はポジションを取らない
    out.iloc[: cfg.warmup] = 0.0
    return out
