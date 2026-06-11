"""目標ウェイト計算パイプライン(設計方針 v2)。

価格履歴(+任意のキャリー)→ 統合フォーキャスト × リスクパリティ
→ ボラターゲット → 制約、の順に日次の目標ウェイト系列を生成する。
バックテストとペーパートレードの両方がこの同一関数を使うため、
検証した挙動がそのまま運用される。

t 行のウェイトは t 日終値までの情報のみで計算される(因果的)。
翌日リターンへの適用(1日シフト)はバックテストエンジン側の責務。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finbot.config import BotConfig
from finbot.risk.covariance import blended_vol, build_covariance, ewma_correlation
from finbot.risk.vol_target import vol_target_leverage
from finbot.signals.ensemble import combined_forecast


def compute_target_weights(
    prices: pd.DataFrame,
    cfg: BotConfig,
    carry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """日次目標ウェイト(index: 日付, columns: 資産)を返す。

    - リスクパリティ: 逆ボラ加重(短期 EWMA + 長期アンカーのブレンドボラ)
    - フォーキャストティルト: 統合フォーキャスト/10 を乗算([-2, 2])
    - ボラターゲット: ボラ×相関の分離推定による予測ボラを目標値に正規化
    - 制約: 資産別 |w| <= max_weight, グロス <= max_leverage
    """
    rets = prices.pct_change()

    ann_vol = blended_vol(
        rets,
        halflife=cfg.vol_halflife,
        blend=cfg.vol_blend,
        long_min_periods=cfg.vol_long_min_periods,
        trading_days=cfg.trading_days,
    )
    forecast = combined_forecast(prices, cfg, ann_vol, carry)

    inv_vol = 1.0 / ann_vol.where(ann_vol > 1e-8)
    rp = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    raw = (rp * forecast / cfg.forecast_abs_target).fillna(0.0)

    corr = ewma_correlation(rets, cfg.corr_halflife, cfg.corr_shrinkage)
    cov = build_covariance(ann_vol.fillna(0.0), corr, cfg.trading_days)
    raw_np = raw.to_numpy()
    weights = np.zeros_like(raw_np)
    for t in range(len(raw)):
        w = raw_np[t]
        lev = vol_target_leverage(
            w, cov[t], cfg.target_vol, cfg.max_leverage, cfg.trading_days
        )
        weights[t] = np.clip(w * lev, -cfg.max_weight, cfg.max_weight)

    out = pd.DataFrame(weights, index=prices.index, columns=prices.columns)
    # 部分調整(Garleanu-Pedersen の実務簡略形): EWMA 平滑化は
    # 「毎日 κ=2/(span+1) だけ目標へ近づく」ことと等価で、二次コスト下の
    # 最適執行を近似する。過去方向の平均なので因果性は保たれる。
    if cfg.weight_smooth_span > 1:
        out = out.ewm(span=cfg.weight_smooth_span).mean()
    # ウォームアップ期間はポジションを取らない
    out.iloc[: cfg.warmup] = 0.0
    return out
