"""目標ウェイト計算パイプライン。

価格履歴 → シグナル × リスクパリティ → ボラターゲット → 制約 →
GP 部分調整、の順に日次の目標ウェイト系列を生成する。バックテストと
ペーパートレードの両方がこの同一関数を使うため、検証した挙動が
そのまま運用される。

t 行のウェイトは t 日終値までの情報のみで計算される(因果的)。
翌日リターンへの適用(1日シフト)はバックテストエンジン側の責務。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finbot.config import BotConfig
from finbot.data.base import OHLCFrames
from finbot.portfolio.gp import partial_adjustment
from finbot.risk.covariance import ewma_covariance, rescale_diagonal
from finbot.risk.range_vol import yang_zhang_variance
from finbot.risk.vol_target import vol_target_leverage
from finbot.signals.ensemble import combined_signal


def compute_target_weights(
    prices: pd.DataFrame,
    cfg: BotConfig,
    ohlc: OHLCFrames | None = None,
) -> pd.DataFrame:
    """日次目標ウェイト(index: 日付, columns: 資産)を返す。

    - リスクパリティ: 逆ボラ加重で各資産のリスク寄与を平準化
    - シグナルティルト: モメンタムアンサンブル [-1,1] を乗算
    - ボラターゲット: 共分散による予測ボラを目標値に正規化
    - 制約: 資産別 |w| <= max_weight, グロス <= max_leverage
    - GP 部分調整: エイムへ毎日 gp_trade_rate だけ近づく(コスト最適化)

    ohlc が与えられ use_range_vol が真のとき、資産別ボラと共分散の
    対角成分を Yang-Zhang レンジ推定に置き換える(相関は EWMA を維持)。
    """
    rets = prices.pct_change()
    signal = combined_signal(prices, cfg)

    use_yz = ohlc is not None and cfg.use_range_vol
    if use_yz:
        ohlc = ohlc.aligned_to(prices)
        yz_var = yang_zhang_variance(ohlc, span=cfg.vol_span)
        ann_vol = np.sqrt(yz_var * cfg.trading_days)
    else:
        ann_vol = rets.ewm(span=cfg.vol_span, min_periods=cfg.vol_span // 2).std() * np.sqrt(
            cfg.trading_days
        )
    inv_vol = 1.0 / ann_vol.where(ann_vol > 1e-8)
    rp = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    raw = (rp * signal).fillna(0.0)

    cov = ewma_covariance(rets, span=cfg.cov_span)
    if use_yz:
        # 相関は EWMA のまま、分散だけ共分散と同じ平滑度の YZ 推定に差し替え
        cov_var = yang_zhang_variance(ohlc, span=cfg.cov_span).to_numpy(dtype=float)
        cov = rescale_diagonal(cov, cov_var)
    raw_np = raw.to_numpy()
    weights = np.zeros_like(raw_np)
    for t in range(len(raw)):
        w = raw_np[t]
        lev = vol_target_leverage(
            w, cov[t], cfg.target_vol, cfg.max_leverage, cfg.trading_days
        )
        weights[t] = np.clip(w * lev, -cfg.max_weight, cfg.max_weight)

    if cfg.gp_enabled:
        # GP 部分調整: 全量リバランスではなくエイムへ毎日 τ だけ近づける。
        # 過去方向の再帰なので因果性は保たれ、凸結合なので制約も保存される。
        weights = partial_adjustment(weights, cfg.gp_trade_rate)
        out = pd.DataFrame(weights, index=prices.index, columns=prices.columns)
    else:
        out = pd.DataFrame(weights, index=prices.index, columns=prices.columns)
        # EWMA 平滑化: 日々の目標の揺れを均し、ターンオーバー(=コスト)を削減する。
        # 過去方向の平均なので因果性は保たれる。
        if cfg.weight_smooth_span > 1:
            out = out.ewm(span=cfg.weight_smooth_span).mean()
    # ウォームアップ期間はポジションを取らない
    out.iloc[: cfg.warmup] = 0.0
    return out
