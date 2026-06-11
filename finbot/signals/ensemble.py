"""フォーキャスト統合層(Carver 規約 / AQR "Don't Just Mix, Integrate")。

各シグナルを共通のフォーキャスト単位(E|f| = 10、±20 キャップ)に
スケールしてから加重平均し、フォーキャスト分散乗数(FDM)で
統合後のスケールを回復する。スリーブ分割(シグナルごとに別の
ポートフォリオを持つ)より統合の方が情報比で優位という実証に従う。

スケーラは因果的な expanding 平均で推定する(t 日のスケーラは
t 日までの |raw| のみに依存)。固定スカラーのチューニングを避け、
ユニバースを差し替えてもそのまま動くようにするため。
"""

from __future__ import annotations

import pandas as pd

from finbot.config import BotConfig
from finbot.signals.carry import carry_signal
from finbot.signals.trend import ewmac_trend


def scale_forecast(
    raw: pd.DataFrame,
    target_abs: float = 10.0,
    cap: float = 20.0,
    burn: int = 63,
) -> pd.DataFrame:
    """生シグナルをフォーキャスト単位にスケール(E|f| ≈ target_abs, |f| <= cap)。"""
    abs_mean = raw.abs().mean(axis=1)
    level = abs_mean.expanding(min_periods=burn).mean()
    scalar = target_abs / level.where(level > 1e-12)
    return raw.mul(scalar, axis=0).clip(-cap, cap)


def combined_forecast(
    prices: pd.DataFrame,
    cfg: BotConfig,
    ann_vol: pd.DataFrame,
    carry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """トレンド + キャリーの統合フォーキャスト。各資産 [-cap, cap]。

    carry が無い(データ未提供 or carry_weight=0)場合はトレンド単独。
    """
    f_trend = scale_forecast(
        ewmac_trend(
            prices,
            cfg.trend_pairs,
            cfg.trend_price_vol_window,
            cfg.trend_signal_vol_window,
        ),
        cfg.forecast_abs_target,
        cfg.forecast_cap,
        cfg.forecast_scale_burn,
    )

    use_carry = carry is not None and cfg.carry_weight > 0
    if not use_carry:
        return f_trend

    f_carry = scale_forecast(
        carry_signal(
            carry.reindex(index=prices.index, columns=prices.columns),
            ann_vol,
            cfg.carry_smooth_span,
        ),
        cfg.forecast_abs_target,
        cfg.forecast_cap,
        cfg.forecast_scale_burn,
    )
    w_sum = cfg.trend_weight + cfg.carry_weight
    mixed = (cfg.trend_weight * f_trend + cfg.carry_weight * f_carry.fillna(0.0)) / w_sum
    return (cfg.fdm * mixed).clip(-cfg.forecast_cap, cfg.forecast_cap)
