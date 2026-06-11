"""ウォークフォワード(ローリング・アウトオブサンプル)評価。

v2 での位置づけ: ウォークフォワードは false discovery の防止では
CPCV に劣る(合成環境の比較実験で最弱)ため、モデル選択には
cpcv を使い、こちらは「時系列順の現実的シミュレーション」として
最終確認に用いる(docs/DESIGN_V2.md §7)。

各フォールドで「学習窓のシャープが最良のパラメータ」を選び、
それを直後のテスト窓に適用した OOS リターンだけを連結して評価する。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finbot.backtest.engine import run_backtest
from finbot.backtest.grid import DEFAULT_GRID
from finbot.backtest.metrics import summary
from finbot.config import BotConfig


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    stats: dict[str, float]
    folds: list[dict]


def run_walkforward(
    prices: pd.DataFrame,
    cfg: BotConfig,
    train_days: int = 756,
    test_days: int = 252,
    grid: tuple[dict, ...] = DEFAULT_GRID,
    carry: pd.DataFrame | None = None,
) -> WalkForwardResult:
    oos_parts: list[pd.Series] = []
    folds: list[dict] = []

    start = 0
    while start + train_days + test_days <= len(prices):
        train = prices.iloc[start : start + train_days]
        # テスト窓にはシグナルのウォームアップ用の履歴を前置する
        test_begin = start + train_days
        ctx_begin = max(0, test_begin - cfg.warmup - 5)
        test_ctx = prices.iloc[ctx_begin : test_begin + test_days]

        best_sharpe, best_params = -float("inf"), grid[0]
        for params in grid:
            res = run_backtest(train, cfg.with_overrides(**params), carry=carry)
            if res.stats["sharpe"] > best_sharpe:
                best_sharpe, best_params = res.stats["sharpe"], params

        oos_cfg = cfg.with_overrides(**best_params)
        oos_res = run_backtest(test_ctx, oos_cfg, carry=carry)
        oos = oos_res.returns.loc[prices.index[test_begin] :]
        oos_parts.append(oos)
        folds.append(
            {
                "test_start": str(prices.index[test_begin].date()),
                "params": best_params,
                "train_sharpe": best_sharpe,
                "oos_sharpe": summary(oos, cfg.rf_rate, cfg.trading_days)["sharpe"],
            }
        )
        start += test_days

    if not oos_parts:
        raise ValueError(
            f"データ不足: walkforward には {train_days + test_days} 日以上必要"
        )

    oos_returns = pd.concat(oos_parts)
    stats = summary(oos_returns, cfg.rf_rate, cfg.trading_days)
    return WalkForwardResult(oos_returns=oos_returns, stats=stats, folds=folds)
