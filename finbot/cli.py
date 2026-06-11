"""CLI エントリポイント。

    python -m finbot backtest    [--source synthetic|csv] [--csv-path P] [--carry-csv P] [--days N] [--seed S]
    python -m finbot cpcv        [同上] [--groups N] [--k-test K]
    python -m finbot walkforward [同上]
    python -m finbot paper       [同上] [--state paper_state.json] [--cash 1000000]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from finbot.backtest.cpcv import run_cpcv
from finbot.backtest.engine import run_backtest
from finbot.backtest.metrics import format_report
from finbot.backtest.walkforward import run_walkforward
from finbot.config import BotConfig
from finbot.data import CSVSource, DataSource, SyntheticSource
from finbot.live.runner import run_paper_tick


def _build_source(args: argparse.Namespace) -> DataSource:
    if args.source == "csv":
        if not args.csv_path:
            sys.exit("--source csv には --csv-path が必要です")
        return CSVSource(args.csv_path, carry_path=args.carry_csv)
    return SyntheticSource(days=args.days, seed=args.seed)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=["synthetic", "csv"], default="synthetic")
    p.add_argument("--csv-path", default=None)
    p.add_argument("--carry-csv", default=None, help="年率キャリーの CSV(任意)")
    p.add_argument("--days", type=int, default=2520, help="合成データの日数")
    p.add_argument("--seed", type=int, default=42, help="合成データのシード")
    p.add_argument("--target-vol", type=float, default=None, help="目標年率ボラ")


def _build_config(args: argparse.Namespace) -> BotConfig:
    cfg = BotConfig()
    if args.target_vol is not None:
        cfg = cfg.with_overrides(target_vol=args.target_vol)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finbot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_bt = sub.add_parser("backtest", help="インサンプル・バックテスト")
    _add_common(p_bt)

    p_cv = sub.add_parser("cpcv", help="CPCV + Deflated Sharpe(モデル選択用)")
    _add_common(p_cv)
    p_cv.add_argument("--groups", type=int, default=10)
    p_cv.add_argument("--k-test", type=int, default=2)

    p_wf = sub.add_parser("walkforward", help="ウォークフォワード OOS 評価(最終確認用)")
    _add_common(p_wf)

    p_paper = sub.add_parser("paper", help="ペーパートレード 1 ティック実行")
    _add_common(p_paper)
    p_paper.add_argument("--state", default="paper_state.json")
    p_paper.add_argument("--cash", type=float, default=1_000_000.0)

    args = parser.parse_args(argv)
    cfg = _build_config(args)
    source = _build_source(args)

    if args.command == "backtest":
        prices = source.load()
        res = run_backtest(prices, cfg, carry=source.load_carry())
        print(format_report(res.stats, "バックテスト(インサンプル・参考値)"))
        print("\n注意: インサンプルの数値は楽観的です。cpcv / walkforward の OOS 値を信頼してください。")

    elif args.command == "cpcv":
        prices = source.load()
        res = run_cpcv(
            prices,
            cfg,
            n_groups=args.groups,
            k_test=args.k_test,
            carry=source.load_carry(),
        )
        ps = np.array(res.path_sharpes)
        print("=== CPCV(OOS パス分布・モデル選択用)===")
        print(f"OOS パス数        : {len(ps)}")
        print(f"パス・シャープ     : 中央値 {np.median(ps):+.2f} / 平均 {ps.mean():+.2f}")
        print(f"                    最小 {ps.min():+.2f} / 最大 {ps.max():+.2f} / 負率 {(ps < 0).mean():.0%}")
        print(f"PBO(過学習確率)  : {res.pbo:.2f}   (目安: 0.2 以下)")
        print(f"PSR               : {res.psr:.3f}")
        print(f"DSR               : {res.dsr:.3f}  (採用基準: 0.95 超)")
        print(f"全期間最良コンフィグ: {res.best_params or '基準設定'}")
        if res.dsr <= 0.95:
            print("\n警告: DSR <= 0.95。選択バイアスを補正すると有意なスキルが残りません。")

    elif args.command == "walkforward":
        prices = source.load()
        res = run_walkforward(prices, cfg, carry=source.load_carry())
        print(format_report(res.stats, "ウォークフォワード OOS(時系列順の最終確認)"))
        print("\n--- フォールド詳細 ---")
        for f in res.folds:
            print(
                f"{f['test_start']}〜: train SR {f['train_sharpe']:+.2f} "
                f"→ OOS SR {f['oos_sharpe']:+.2f}  params={f['params']}"
            )

    elif args.command == "paper":
        broker, orders, target = run_paper_tick(
            source, cfg, state_path=args.state, initial_cash=args.cash
        )
        prices = source.load()
        print("=== ペーパートレード ティック実行 ===")
        print(f"基準日: {prices.index[-1].date()}")
        print("\n目標ウェイト:")
        for asset, w in target.items():
            print(f"  {asset:<10}: {w:+.3f}")
        print(f"\n執行注文: {len(orders)} 件")
        for o in orders:
            side = "BUY " if o.units > 0 else "SELL"
            print(f"  {side} {o.asset:<10} {abs(o.units):>12.2f} units @ {o.price:.2f}")
        print(f"\nポートフォリオ評価額: {broker.portfolio_value(prices.iloc[-1]):,.0f}")
        print(f"現金残高           : {broker.cash:,.0f}")
        print(f"状態ファイル       : {args.state}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
