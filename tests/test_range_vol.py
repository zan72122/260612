import numpy as np
import pandas as pd

from finbot.config import BotConfig
from finbot.data import SyntheticSource
from finbot.data.base import OHLCFrames
from finbot.portfolio import compute_target_weights
from finbot.risk.range_vol import yang_zhang_ann_vol, yang_zhang_variance


def _gbm_ohlc(days=1500, vol=0.16, seed=7, m=78) -> OHLCFrames:
    """一定ボラの GBM からブラウン橋で OHLC を作る(検証用)。"""
    rng = np.random.default_rng(seed)
    sd = vol / np.sqrt(252.0)
    rets = rng.standard_normal(days) * sd
    e = rng.standard_normal((days, m)) * (sd / np.sqrt(m))
    e += (rets[:, None] - e.sum(axis=1, keepdims=True)) / m
    log_close = np.log(100.0) + np.cumsum(rets)
    log_prev = np.concatenate([[np.log(100.0)], log_close[:-1]])
    path = log_prev[:, None] + np.cumsum(e, axis=1)
    idx = pd.bdate_range("2018-01-01", periods=days)

    def f(a):
        return pd.DataFrame({"X": np.exp(a)}, index=idx)

    return OHLCFrames(
        open=f(path[:, 0]),
        high=f(np.maximum(path.max(axis=1), log_close)),
        low=f(np.minimum(path.min(axis=1), log_close)),
        close=f(log_close),
    )


def test_synthetic_ohlc_invariants():
    src = SyntheticSource(days=800, seed=42)
    ohlc = src.load_ohlc()
    assert (ohlc.high.to_numpy() >= ohlc.open.to_numpy() - 1e-12).all()
    assert (ohlc.high.to_numpy() >= ohlc.close.to_numpy() - 1e-12).all()
    assert (ohlc.low.to_numpy() <= ohlc.open.to_numpy() + 1e-12).all()
    assert (ohlc.low.to_numpy() <= ohlc.close.to_numpy() + 1e-12).all()


def test_synthetic_ohlc_close_matches_load():
    """OHLC の終値は load() の終値と一致する(乱数ストリーム分離の確認)。"""
    src = SyntheticSource(days=800, seed=42)
    pd.testing.assert_frame_equal(src.load(), src.load_ohlc().close)


def test_yang_zhang_recovers_true_vol():
    # 離散グリッド(78 ステップ/日)では高値/安値が連続パスの極値を
    # 取りこぼすため、レンジ推定は ~10% の下方バイアスを持つ。
    # 実測値 0.146 前後(真値 0.16)を包含する範囲で検証する。
    ohlc = _gbm_ohlc(days=2000, vol=0.16)
    ann = yang_zhang_ann_vol(ohlc, span=60)["X"].dropna()
    med = float(ann.median())
    assert 0.13 < med < 0.18


def test_yang_zhang_less_noisy_than_close_to_close():
    """同じ span で YZ の推定値のバラつきは終値ベースより小さい(効率の検証)。"""
    ohlc = _gbm_ohlc(days=2000, vol=0.16)
    span = 60
    yz = yang_zhang_ann_vol(ohlc, span=span)["X"].dropna()
    cc = (
        ohlc.close.pct_change().ewm(span=span, min_periods=span // 2).std()
        * np.sqrt(252.0)
    )["X"].dropna()
    common = yz.index.intersection(cc.index)
    assert float(yz.loc[common].std()) < float(cc.loc[common].std())


def test_yang_zhang_causality():
    src = SyntheticSource(days=900, seed=11)
    ohlc = src.load_ohlc()
    var_full = yang_zhang_variance(ohlc, span=60)

    cut = 700
    altered = OHLCFrames(*(df.copy() for df in ohlc))
    for df in altered:
        df.iloc[cut:] *= 3.0
    var_altered = yang_zhang_variance(altered, span=60)
    pd.testing.assert_frame_equal(var_full.iloc[:cut], var_altered.iloc[:cut])


def test_weights_with_ohlc_respect_constraints_and_causality():
    cfg = BotConfig()
    src = SyntheticSource(days=900, seed=11)
    prices = src.load()
    ohlc = src.load_ohlc()
    w = compute_target_weights(prices, cfg, ohlc=ohlc)
    assert (w.abs() <= cfg.max_weight + 1e-9).all().all()
    assert (w.abs().sum(axis=1) <= cfg.max_leverage + 1e-6).all()

    cut = 700
    altered_prices = prices.copy()
    altered_prices.iloc[cut:] *= 3.0
    altered_ohlc = OHLCFrames(*(df.copy() for df in ohlc))
    for df in altered_ohlc:
        df.iloc[cut:] *= 3.0
    w_altered = compute_target_weights(altered_prices, cfg, ohlc=altered_ohlc)
    pd.testing.assert_frame_equal(w.iloc[:cut], w_altered.iloc[:cut])


def test_ohlc_changes_weights():
    """YZ ボラが実際に使われている(終値のみの結果と異なる)。"""
    cfg = BotConfig()
    src = SyntheticSource(days=900, seed=12)
    prices = src.load()
    w_close = compute_target_weights(prices, cfg)
    w_yz = compute_target_weights(prices, cfg, ohlc=src.load_ohlc())
    assert not np.allclose(w_close.to_numpy(), w_yz.to_numpy())
