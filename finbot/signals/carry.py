"""キャリーシグナル(Koijen-Moskowitz-Pedersen-Vrugt, JFE 2018)。

キャリー = 価格が動かなかった場合の期待リターン(年率)。
資産クラスごとの計算はデータ層の責務:
    先物       : (スポット - 先物) / 先物、またはロールイールド
    株式 ETF   : 配当利回り - 短期金利
    債券 ETF   : 最終利回り + ロールダウン - 短期金利
    コモディティ: 期近-期先のロールイールド

ここでは「年率キャリーの DataFrame」を受け取り、リスク調整
(キャリー / 年率ボラ)と EWMA 平滑化(Koijen et al.: 時間平均
シグナルは SR を保ったままターンオーバー半減)だけを行う。
トレンドとの相関が低く、追加シグナルとして最も実証の厚い直交収益源。
"""

from __future__ import annotations

import pandas as pd


def carry_signal(
    carry: pd.DataFrame,
    ann_vol: pd.DataFrame,
    smooth_span: int = 90,
) -> pd.DataFrame:
    """リスク調整済みキャリー。carry / ann_vol を EWMA 平滑化して返す。

    carry: 年率キャリー(index: 日付, columns: 資産)。t 行は t 日時点の観測値。
    ann_vol: 年率ボラ(同形状)。リスク 1 単位あたりの期待キャリーに変換する。
    """
    risk_adj = carry / ann_vol.where(ann_vol > 1e-8)
    return risk_adj.ewm(span=smooth_span, min_periods=smooth_span // 3).mean()
