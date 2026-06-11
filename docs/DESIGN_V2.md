# finbot 設計方針 v2 — OOSシャープレシオ最大化のための改訂設計書

本書は v1 設計方針(README記載の7要素)を、学術文献(JF/JFE/RFS/JPM/FAJ)と実務家研究
(AQR・Man AHL・CFM・Quantica・Robeco・Rob Carver/pysystemtrade)の網羅的調査に基づいて
全面改訂したものである。調査は5系統の並列リサーチ+2系統の敵対的ファクトチェック
(反証・訂正の探索)を経ており、各主張に信頼度を付した。

**根本的な認識転換**: v1 は「リスク管理の積み上げ」に偏っていた。しかし実証が示すのは、
シャープの分母(リスク制御)系の工夫はおおむね出尽くしており、OOSシャープを実際に動かす
レバーは (a) **直交シグナルの追加(キャリー・バリュー)**、(b) **シグナル関数形のコスト効率**、
(c) **ブレッドス(資産数)**、(d) **過学習の統計的排除** の4つだということである。
逆に、DDブレーキ・ストップロス・高度なML・HRP等は OOS シャープに中立か有害である。

---

## 0. 判定サマリー

| # | v1 コンポーネント | 判定 | 改訂内容(要点) |
|---|---|---|---|
| 1 | 多資産分散(8資産) | **維持+拡張** | 唯一最大のレバー。可能なら8→15-25資産へ。IDM≈1.25(8資産)→1.35-2.5(10-30資産) |
| 2a | TSモメンタム(21/63/126/252日 単純リターン平均) | **置換** | Baz et al. (2015) EWMAC 3ペア+非線形応答関数。連続シグナル化でターンオーバー約1/3減、ネットシャープ改善 |
| 2b | クロスセクショナル・モメンタム | **縮小(原則撤廃)** | TSがXSを包含(MOP 2012; Goyal-Jegadeesh 2018)。8資産ではランクがノイズ。最大でも小スリーブ |
| 3 | EWMA共分散リスクパリティ+ティルト | **修正** | ボラと相関を分離推定: ボラ=短期ブレンド(HL≈20-32日×0.7 + 長期アンカー×0.3、レンジベース推定量で雑音低減)、相関=HL≈150-250日+軽い線形シュリンク。Σ=D·C·D で再合成 |
| 4 | ボラターゲティング(年率10%) | **修正** | 維持するが期待値を修正: シャープ改善は株式/クレジットのみ、債券/コモディティ/FXは中立(Harvey et al. 2018)。全資産でテール/vol-of-vol削減効果はある。1/σ²型のアグレッシブなボラタイミングは行わない(Cederburg et al. 2020) |
| 5 | コスト制御(bps+リバランスバンド) | **修正** | バンドを「平均ポジションの±10%バッファ」に再定義し、バンド端まで部分調整。ルール採否にコスト上限(年間 ≤0.13-0.15 SR単位/ルール)を導入 |
| 6 | ドローダウン・ブレーキ | **撤廃** | OOSシャープに中立〜マイナス。ボラターゲット+トレンドシグナルが既にデレバレッジを内包しており二重カウント。クライシスアルファ(凸性)を毀損する |
| 7 | ウォークフォワード検証 | **置換** | CPCV(purge+embargo)による選択 + Deflated Sharpe Ratio による多重検定補正 + 最終時系列ホールドアウト(1回限り)。WF単独は false discovery 防止で最弱(Arian et al. 2024) |
| — | (新規) キャリーシグナル | **追加** | 最優先の追加。トレンドとの相関が低く、診断SR(グロス)はクラス内≈0.8/分散後≈1.2(Koijen et al. JFE 2018)。トレンド60/キャリー40の統合フォーキャスト |
| — | (新規) バリューシグナル | **追加(第2優先)** | 5年リバーサル(Asness-Moskowitz-Pedersen JF 2013)。モメンタムと相関≈−0.5、コンボでSRが約2倍 |
| — | (新規) フォーキャスト統合層 | **追加** | Carver方式: 各シグナルを E\|f\|=10 にスケール、±20でキャップ、フォーキャスト分散乗数(FDM)で再スケール。スリーブ分割ではなく統合(AQR "Don't Just Mix, Integrate") |
| — | (検討時の禁止事項) ML/HRP/ストップ等 | **不採用** | §9 参照 |

---

## 1. 多資産分散 — 維持+拡張【信頼度: 高】

- 分散がシャープに与える効果は乗数的: Carver の計測では Instrument Diversification
  Multiplier (IDM) は 1銘柄=1.0、4-5銘柄≈1.25、10銘柄≈1.35、30銘柄超≈2.5
  (qoppac/pysystemtrade、100+銘柄バックテスト)。**8資産・4資産クラスの現構成の
  期待値は IDM≈1.3-1.5、トレンド+キャリーのシステムSRでグロス0.6-0.8**。
- AQR "Trends Everywhere" (Babu et al., JOIM 2020): 代替市場でもトレンドSRは伝統市場と
  同等以上かつ低相関 → **シグナルを凝るより資産を増やす方が効率的**。
- 実装指針: ユニバースを設定可能にし、流動性とコスト(§5)が許す限り 15-25 資産
  (株価指数複数地域、国債2-3年限、金+産業金属、エネルギー、農産物、可能ならFX)へ。
  資産クラス内の相関が高いものはクラスタとして扱い、クラスタ間で等リスク配分。

## 2. シグナル生成 — 置換

### 2.1 トレンド: Baz et al. (2015) EWMAC + 非線形応答【信頼度: 高(独立再現2件で正確なパラメータ確認済み)】

「Dissecting Investment Strategies in the Cross Section and Time Series」
(Baz, Granger, Harvey, Le Roux, Rattray 2015, SSRN 2695101)の構成をそのまま採用する:

```
タイムスケールペア (S, L) ∈ {(8,24), (16,48), (32,96)}   # n→HL変換: HL = log(0.5)/log(1−1/n)
x_k = EWMA_S(price) − EWMA_L(price)
y_k = x_k / rolling_std(price, 63日)        # 価格の63日標準偏差(リターンではなく価格)
z_k = y_k / rolling_std(y_k, 252日)
u_k = z_k · exp(−z_k² / 4) / 0.89           # 応答関数(z=±√2でピーク、極端値で減衰)
trend_forecast = (1/3) Σ_k u_k
```

採用理由(すべて検証済み):
- **連続シグナルの優位はアルファではなくコスト経由**: Baltas-Kosowski のグロスSRは
  SIGN 1.04 vs 連続TREND 0.99 で統計的に差がないが、連続化+優良ボラ推定量の併用で
  ターンオーバーが1/3超削減され、**ネット**で優位。
- MAクロスオーバーとTSMOMは線形フィルタとして等価(Levine-Pedersen FAJ 2016)なので、
  関数形の選択で重要なのは「速度の混合」と「非線形応答」だけ。
- 応答関数 z·exp(−z²/4) は極端なトレンドでポジションを自然に縮め、CFMが実証した
  P&L飽和(tanh型; Dao et al. 2016, arXiv:1607.02410)と整合。キャップの主目的は
  リスク制御と凸性保全であり、グロスSR向上ではない点に注意。
- v1 の 21/63/126/252 日「単純リターン平均」は (16,48)〜(32,96) 帯に概ね対応するが、
  正規化と応答関数がないためボラ変化に対しポジションが不安定になる。

**速度配分**: Sharpe は遅いほど高く、凸性(クライシスアルファ)は速いほど高い
(Man AHL "The Need for Speed" 2023; Martin 2021)。Quantica の実証では 2015-2022 は
HL 60-100日が最強、SG Trend指数の複製は HL 60-70日。→ **3ペア等ウェイトを基本とし、
過去データで「最良の速度」を選ばない**(選ぶならDSRで罰する; §7)。

### 2.2 クロスセクショナル・モメンタム — 原則撤廃【信頼度: 高】

- MOP (JFE 2012): TSMOMはXSMOMを回帰で包含。Goyal-Jegadeesh (RFS 2018): TS−XSの差は
  実質「時変ネットロング(マーケットタイミング)」であり追加のモメンタム効果ではない。
- 8資産・異種クラスではランクの母数が小さすぎノイズが支配的。撤廃するか、
  資産クラス内(例: 株価指数間)に限定した小スリーブに縮小する。

### 2.3 スキップ月は不要【信頼度: 高】

12-2(直近1ヶ月除外)は個別株の短期リバーサル対策の慣行。先物/指数ETFには
短期リバーサルがなく、MOPは1ヶ月TSMOMも有効と報告 → ルックバックは直近日まで使う。

## 3. リスク推定 — 修正(ボラと相関の分離)【信頼度: 高(構成要素ごとに中〜高)】

v1 の単一EWMA共分散を、**ボラ(対角)と相関(非対角)の分離推定**に置き換える:

```
σ_i = 0.7 · EWMA_vol(HL≈20-32日) + 0.3 · 長期平均ボラ(≈10年)     # Carverブレンド
      短期レッグはレンジベース推定量(Parkinson/Garman-Klass/Yang-Zhang、
      ギャップ・離散化バイアス補正済み)で雑音を削減
C   = EWMA_corr(HL≈150-250日) に軽い線形シュリンク(δ≈0.1-0.3、定相関ターゲット)
Σ   = D(σ) · C · D(σ)
```

- RiskMetrics λ=0.94(HL≈11日)が古典だが、ポジションサイジング用途では速すぎて
  ターンオーバーを浪費する。Quantica は decay 0.94/スパン32観測を明示使用。
  Harvey et al. (2018) は HL20日で頑健と報告。**HL 20-32日+長期アンカー70/30** が
  実務のコンセンサス帯(Carver自身は回帰上 ~59/41 とも報告、70/30は丸め)。
- 長期アンカーの役割: 静穏期の後にポジションが過大化するのを防ぐ(ボラの長期平均回帰)。
- 相関は遅く推定する: Barra USE4 の標準(ボラHL42日 vs 相関HL200日)と RMT研究
  (~1年窓が実現分散の最良推定)に一致。
- N=8, T≥250 なら N/T≈0.03 でシュリンケージの限界効用は小さい(Ledoit-Wolf線形で十分、
  非線形シュリンクは大規模ユニバース用)。**「どの構造推定量か」より「どの時定数か」が支配的**。
- 非対称応答(GJR型: 下落でボラ推定を速める)は**株式スリーブのみ**有効
  (Hansen-Lunde 2005: FXではGARCH(1,1)を超えるものなし)。
- HAR-RVは日中データがあって初めてEWMAに勝つ。日次OHLCVのみなら
  「HAR-on-Parkinson」か上記ブレンドで実益の大半を回収できる。

## 4. ボラティリティ・ターゲティング — 修正(期待値の再設定)【信頼度: 高】

**維持する構造**: (a) 資産レベルで 1/σ_i サイジング(リスクパリティと等価)、
(b) ポートフォリオレベルで年率目標(10%)に正規化。

**修正する認識**(ここが v1 の最大の誤り):
- Harvey et al. (JPM 2018, 60+資産, 1926-2017): ボラターゲティングのシャープ改善は
  **株式・クレジットのみ**(US株 SR 0.40→0.48-0.51)。債券・FX・コモディティは中立。
  ただし**全資産クラス**で vol-of-vol 削減(例: 4.6%→1.8%)と左テール切断の効果はある。
  メカニズムはレバレッジ効果(リターンとボラ変化の負相関)による事実上の
  短期モメンタムオーバーレイ。
- Moreira-Muir (JF 2017) 型の 1/σ² アグレッシブなボラタイミングは、
  Cederburg et al. (JFE 2020) がリアルタイム実装不能・OOSで劣後と示し、
  Barroso-Detzel (JFE 2021) がコスト後はマーケットポートフォリオ以外で消滅と示した。
  → **行わない**。(Xu 2024等の修正ルールで復活するとの反論もあるが、低〜中信頼度)
- TSMOMの「アルファ」の相当部分はボラスケーリング由来(Kim-Tse-Wald JFM 2016:
  スケールあり1.27%/月 vs なし0.41%/月)。つまり **vol-scaling は既にシステムの
  リターンエンジンの一部**であり、これ以上の上乗せ(DDブレーキ等)は二重カウント。
- 実装: ターゲット超過時の縮小は毎日、ただし§5のバッファ経由で執行。
  レバレッジ上限(例: グロス2-3倍)を明示し、極端な静穏期の過大レバレッジを防ぐ。
  Robeco の条件付きターゲティング(極端ボラ状態のみ調整; Bongaerts et al. FAJ 2020,
  10/10市場で連続型に勝利・低ターンオーバー)は採用候補【信頼度: 中】。

## 5. コスト制御 — 修正(バッファリング+部分調整+ルール採否基準)【信頼度: 高】

- **コストモデルの較正**: 一律5bpsは過大。実測ベースで先物 0.5-2bp、ETF 1-5bp(片道)
  + ETFは経費率(例: TLT 15bp/年)。ES≈0.4-0.5bp、米国債先物≈1-1.6bp、金≈2-6bp
  (CME TCA; Frazzini-Israel-Moskowitz は学術推定の1/10の実コストを実証)。
  過大なコスト想定は「動かないbot」という別の過学習を生む。
- **ポジションバッファ(リバランスバンドの再定義)**: 目標ポジションの周囲
  ±10%(平均ポジション比)の帯を置き、帯内なら売買せず、帯外なら**帯の端まで**戻す
  (中心まで戻さない)。比例コスト下の最適方策は「バンド端への最小調整」であり
  (Davis-Norman 1990)、バンド幅はコストの**3乗根**でしかスケールしない
  (コスト見積りが10倍違っても最適幅は約2.15倍)→ ±10%の固定値で十分頑健。
- **部分調整(Garleanu-Pedersen JF 2013)**: 二次コスト(マーケットインパクト)に対する
  最適は「aimポートフォリオへ毎日一定割合だけ近づく」+「減衰の遅いシグナルを
  aimで過大評価する」。実務簡略形: `pos_t = pos_{t-1} + κ·(target_t − pos_{t-1})`,
  κ≈0.1-0.25/日(κの値自体は実務ヒューリスティック)。
  注意: 比例コスト→バンド、二次コスト→部分調整、と理論的根拠が別物。両方を重ねる
  (部分調整した目標に対してバッファ判定)のが実務の標準。
- **ルール採否のコスト上限(Carver "speed limit")**: ルールごとの年間コストを
  SR単位(コスト% ÷ 年率ボラ%)で見積り、**年間 0.13-0.15 SR単位以下**
  (= 期待プレコストSRの約1/3以下)のルールのみ採用。EWMAC(2,8)のような高速ルールは
  低コスト先物以外で自動的に脱落する。
- フォーキャスト平滑化(シグナルのEWMA)よりポジションバッファを最終層に置く方が
  頑健(平滑化はボラ急騰時の反応を遅らせるが、バッファは危機時に自然に貫通する)。

## 6. ドローダウン・ブレーキ — 撤廃【信頼度: 中〜高】

- メカニカルな損失後デレバレッジは両テールを切断するだけで、ボラターゲット済みの
  トレンドシステムでは**OOSシャープへの期待効果はゼロ〜マイナス**
  (Harvey-Rattray-van Hemert『Strategic Risk Management』2021; van Hemert et al.
  "Drawdowns" SSRN 3583864)。Grossman-Zhou (1993) 系のDD制約は成長率を犠牲にする
  保険であり、Klass-Nowicki (2005) は最適ですらない場合を示した。
- システムは既にデレバレッジを2経路で内包する: (i) トレンドシグナルの符号反転・減衰、
  (ii) ボラ急騰時のボラターゲット縮小。第3のブレーキは二重カウント。
- 凸性コスト: トレンドのクライシスアルファは「速く・キャップなし」で最大になり、
  ブレーキは危機リバウンド直前にデレバレッジする(Man AHL凸性研究)。
- 個別ポジションのストップロスも不採用: ランダムウォーク下でストップは期待リターンを
  常に毀損し(Kaminski-Lo JFM 2014)、モメンタムはシグナルが既に収穫している。
  Clare et al. (2013) もトレンドルールへのストップ追加は性能悪化と報告。
  (反証として Han-Zhou-Zhu はモメンタム戦略でストップがシャープ改善と報告
  しており完全な決着はないが、ボラスケール済みトレンドへの追加効果は未実証)
- **代替**: DDが心理的・契約的に問題なら、目標ボラ自体を下げる(分母の恒久調整)か、
  口座レベルの非常停止(kill switch)として「想定ボラのk倍超の損失で人間レビュー」を
  置く。これはアルファ装置ではなく運用ガバナンスとして実装する。

## 7. 検証 — 置換(CPCV + DSR + 単発ホールドアウト)【信頼度: 高】

- **選択フェーズ**: Combinatorial Purged Cross-Validation(例 N=10グループ, k=2 →
  45分割・複数OOSパス)で設計比較。日次ラベルならpurgeは軽微で良いが、
  多日ホライズンのラベルを使うならembargo≈全サンプルの1%。合成環境の比較実験
  (Arian-Norouzi-Seco, KBS 2024, SSRN 4686376)で **CPCVがPBO最小・WFがfalse
  discovery防止で最弱**。WFは「最終確認の現実シミュレーション」に格下げする。
- **多重検定補正**: 試行ログを必ず取り(コンフィグ毎に記録)、
  Deflated Sharpe Ratio を報告する:
  - PSR(SR*) = Φ[(ŜR−SR*)·√(T−1) / √(1 − γ₃ŜR + ((γ₄−1)/4)ŜR²)]
  - SR₀ = √V[{SR_n}]·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))], γ≈0.5772(N=実効独立試行数)
  - 採用基準: **DSR > 0.95**(おおむね t>3 の Harvey-Liu 基準と整合)。
  - MinTRL: SR≈1.0なら日次約2-3年で帰無SR=0を95%棄却可能 — これより短い検証期間の
    比較は統計的に無意味と自覚する。
- **パラメータは選ばず平均する**: ルックバック・HL等は「最良値の選択」をやめ、
  グリッド全体のアンサンブル(§2の3ペア等ウェイト、§3の帯域中央値)に固定する。
  これにより選択問題(N試行ぶんのDSRデフレーション)が単一の事前登録試行に変わる。
- **最終ホールドアウト**: 直近3-5年の時系列ブロックを開発中は一切触れず、
  設計凍結後に1回だけ評価。この数値だけを対外報告する。
- データ要件: 8資産日次で20-30年(2000-02, 2008, 2013, 2020, 2022の各レジームを含む)。
  10-30個の設計自由度の大半は**探索ではなく事前の経済的根拠で固定**する
  (実効サンプルはクロス相関のため名目60k観測よりはるかに小さい)。

## 8. 追加コンポーネント(優先順)

### 8.1 キャリー — 最優先追加【存在・分散効果: 高 / 水準: 中】

- Koijen-Moskowitz-Pedersen-Vrugt "Carry" (JFE 2018): キャリー(価格不変時の期待リターン)
  は全資産クラスでクロスセクション・時系列の両方を予測。**クラス内平均グロスSR≈0.8、
  グローバル分散キャリーファクター≈1.2**(検証で 0.7/1.1→0.8/1.2 に訂正済み)。
  時間平均シグナル版はSR維持のままターンオーバー半減。
- 計算(日次OHLCVボットでの実装): 先物なら (スポット−先物)/先物 または期近-期先ロール
  イールド。ETF代理なら: 株式=配当利回り−短期金利、債券=イールド+ロールダウン、
  金=−(短期金利)+リース率近似、コモディティ=ロールイールド。
- トレンドとの相関は低い(0近傍、実務報告0-0.2だが正確な出典は未確定)。
  Carver/QuantConnect再現: トレンド単独SR 0.294 → トレンド+キャリーSR 0.749。
- 注意: キャリーは負スキュー(クラッシュリスク)。フォーキャストキャップ(±20)と
  ボラターゲットで管理し、キャリー配分は40%を上限とする。

### 8.2 バリュー(5年リバーサル) — 第2優先【信頼度: 中〜高】

- Asness-Moskowitz-Pedersen (JF 2013): 非株式クラスのバリュー=過去5年リターンの符号反転。
  モメンタムとの相関≈−0.5〜−0.6、50/50コンボはSR 0.80 とバリュー単独の2倍超。
- 配分は小さく(10-20%)。低速・低ターンオーバーなのでコスト面はほぼ無料。

### 8.3 フォーキャスト統合層(Carver/AQR方式)【信頼度: 中〜高】

- 各シグナルを **E|forecast|=10 にスケール、±20でキャップ**。
- 統合: `combined = FDM · Σ_j w_j · f_j`、FDM=フォーキャスト分散乗数
  (統合後にE|f|=10を回復、シグナル間相関が低いほど大)。
- スタイルウェイト: **トレンド≈60% / キャリー≈40%**(SR最大化点; Carver)、
  バリュー追加時は 50/35/15 程度から。スリーブ分割ではなく統合方式
  (AQR Fitzgibbons et al. 2017: 統合はミックスより約1%/年・IR+40%)。
- ポジション: `N_i = Capital · IDM · w_i · combined_i/10 · τ / (σ_i · price_i)`。

### 8.4 不採用(調査の結果、追加しないと決めたもの)

- **季節性**: 商品季節性のOOSエビデンスは2000年以降減衰。最下位優先度、スキップ。
- **スキュー**: プレミアムは実在(Lempérière et al.: SR ≈ 1/3 − ζ/4)だが
  8資産ではクロスセクションが薄すぎる。マイナーティルト以上にしない。

## 9. アンチパターン(明示的にやらないこと)

| 手法 | 理由 |
|---|---|
| DDブレーキ / CPPI / 個別ストップ | §6。シャープ中立〜マイナス、凸性毀損、二重カウント |
| 1/σ² ボラタイミング(Moreira-Muir型) | リアルタイム実装不能・コスト後消滅(Cederburg 2020; Barroso-Detzel 2021) |
| HRP / NCO | 優位はN大・クラスタ構造前提。N=8では逆ボラ/ERCに対する優位は未確立(矛盾文献あり) |
| ディープラーニング(DMN/Transformer) | 8資産×30年は小データ。文献の優位もコスト2-3bpで消滅、独立再現が乏しい |
| 戦略レベルのモメンタム(エクイティカーブ・トレーディング) | ファクターモメンタムは株式ファクター固有。CTA曲線では効果なし |
| ルックバック/HLの「最良値」選択 | アンサンブル平均で置換(§7)。選ぶならDSRで罰する |
| クロスセクショナル・モメンタム(8資産) | §2.2 |
| ML予測モデル全般の上限 | 認めるのは正則化線形(ridge)か制約付き浅いGBRT(単調性制約・少特徴・シード平均)までで、まず線形と比較して有意差(DSR基準)を要求 |

## 10. 現実的な期待値とKPI

- グロスSR目標: 8資産・トレンド+キャリー(+バリュー)で **0.7-1.0**。
  ネット(コスト後): **0.6-0.8**。これを大きく超えるバックテストはバグか過学習を疑う。
- 報告するのは最終ホールドアウトのネットSRとDSRのみ。あわせて
  ターンオーバー(年率)、コストドラッグ(SR単位)、実現ボラの目標乖離、
  ボラ・オブ・ボラ、左テール(5%ES)を常設KPIとする。
- シャープをさらに上げる正攻法は一つ: **ユニバース拡大**(§1)。

## 11. 主要な論争点・信頼度の注記(誠実性のための開示)

1. **ボラターゲティングの価値**: Harvey et al.(中庸・資産クラス別)と
   Cederburg et al.(ファクターには無効)は整合的に「広義リスク資産のみ有効」で決着。
   Moreira-Muir の楽観的な主張は採用しない。
2. **TSMOM の実在**: Huang et al. (JFE 2020) は資産別予測力の弱さを指摘、
   Kim-Tse-Wald はvol-scaling寄与を指摘。Babu et al. "Trends Everywhere" が
   82+16新市場のOOSで反証。本設計は「トレンド+vol-scalingの複合体」として
   期待値を控えめに置くことで両論に頑健。
3. **CPCV vs WF**: CPCV優位の主証拠は合成環境(外的妥当性に限界)。
   WF擁護は「現実的シミュレーション」論にとどまる。本設計は両方使う(役割分担)。
4. **速度論争**: 長期文献は12ヶ月最強、2015年以降はQuanticaの60-100日HL最強、
   2023年以降は中速も不安定。→ 速度は選ばず混合(これ自体がメタ判断)。
5. 取得できなかった一次資料(403)は複数の独立二次資料で照合済み。
   Baz et al. のパラメータは2件の独立再現論文と一致(高信頼)。
   Garleanu-Pedersen の κ≈0.1-0.25/日 は論文記載値ではなく実務ヒューリスティック。

## 主要参考文献

- Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum", JFE 104.
- Baz, Granger, Harvey, Le Roux, Rattray (2015) "Dissecting Investment Strategies in the Cross Section and Time Series", SSRN 2695101.
- Baltas, Kosowski (2013) "Demystifying Time-Series Momentum Strategies", SSRN 2140091.
- Levine, Pedersen (2016) "Which Trend Is Your Friend?", FAJ 72(3).
- Dao, Nguyen, Deremble, Lempérière, Bouchaud, Potters (2016) "Tail protection for long investors", arXiv:1607.02410.
- Hurst, Ooi, Pedersen (2017) "A Century of Evidence on Trend-Following", JPM.
- Babu, Levine, Ooi, Pedersen, Stamelos (2020) "Trends Everywhere", JOIM.
- Goyal, Jegadeesh (2018) "Cross-Sectional and Time-Series Tests of Return Predictability", RFS 31(5).
- Huang, Li, Wang, Zhou (2020) "Time Series Momentum: Is It There?", JFE 135.
- Kim, Tse, Wald (2016) "Time Series Momentum and Volatility Scaling", JFM 30.
- Koijen, Moskowitz, Pedersen, Vrugt (2018) "Carry", JFE 127(2).
- Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere", JF 68(3).
- Moreira, Muir (2017) "Volatility-Managed Portfolios", JF 72(4).
- Cederburg, O'Doherty, Wang, Yan (2020) "On the Performance of Volatility-Managed Portfolios", JFE 138(1).
- Barroso, Detzel (2021) "Do Limits to Arbitrage Explain the Benefits of Volatility-Managed Portfolios?", JFE 140(3).
- Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, van Hemert (2018) "The Impact of Volatility Targeting", JPM 45(1).
- Bongaerts, Kang, van Dijk (2020) "Conditional Volatility Targeting", FAJ 76(4).
- Garleanu, Pedersen (2013) "Dynamic Trading with Predictable Returns and Transaction Costs", JF 68(6).
- Davis, Norman (1990) "Portfolio Selection with Transaction Costs", Math. OR 15(4).
- Frazzini, Israel, Moskowitz (2018) "Trading Costs", SSRN 3229719.
- Kaminski, Lo (2014) "When Do Stop-Loss Rules Stop Losses?", JFM 18.
- Rattray, Granger, Harvey, van Hemert (2020) "Strategic Rebalancing", JPM 46(6).
- van Hemert et al. (2020) "Drawdowns", SSRN 3583864.
- Bailey, López de Prado (2014) "The Deflated Sharpe Ratio", JPM 40(5).
- Harvey, Liu (2015) "Backtesting", JPM 42(1).
- López de Prado (2018) "Advances in Financial Machine Learning", Wiley.
- Arian, Norouzi, Seco (2024) "Backtest Overfitting in the Machine Learning Era", Knowledge-Based Systems 305 / SSRN 4686376.
- Ledoit, Wolf (2004) "Honey, I Shrunk the Sample Covariance Matrix", JPM 30(4).
- Hansen, Lunde (2005) "A Forecast Comparison of Volatility Models", J. Applied Econometrics 20(7).
- Bollerslev, Hood, Huss, Pedersen (2018) "Risk Everywhere", RFS 31(7).
- Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning", RFS 33(5).
- Avramov, Cheng, Metzker (2023) "Machine Learning vs. Economic Restrictions", Management Science.
- Lim, Zohren, Roberts (2019) "Enhancing Time-Series Momentum Strategies Using Deep Neural Networks", arXiv:1904.04912.
- Lempérière et al. (2014) "Two Centuries of Trend Following", arXiv:1404.3274; (2017) "Risk Premia: Asymmetric Tail Risks and Excess Returns", arXiv:1409.7720.
- Martin (2021) "Design and Analysis of Momentum Trading Strategies", arXiv:2101.01006.
- Carver, R. "Systematic Trading" (2015), "Advanced Futures Trading Strategies" (2023), qoppac.blogspot.com / pysystemtrade.
- Man AHL: "The Need for Speed in Trend-Following Strategies" (2023); Quantica Capital Quarterly Insights ("The Speed Factor" 2025Q4 ほか).
- Fitzgibbons, Friedman, Pomorski, Serban (2017) "Long-Only Style Investing: Don't Just Mix, Integrate", JOI.
- Israel, Jiang, Ross (2017) "Craftsmanship Alpha", JPM.
