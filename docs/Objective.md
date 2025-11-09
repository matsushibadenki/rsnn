# **L-RSNN (Liquid Recurrent Spiking Neural Network) プロジェクト 目標達成ロードマップ**

## **1\. プロジェクト概要**

本プロジェクトは、従来の RSNN (Recurrent Spiking Neural Network) のアーキテクチャ \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/README.md\] と、LNN (Liquid Neural Network) / LSM (Liquid State Machine) の設計思想 \[cite: LNN\_RSNN\_design\_spec.md\] を融合させた、次世代のハイブリッドモデル「**L-RSNN**」の確立を目指す。

提示された5つの目標（①高精度, ②低消費電力, ③再現性, ④非GPU, ⑤非BP）\[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\] を達成するため、本ロードマップは特性の異なる2つの主要な学習アプローチを並行して追求する。

* **アプローチA（勾配ベース）:** Surrogate Gradient法 \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md, matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/RSNNプロジェクトの技術的解決策リサーチ.md\] などを用い、ネットワーク全体の重みを最適化し、SOTA（最高水準）の**高精度**（目標①）を追求する。  
* **アプローチB（非勾配・LSM型）:** LNN\_RSNN\_design\_spec.md の設計思想 \[cite: LNN\_RSNN\_design\_spec.md\] に基づき、微分演算を使用しない \[cite: LNN\_RSNN\_design\_spec.md\]。内部状態（Reservoir / Liquid）は局所ルール（STDPなど）で自律的に更新し、学習を出力層（Readout）に集中させることで、**計算安定性**と**低コスト**（目標②, ④）を追求する \[cite: LNN\_RSNN\_design\_spec.md, matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/RSNNプロジェクトの技術的解決策リサーチ.md\]。

## **フェーズ1: 基盤安定化とベンチマーク（1〜3ヶ月）**

**目標:** CIFAR-10を扱える基盤を構築し、目標③（学習再現性）のベースラインを確立する。

### **1.1. データセットの拡張 (CIFAR-10対応)**

* **タスク:** 現在のトイ・データセット (src/rsnn/experiments/dataset.py) \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/src/rsnn/experiments/dataset.py\] に加え、CIFAR-10（またはステップとしてMNIST/Fashion-MNIST）をロードし、スパイク信号に変換する機能を追加する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **担当:** src/rsnn/core/encoding.py, src/rsnn/experiments/dataset.py  
* **ゴール:** CIFAR-10画像をPoisson符号化またはLatency符号化（latency\_burst\_encoding）でネットワークに入力できるようにする \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。

### **1.2. ネットワークアーキテクチャの多層化**

* **タスク:** src/rsnn/core/base\_rsnn.py \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/src/rsnn/core/base\_rsnn.py\] を拡張し、単層隠れ層だけでなく、多層（例：Input \-\> L1(SNN) \-\> L2(SNN) \-\> Readout）のSNNを構築できるようにする \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **担当:** src/rsnn/core/models.py, src/rsnn/di/containers.py  
* **ゴール:** 畳み込みSNN（CSNN）の足掛かりとして、層間の接続と学習ルール（STDP）を定義できるようにする \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。

### **1.3. 再現性テストパイプラインの構築 (目標③)**

* **タスク:** tools/health\_check.py \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/tools/health\_check.py\] や src/rsnn/services/experiment\_service.py \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/src/rsnn/services/experiment\_service.py\] を拡張し、複数シード（例：30回）での学習を自動実行し、精度の平均と標準偏差、および「目標精度（例：80%）に達した割合」をレポートする機能を追加する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **担当:** tools/, src/rsnn/experiments/reporting.py  
* **ゴール:** 学習再現性の定量的ベンチマークを確立する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。

### **1.4. 代替学習ルールの調査・実装 (目標⑤)**

* **タスク:** 現状のSTDP (src/rsnn/core/learning\_rules.py) \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/src/rsnn/core/learning\_rules.py\] に加え、BPを使わない他の学習ルールを調査・実装する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **アプローチA（勾配ベース）の調査:** R-STDP、Tempotronなど \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **アプローチB（非勾配・LSM型）の調査:** LNN\_RSNN\_design\_spec.md のアプローチ \[cite: LNN\_RSNN\_design\_spec.md\] に基づき、Reservoir Computing (Liquid State Machine) \[cite: LNN\_RSNN\_design\_spec.md, matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/RSNNプロジェクトの技術的解決策リサーチ.md\] をプロトタイプ実装する。  
* **担当:** src/rsnn/core/learning\_rules.py  
* **ゴール:** 両アプローチの基本的な実現可能性を検証する。

## **フェーズ2: 精度向上（目標①）と効率化（目標④, ②）**

**目標:** ネットワーク構造と学習則を根本的に強化し、目標①（高精度）の達成を目指す。同時に、目標④・②（非GPU・低消費電力）の基盤を構築する。

### **2.1. 疎なイベント駆動型シミュレータの実装 (目標④, ②)**

* **タスク:** 現状のNumpyベースのシミュレーション（毎ステップ全ニューロンの電圧を計算）から、スパイク（イベント）が発生したニューロンとそれに接続するシナプスのみを計算する「イベント駆動型」シミュレータへ移行する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **担当:** src/rsnn/core/ (新規モジュール)  
* **ゴール:** 行列計算への依存を排除し（目標④）、計算量を大幅に削減する（目標②の基盤） \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。

### **2.2. 畳み込みSNN（CSNN）アーキテクチャの導入**

* **タスク:** フェーズ1.2の多層化を発展させ、畳み込み層（重み共有）とプーリング層（スパイクベース）の概念をNumpyベース（またはイベント駆動型）で実装する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **担当:** src/rsnn/core/models.py  
* **ゴール:** CIFAR-10のような画像タスクにおいて、特徴抽出の効率と精度を向上させる \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。

### **2.3. BP代替学習アルゴリズムの本格導入 (目標⑤)**

BPの代替として有望な2つのアプローチを本格的に実装・比較検討する。

#### **2.3.1 アプローチA: 勾配ベース（Surrogate Gradient法）**

* **タスク:** BPの代替として有望なSurrogate Gradient法（勾配近似）を本格的に実装・比較検討する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **備考:** Surrogate Gradient法はBPに似るが、SNNの文脈では標準的な手法の一つ \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md, matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/RSNNプロジェクトの技術的解決策リサーチ.md\]。  
* **担当:** src/rsnn/core/learning\_rules.py  
* **ゴール:** CIFAR-10において、SOTA（最高水準）の精度を達成する勾配ベース学習法を選定する。

#### **2.3.2 アプローチB: 非勾配・Reservoir型（L-RSNNコンセプト）**

* **タスク:** LNN\_RSNN\_design\_spec.md の設計 \[cite: LNN\_RSNN\_design\_spec.md\] に基づき、微分演算に依存しないReservoir Computing（LSM）アプローチを本格実装する。  
* **メカニズム:**  
  1. **Liquid Reservoir（内部）:** CSNN（2.2）の内部重みをSTDP \[cite: LNN\_RSNN\_design\_spec.md\] や恒常性（Homeostasis）\[cite: LNN\_RSNN\_design\_spec.md, matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/src/rsnn/core/learning\_rules.py\] といった局所ルールのみで（非教師ありで）更新する。あるいはランダムに固定する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/RSNNプロジェクトの技術的解決策リサーチ.md\]。  
  2. **Readout（出力層）:** 学習を「出力層学習（LNN的線形回帰）」\[cite: LNN\_RSNN\_design\_spec.md\]（例：src/rsnn/experiments/evaluation.py のRidge回帰 \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/src/rsnn/experiments/evaluation.py\]）に集中させる。  
* **担当:** src/rsnn/core/models.py, src/rsnn/services/experiment\_service.py  
* **ゴール:** 高速な学習と計算安定性を両立する、微分非依存のアーキテクチャを選定する。

### **2.4. 消費電力（スパイク数）の精密計測 (目標②)**

* **タスク:** src/rsnn/experiments/evaluation.py \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/src/rsnn/experiments/evaluation.py\] に、単なる精度（acc）だけでなく、推論1回あたりの総スパイク数、平均スパイクレートを計測する機能を追加する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **担当:** src/rsnn/experiments/evaluation.py, src/rsnn/services/experiment\_service.py  
* **ゴール:** モデルの「消費電力（＝スパイク数）」を定量化し、最適化の指標とする \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。

## **フェーズ3: 統合と目標実証（3〜6ヶ月）**

**目標:** 全ての目標（①〜⑤）を同時に達成するモデルと構成を実証する。

### **3.1. ハイパーパラメータ自動最適化**

* **タスク:** フェーズ2で構築したアプローチA（勾配ベース）とアプローチB（非勾配・LSM型）の多数のパラメータ（config/experiment\_params.json \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/config/experiment\_params.json\]）を効率的に最適化するため、scikit-learnのGridSearchCVやOptunaなどのベイズ最適化ツールと連携する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **担当:** src/main.py, scripts/ (新規)  
* **ゴール:** 「精度（アプローチA）」「低消費電力・安定性（アプローチB）」のバランスが取れた最適パラメータを発見する。

### **3.2. 精度ベンチマーク（CIFAR-10） (目標①)**

* **タスク:** 最適化されたモデル（3.1）を使用し、CIFAR-10での認識精度 95〜97% の達成を実証する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **ゴール:** 目標①の達成。

### **3.3. 消費電力（演算）シミュレーション (目標②)**

* **タスク:** イベント駆動シミュレータ（2.1）で計測した演算量（SNN）と、同等の精度を持つANN（例：小規模CNN）のFLOPs（浮動小数点演算回数）を比較・試算する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **ゴール:** SNNの演算量がANN比で1/50以下（またはそれ以下）となることを実証する（目標②） \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。

### **3.4. 最終再現性検証 (目標③)**

* **タスク:** 最終モデル（3.1、特に安定性が期待されるアプローチB）に対し、フェーズ1.3のパイプラインを用いて大規模な再現性テスト（例：100シード）を実行する \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。  
* **ゴール:** 学習成功率（例：精度95%以上を達成する確率）が95%以上であることを実証する（目標③） \[cite: matsushibadenki/rsnn/rsnn-97cbb891138d8bafea4e4e6a7201d675727d7d89/docs/Objective.md\]。
