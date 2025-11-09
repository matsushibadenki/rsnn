# **L-RSNN プロジェクト**

このプロジェクトは、既存のRSNN（Recurrent Spiking Neural Network）の実験コード（STDP、Homeostasis、E/I分離、Latency符号化など）を、依存性注入（DI）コンテナとLangChain（LCEL）を使用して再構築したものです。

目的は、関心事の分離（SoC）を徹底し、コンポーネント（モデル、学習ルール、データ、評価）の結合を疎にし、実験パイプラインのオーケストレーションを柔軟に行えるようにすることです。

## **1\. 主な技術スタック**

* **Python 3.10+** \* **依存性注入 (DI):** dependency-injector  
  * モデル、サービス、設定などの依存関係を一元管理し、注入します。  
* **オーケストレーション:** langchain-core  
  * LCEL (LangChain Expression Language) を使用して、実験の実行シーケンス（設定 \-\> 複数実験の並列実行 \-\> レポート生成）を定義します。  
* **SNNシミュレーション:** numpy  
  * LIFニューロンモデル、STDP、恒常性などのコアロジックを実装します。  
* **評価:** scikit-learn  
  * 隠れ層の活動を評価するための線形リードアウト（Ridge回帰）に使用します。

## **2\. ディレクトリ構成**

プロジェクトの全体像は以下の通りです。

rsnn\_langchain\_di\_project/

├── config/

│ └── experiment\_params.json \# 実験パラメータ定義 (JSON)

├── requirements.txt \# 必要なライブラリ

├── tools/  
│ └── health\_check.py \# (ヘルスチェック本体)  
├── scripts/  
│ ├── run\_health\_check.sh \# (ヘルスチェック実行ラッパー)  
│ └── visualize\_full\_health.py \# (ヘルスチェック可視化)  
└── src/  
├── rsnn/

│ ├── core/ \# RSNNのコアロジック

│ │ ├── base\_rsnn.py \# (RSNN基底クラス)

│ │ ├── encoding.py \# (Poisson, Latency符号化)

│ │ ├── learning\_rules.py \# (STDP, Homeostasisロジック)

│ │ └── models.py \# (RSNN\_Homeo, RSNN\_EI実装)

│ ├── di/ \# 依存性注入

│ │ └── containers.py \# (DIコンテナ定義)

│ ├── experiments/ \# 実験関連

│ │ ├── chains.py \# (LangChainによる実験パイプライン)

│ │ ├── dataset.py \# (データセット生成)

│ │ ├── evaluation.py \# (Ridge回帰による評価)

│ │ └── reporting.py \# (結果の保存、README生成)

│ └── services/ \# ビジネスロジック/サービス

│ └── experiment\_service.py \# (実験実行サービス)

└── main.py \# 実行エントリポイント

## **3\. 主要コンポーネントの説明**

### **src/rsnn/core/ (コアロジック)**

* **base\_rsnn.py**: BaseRSNN 抽象クラスを定義。  
* **models.py**: RSNN\_Homeo (STDP \+ 恒常性) と RSNN\_EI (E/I分離 \+ k-Winners) の具体的なモデル実装。  
* **learning\_rules.py**: STDP と Homeostasis のロジックをカプセル化したクラス。  
* **encoding.py**: poisson\_encoding と latency\_burst\_encoding 関数を提供。

### **src/rsnn/experiments/ (実験コンポーネント)**

* **dataset.py**: DatasetGenerator がトイ・データセットを生成。  
* **evaluation.py**: ReadoutEvaluator がRidge回帰による訓練と評価を実行。  
* **reporting.py**: ResultReporter が結果をJSONとMarkdownで保存。

### **src/rsnn/di/containers.py (依存性注入)**

* dependency\_injector を使用して、ApplicationContainer を頂点とするDIコンテナを定義します。  
* CoreContainer, ModelsContainer, ExperimentContainer, ServicesContainer に分割され、それぞれが担当するコンポーネントの生成と注入（providers.Factory, providers.Singleton）を管理します。  
* config プロバイダを通じて、config/experiment\_params.json の設定値を各コンポーネントに注入します。

### **src/rsnn/services/experiment\_service.py (サービス)**

* ExperimentService は、DIコンテナから DatasetGenerator や ReadoutEvaluator を注入されます。  
* run\_experiment メソッドは、注入されたモデルとデータ、エンコーディング設定に基づき、STDP訓練、活動収集、リードアウト評価の一連の流れを実行する責務を持ちます。

### **src/rsnn/experiments/chains.py (LangChainパイプライン)**

* ExperimentChains クラスが、LCEL (LangChain Expression Language) を用いて実験フローを定義します。  
* DIコンテナ（container）を保持し、RunnableLambda を介して ExperimentService や ResultReporter のメソッドを呼び出します。  
* get\_full\_experiment\_chain は、複数の実験（Homeo+Poisson, Homeo+Latency, EI+Poisson）を並列実行（RunnablePassthrough.assign）し、最後に結果をレポートする（\_generate\_report）一連のChainを構築します。

## **4\. 実行フロー (LangChainによる制御)**

1. **起動**: python src/main.py が実行されます。  
2. **DIコンテナ初期化**: main.py 内で ApplicationContainer がインスタンス化されます。  
3. **設定ロード**: container.config.from\_json() が config/experiment\_params.json を読み込み、DIコンテナの設定プロバイダを更新します。  
4. **Chain構築**: ExperimentChains(container) がインスタンス化され、get\_full\_experiment\_chain() が呼び出されます。  
5. **Chain実行**: full\_chain.invoke(initial\_input) が呼び出され、LCELパイプラインがスタートします。  
6. **並列実行**: LCELの RunnablePassthrough.assign により、以下の実験が（論理的に）並列で実行されます。  
   * \_run\_homeo\_poisson (Homeoモデル \+ Poisson符号化)  
   * \_run\_homeo\_latency (Homeoモデル \+ Latency符号化)  
   * \_run\_ei\_poisson (EIモデル \+ Poisson符号化)  
7. **結果集約とレポート**: 全ての実験が完了すると、それらの結果（Dict）が最後の RunnableLambda(\_generate\_report) に渡されます。  
8. **保存**: ResultReporter が最終的なサマリを outputs/ ディレクトリにJSON (rsnn\_summary\_di\_lc.json) と Markdown (README\_rsnn\_di\_lc.md) として保存します。

## **5\. 実行方法**

1. 依存ライブラリのインストール: \`\`\`bash  
   pip install \-r requirements.txt

2. 実験の実行: \`\`\`bash  
   python src/main.py  
   \* src/main.py は、config/experiment\\\_params.json を自動的に読み込み、LangChainパイプラインを実行します。    
   \* 実行が完了すると、結果が outputs/ ディレクトリに生成されます。

## **6\. 設定の変更**

実験パラメータ（隠れ層のニューロン数、学習率、エポック数、シード値など）は、config/experiment\_params.json ファイルを編集することで一元的に変更できます。

## **7\. ヘルスチェック (環境確認)**

プロジェクトの実行環境が正しくセットアップされているかを確認するためのヘルスチェックスクリプトが提供されています。

このスクリプトは以下の項目をチェックします:

1. Python環境および requirements.txt に基づくパッケージのインストールの確認。  
2. DIコンテナの初期化と設定ファイルの読み込み（スモークテスト）。  
3. src/main.py のサブプロセス実行による実験パイプラインの動作確認（時間がかかる場合があります）。  
4. 上記の結果に基づき、レポート（JSON）と可視化グラフ（PNG）を outputs/ ディレクトリに生成します。

### **実行方法**

scripts/run\_health\_check.sh を実行してください。

\# スクリプトに実行権限を付与 (初回のみ)  
chmod \+x scripts/run\_health\_check.sh

\# ヘルスチェックの実行  
./scripts/run\_health\_check.sh

### **生成されるファイル**

ヘルスチェックを実行すると、outputs/ ディレクトリ（config/experiment\_params.json で指定された出力先）に以下のファイルが生成されます。

* outputs/health\_report.json: チェック結果（パッケージバージョン、テスト成否など）を含むJSONレポート。  
* outputs/full\_health\_check.png: health\_report.json の内容を可視化したグラフ画像。
