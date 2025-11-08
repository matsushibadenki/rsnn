# ./src/main.py
# タイトル: プロジェクト実行エントリポイント
# 機能説明: DIコンテナを初期化し、設定ファイルを読み込み、
#           LangChainで定義された実験パイプラインを実行します。
from __future__ import annotations
import os
import sys

# Pythonパスに 'src' を追加
sys.path.append(os.path.dirname(__file__))

from rsnn.di.containers import ApplicationContainer
from rsnn.experiments.chains import ExperimentChains

def main(config_path: str):
    """
    メイン実行関数
    1. DIコンテナの初期化
    2. 設定ファイルのロード
    3. LangChain Chainの取得
    4. Chainの実行
    """
    
    print("--- RSNN DI + LangChain Project ---")
    
    # 1. DIコンテナの初期化
    # (コンテナ自体はシングルトンとして振る舞う)
    container = ApplicationContainer()
    
    # 2. 設定ファイルのロード
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        # 修正: 設定ファイルが見つからない場合は非ゼロで終了
        sys.exit(1)
        
    print(f"Loading config from: {config_path}")
    container.config.from_json(config_path) # type: ignore[attr-defined]
    
    # 3. LangChain Chainの取得
    # (ChainはDIコンテナをクロージャとして保持)
    experiment_runner = ExperimentChains(container)
    full_chain = experiment_runner.get_full_experiment_chain()
    
    print("Initializing experiment chain...")
    
    # 4. Chainの実行
    # 入力として設定辞書（または空の辞書）を渡す
    # Chain内部でDIコンテナから設定を参照する
    try:
        print("Invoking chain...")
        # configを渡すが、実際にはコンテナにロード済みのものを使用
        initial_input = container.config() # type: ignore[attr-defined]
        
        final_summary = full_chain.invoke(initial_input)
        
        print("\n--- Experiment finished successfully ---")
        print(f"Summary JSON saved to: {final_summary['config']['output_paths']['output_dir']}")
        print(f"README saved to: {final_summary['config']['output_paths']['output_dir']}")
        
    except Exception as e:
        print(f"\n--- Experiment failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        # 修正: エラーが発生したら、非ゼロのステータスコードで終了する
        sys.exit(1)

if __name__ == "__main__":
    # configファイルへのパスを指定
    # (このスクリプト (src/main.py) から見て ../config/experiment_params.json)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 修正: '..' を1つ削除し、プロジェクトルートを正しく指すように変更
    default_config_path = os.path.join(base_dir, "..", "config", "experiment_params.json")
    
    # 正規化してパスを明確にする (オプションだが堅牢性が増す)
    default_config_path = os.path.normpath(default_config_path)

    main(config_path=default_config_path)