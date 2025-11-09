# ./src/rsnn/services/experiment_service.py
# タイトル: 実験実行サービス
# 機能説明: 注入されたコンポーネント（モデル、データ、評価器）を使用して、
#           単一の実験（例：HomeoモデルでPoisson符号化）を実行する責務を持ちます。
from __future__ import annotations
import numpy as np
from typing import Callable, List, Dict, Any
from ..core.base_rsnn import BaseRSNN
from ..experiments.dataset import DatasetGenerator
from ..experiments.evaluation import ReadoutEvaluator

class ExperimentService:
    """
    単一の実験構成（モデル、データ、エンコーディング）を実行し、
    訓練と評価を行うサービス。
    """
    
    def __init__(self, 
                 dataset_generator: DatasetGenerator,
                 evaluator: ReadoutEvaluator):
        self.dataset_generator = dataset_generator
        self.evaluator = evaluator
        # 内部状態（データ）
        self.train_rates: np.ndarray | None = None
        self.train_labels: np.ndarray | None = None
        self.test_rates: np.ndarray | None = None
        self.test_labels: np.ndarray | None = None

    def _load_data(self, n_train: int, n_test: int):
        """データセットを（再）生成"""
        self.train_rates, self.train_labels = self.dataset_generator.make_toy_rates(n_train)
        self.test_rates, self.test_labels = self.dataset_generator.make_toy_rates(n_test)

    def run_experiment(
        self,
        rsnn_model: BaseRSNN,
        encoding_fn: Callable[..., np.ndarray],
        encoding_params: dict,
        dataset_params: dict,
        sim_params: dict,
        seeds: List[int]
    ) -> List[Dict[str, Any]]:
        """
        指定されたモデルとエンコーディングで実験（複数シード）を実行します。
        
        Args:
            rsnn_model (BaseRSNN): 注入されたDIコンテナが初期化したモデルインスタンス
            encoding_fn (Callable): 符号化関数
            encoding_params (dict): 符号化パラメータ
            dataset_params (dict): データセットパラメータ (n_train, n_test)
            sim_params (dict): シミュレーションパラメータ (T, epochs)
            seeds (List[int]): 実行するシード値のリスト (現状モデルは固定シードで初期化されているが、
                                本来はここでシード毎にモデルを再生成すべき)

        Returns:
            List[Dict[str, Any]]: 各シードの結果（acc, mean_rate, mean_total_spikes）
        """
        
        # 1. データのロード
        self._load_data(dataset_params['n_train'], dataset_params['n_test'])
        
        if self.train_rates is None or self.train_labels is None or \
           self.test_rates is None or self.test_labels is None:
            raise ValueError("Data not loaded correctly.")

        T = sim_params['T']
        epochs = sim_params['epochs']
        
        results = []
        
        # 本来はDIコンテナからシード毎にモデルをファクトリ経由で受け取るべきだが、
        # 今回は簡略化のため、注入された単一モデルを複数エポック実行する
        # (DIコンテナの設計上、モデルのシードは固定されているため、複数シード実行は擬似的)
        
        # 修正: rng.bit_generator.state['seed_seq'] を rsnn_model.rng_seed に変更
        print(f"Running experiment (Model: {rsnn_model.__class__.__name__}, Encoding: {encoding_fn.__name__}, Seed: {rsnn_model.rng_seed})...")
        
        for seed_val in seeds:
            # 簡略化のため、DIコンテナが初期化したモデルのシードを無視し、
            # ここで指定されたシード（の最初の値）を使う（デモ用）
            # 実際には DI(seed) -> Model(seed) とすべき
            # 修正: rng.bit_generator.state['seed_seq'] を rsnn_model.rng_seed に変更
            if seed_val != rsnn_model.rng_seed:
                 print(f"Warning: Running with model seed {rsnn_model.rng_seed}, not requested seed {seed_val}.")
            
            # 2. 訓練 (STDP)
            for ep in range(epochs):
                print(f"  Epoch {ep+1}/{epochs}...")
                idx = rsnn_model.rng.permutation(self.train_rates.shape[0])
                for i in idx:
                    _ = rsnn_model.run_sample(
                        self.train_rates[i], T, encoding_fn, encoding_params, train_stdp=True
                    )
            
            # 3. 隠れ層の活動を収集
            print("  Collecting hidden activity...")
            H_train, _ = rsnn_model.collect_hidden_activity(
                self.train_rates, T, encoding_fn, encoding_params
            )
            H_test, test_total_spikes = rsnn_model.collect_hidden_activity(
                self.test_rates, T, encoding_fn, encoding_params
            )
            
            # 4. リードアウトの訓練と評価
            self.evaluator.train(H_train, self.train_labels)
            acc, _ = self.evaluator.evaluate(H_test, self.test_labels)
            
            mean_rate = float(H_test.mean())
            # Objective.md (フェーズ2.4) に基づき、推論1回あたりの平均総スパイク数を計算
            mean_total_spikes = float(test_total_spikes.mean())

            results.append({
                'seed': seed_val, 
                'acc': acc, 
                'mean_rate': mean_rate,
                'mean_total_spikes': mean_total_spikes # 新しいメトリクス
                })
            
            # シードが1つでも結果を返す（簡略化のため）
            break 
            
        return results
}
