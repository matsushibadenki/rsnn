# ./src/rsnn/core/base_rsnn.py
# タイトル: RSNN基底クラス
# 機能説明: RSNNモデルの共通インターフェースと状態（電圧、重み）を定義します。
# 抽象クラスとして設計し、具体的なシミュレーションステップはサブクラスで実装します。
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple
# 修正: 新しいLIFLayerをインポート (Objective 1.2)
from .layers import LIFLayer

class BaseRSNN(ABC):
    """RSNNモデルの抽象基底クラス"""
    
    def __init__(self, n_input: int, n_hidden: int,
                 dt: float, tau_m: float, v_th: float, v_reset: float,
                 rec_delay: int, rng_seed: int):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.dt = dt
        self.rec_delay = rec_delay
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        
        # 修正: LIF層の初期化 (Objective 1.2)
        # 電圧(V)、減衰(decay_m)などの状態はLIFLayerが管理する
        self.lif_layer = LIFLayer(
            shape=(n_hidden,),
            dt=dt,
            tau_m=tau_m,
            v_th=v_th,
            v_reset=v_reset
        )
        
        # 修正: v_th_base などのパラメータをLIFLayerから取得
        self.v_th_base = self.lif_layer.v_th_base # 修正: v_th -> v_th_base
        self.v_reset = self.lif_layer.v_reset
        
        # 修正: 状態変数を BaseRSNN から削除 (LIFLayerに移譲)
        # self.decay_m = (1.0 - self.dt / self.tau_m)
        # self.scale_m = (self.dt / self.tau_m)
        
        # 入力重み (W)
        self.W = self.rng.normal(0.5, 0.1, size=(n_hidden, n_input)).clip(min=0.0)
        
        # リカレント重み (U)
        density = 0.12
        mask = self.rng.random((n_hidden, n_hidden)) < density
        self.U = (self.rng.normal(0.15, 0.04, size=(n_hidden, n_hidden)) * mask).clip(min=0.0)
        np.fill_diagonal(self.U, 0.0)

    @abstractmethod
    def run_sample(self, rates: np.ndarray, T: int,
                   encoding_fn: Callable[..., np.ndarray],
                   encoding_params: dict,
                   train_stdp: bool = True) -> np.ndarray:
        """
        単一のサンプルをネットワークで実行します。
        
        Args:
            rates (np.ndarray): 入力レート (n_input,)
            T (int): タイムステップ数
            encoding_fn (Callable): 使用するエンコーディング関数
            encoding_params (dict): エンコーディング関数への引数
            train_stdp (bool): STDP学習を実行するかどうか
            
        Returns:
            np.ndarray: 隠れ層のスパイク記録 (T, n_hidden)
        """
        pass

    def collect_hidden_activity(self, samples_rates: np.ndarray, T: int,
                                encoding_fn: Callable[..., np.ndarray],
                                encoding_params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        複数のサンプルを実行し、隠れ層の平均発火率と総スパイク数を収集します。
        
        Args:
            samples_rates (np.ndarray): 入力レート行列 (n_samples, n_input)
            T (int): タイムステップ数
            encoding_fn (Callable): エンコーディング関数
            encoding_params (dict): エンコーディング関数への引数
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                (平均発火率行列 (n_samples, n_hidden), 
                 サンプル毎の総スパイク数 (n_samples,))
        """
        n_samples = samples_rates.shape[0]
        hidden_activity = np.zeros((n_samples, self.n_hidden))
        total_spikes_per_sample = np.zeros(n_samples)
        
        for i in range(n_samples):
            # STDP学習はオフにして実行
            rec = self.run_sample(samples_rates[i], T, encoding_fn, encoding_params, train_stdp=False)
            
            hidden_activity[i] = rec.sum(axis=0) / T
            total_spikes_per_sample[i] = rec.sum()
            
        return hidden_activity, total_spikes_per_sample
