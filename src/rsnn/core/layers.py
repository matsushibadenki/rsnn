# ./src/rsnn/core/layers.py
# タイトル: SNN層モジュール
# 機能説明: SNNの基本的な層（LIFニューロン層）のロジックを定義します。
#           (Objective 1.2 / 2.2 の基盤)
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

class BaseLayer(ABC):
    """SNN層の抽象基底クラス"""
    
    @abstractmethod
    def __call__(self, input_current: np.ndarray) -> np.ndarray:
        """
        1タイムステップの順伝播を実行します。
        
        Args:
            input_current (np.ndarray): この層への入力電流
        
        Returns:
            np.ndarray: この層の出力スパイク (0.0 or 1.0)
        """
        pass

    @abstractmethod
    def reset(self):
        """層の状態（電圧など）をリセットします。"""
        pass

class LIFLayer(BaseLayer):
    """
    Leaky Integrate-and-Fire (LIF) ニューロン層。
    ニューロンのダイナミクス（電圧更新、発火、リセット）をカプセル化します。
    """
    
    def __init__(self, 
                 shape: tuple[int, ...], 
                 dt: float, 
                 tau_m: float, 
                 v_th: float, # 修正: 初期値はfloat
                 v_reset: float):
        
        self.shape = shape
        self.dt = dt
        self.tau_m = tau_m
        
        # 修正: v_th を適応的 (ndarray) も許容するように変更
        self.v_th_base: float | np.ndarray = v_th
        self.v_th: float | np.ndarray = v_th_base 
        
        self.v_reset = v_reset
        
        # 電圧減衰と入力スケールの係数
        self.decay_m = (1.0 - self.dt / self.tau_m)
        self.scale_m = (self.dt / self.tau_m)
        
        # 状態変数
        self.V: np.ndarray = np.zeros(shape)
        self.spikes: np.ndarray = np.zeros(shape)

    def __call__(self, I_in: np.ndarray) -> np.ndarray:
        """
        1タイムステップのLIFダイナミクスを実行します。
        
        Args:
            I_in (np.ndarray): 入力電流 (形状は self.shape と一致)
        
        Returns:
            np.ndarray: 出力スパイク (self.shape)
        """
        if I_in.shape != self.shape:
            raise ValueError(f"Input current shape {I_in.shape} must match layer shape {self.shape}")
            
        # 1. 電圧更新 (LIF)
        self.V = self.V * self.decay_m + I_in * self.scale_m
        
        # 2. スパイク判定 (v_thがfloatでもndarrayでも動作)
        self.spikes = (self.V >= self.v_th).astype(float)
        
        # 3. リセット
        self.V = np.where(self.spikes > 0, self.v_reset, self.V)
        
        return self.spikes

    def reset(self):
        """電圧とスパイクをリセットし、閾値をベース値に戻します。"""
        self.V = np.zeros(self.shape)
        self.spikes = np.zeros(self.shape)
        # 修正: 閾値をベース値に戻す
        self.v_th = self.v_th_base
