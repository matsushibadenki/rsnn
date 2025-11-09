# ./src/rsnn/core/encoding.py
# タイトル: スパイクエンコーディングモジュール
# 機能説明: レートベースの入力をスパイク列に変換する関数（Poisson符号化、Latency+Burst符号化）を提供します。
#           また、画像データをレートに変換するヘルパーも提供します。
from __future__ import annotations
import numpy as np

def image_to_rates(image_vector: np.ndarray, min_rate: float = 0.0, max_rate: float = 50.0) -> np.ndarray:
    """
    正規化された画像ベクトル（例: -1.0～1.0）を発火率のベクトルにスケーリングします。
    (Objective 1.1 CIFAR-10対応)
    
    Args:
        image_vector (np.ndarray): 入力画像ベクトル (ピクセル値)。
                                   [-1.0, 1.0] (torchvisionのNormalize) または 
                                   [0.0, 1.0] (torchvisionのToTensor) の範囲を想定。
        min_rate (float): 最小発火率 (Hz)
        max_rate (float): 最大発火率 (Hz)

    Returns:
        np.ndarray: スケーリングされた発火率ベクトル
    """
    # 入力ベクトルの最小値と最大値を確認
    min_val = image_vector.min()
    max_val = image_vector.max()
    
    # 0-1の範囲に正規化
    if min_val < -0.1:
        # [-1.0, 1.0] の範囲と仮定 ( (x + 1) / 2 )
        normalized = (image_vector + 1.0) / 2.0
    elif min_val >= 0.0:
        # [0.0, 1.0] の範囲と仮定
        if max_val > 1.0:
            # [0, 255] の場合は [0, 1] にスケーリング
            normalized = image_vector / 255.0
        else:
            normalized = image_vector
    else:
        # 不明な範囲（[-0.1, X]など）の場合は、クリッピングして0-1に
        normalized = np.clip(image_vector, 0.0, 1.0)

    # 0-1の範囲を [min_rate, max_rate] にスケーリング
    rates = normalized * (max_rate - min_rate) + min_rate
    return np.clip(rates, min_rate, max_rate)


def poisson_encoding(rate_vector: np.ndarray, dt: float, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    シンプルなPoissonスパイクジェネレータ（タイムステップ毎）。
    
    Args:
        rate_vector (np.ndarray): 入力レート (n_input,) (Hz)
        dt (float): タイムステップ（秒）
        T (int): 総タイムステップ数
        rng (np.random.Generator): 乱数生成器
    
    Returns:
        np.ndarray: スパイク行列 (T, n_input)
    """
    # dt=0.001 (1ms), rate=100 (Hz) -> p = 100 * 0.001 = 0.1 (10%の確率で発火)
    p = np.clip(rate_vector * dt, 0.0, 1.0)
    return rng.random((T, rate_vector.size)) < p

def latency_burst_encoding(rate_vector: np.ndarray, T: int, rng: np.random.Generator,
                           burst_prob: float = 0.6, burst_len: int = 2) -> np.ndarray:
    """
    Latency + Burst 符号化。
    高レートほど早いスパイクを生成し、オプションでバースト（連続スパイク）を発生させます。
    
    Args:
        rate_vector (np.ndarray): 入力レート (n_input,) (Hz)
        T (int): 総タイムステップ数
        rng (np.random.Generator): 乱数生成器
        burst_prob (float): バースト発生確率
        burst_len (int): バースト長（初期スパイクを除く追加スパイク数）
    
    Returns:
        np.ndarray: スパイク行列 (T, n_input)
    """
    mins = rate_vector.min()
    maxs = rate_vector.max()
    
    # タイムステップマージン（T-1で発火するとバーストできないため）
    time_margin = max(5, burst_len + 2)
    
    if maxs == mins or maxs <= 0:
        # レートが全て同じ、または発火なしの場合、(T - margin) ステップ目に発火（または発火しない）
        times = np.full(rate_vector.size, T - time_margin, dtype=int)
        # レートが0以下の場合は発火させない
        times[rate_vector <= 0] = T + 10 # 範囲外に設定
    else:
        # 正規化 (0.0 - 1.0)
        # 最小レートをオフセットとして扱う
        norm = (rate_vector - mins) / (maxs - mins)
        # 高レート(norm=1) -> 0, 低レート(norm=0) -> (T - margin)
        times = ((1.0 - norm) * (T - time_margin)).astype(int)
        # レートが0以下の場合は発火させない
        times[rate_vector <= 0] = T + 10 # 範囲外に設定
        
    spikes = np.zeros((T, rate_vector.size), dtype=bool)
    
    for i, t in enumerate(times):
        if t < 0: t = 0
        
        # Tの範囲内でのみ発火
        if t < T:
            spikes[t, i] = True
            
            # バーストの適用
            if rng.random() < burst_prob:
                for b in range(1, burst_len + 1):
                    if t + b < T:
                        spikes[t + b, i] = True
                    
    return spikes
