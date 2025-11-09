# ./src/rsnn/experiments/evaluation.py
# タイトル: 評価モジュール
# 機能説明: 隠れ層の活動に基づき、Ridge回帰（線形リードアウト）を訓練・評価します。
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge # type: ignore[import-untyped]
from typing import Tuple

class ReadoutEvaluator:
    """Ridge回帰を用いたリードアウトの訓練と評価"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.model: Ridge | None = None
        self.W: np.ndarray | None = None # type: ignore[assignment]

    def train(self, readout_train_X: np.ndarray, train_y: np.ndarray):
        """
        Ridge回帰モデルを訓練します。
        
        Args:
            readout_train_X (np.ndarray): 訓練用の隠れ層活動 (n_samples, n_hidden)
            train_y (np.ndarray): 訓練用ラベル (n_samples,)
        """
        C = int(np.max(train_y) + 1)
        # One-hotエンコーディング
        Y = np.zeros((train_y.size, C))
        Y[np.arange(train_y.size), train_y] = 1.0
        
        self.model = Ridge(alpha=self.alpha, fit_intercept=False)
        self.model.fit(readout_train_X, Y)
        self.W = self.model.coef_ # type: ignore[assignment]

    def evaluate(self, readout_test_X: np.ndarray, test_y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        テストデータで精度を評価します。
        
        Args:
            readout_test_X (np.ndarray): テスト用の隠れ層活動 (n_samples_test, n_hidden)
            test_y (np.ndarray): テスト用ラベル (n_samples_test,)
        
        Returns:
            Tuple[float, np.ndarray]: (精度, 予測ラベル)
        """
        if self.model is None or self.W is None:
            raise RuntimeError("Evaluator must be trained first.")
            
        # スコア計算 (W は (C, n_hidden) または (n_hidden, C) ... sklearnでは (C, n_hidden))
        # W @ H.T -> (C, N)
        scores = self.W @ readout_test_X.T
        preds = np.argmax(scores, axis=0)
        
        acc = (preds == test_y).mean()
        return float(acc), preds
