# ./src/rsnn/experiments/dataset.py
# タイトル: データセットモジュール（CIFAR-10対応拡張）
# 機能説明: 合成的なレートベースのデータセット（トイ・データセット）を作成します。
#           また、CIFAR-10データセットをダウンロードし、Numpy形式で提供します。
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Any
import os 

# --- 修正: CIFAR-10対応のためのインポート ---

try:
    import torch # type: ignore[import-untyped]
    import torchvision # type: ignore[import-untyped]
    import torchvision.transforms as transforms # type: ignore[import-untyped]
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: torchvision not installed. CIFAR10Loader will not be available.")
    TORCH_AVAILABLE = False
    # 修正: インポート失敗時 (exceptブロック内) で Any を定義
    torch: Any = None
    torchvision: Any = None
    transforms: Any = None
# --- 修正ここまで ---


class DatasetGenerator:
    """トイ・データセット（レートベース）を生成します。"""
    
    def __init__(self, n_input: int, pattern_size: int, 
                 base_rate: float, pat_rate: float, rng_seed: int):
        self.n_input = n_input
        self.pattern_size = pattern_size
        self.base_rate = base_rate
        self.pat_rate = pat_rate
        self.rng = np.random.default_rng(rng_seed)

    def make_toy_rates(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        合成レートデータセットを作成します。
        クラス0: F 'pattern_size'ニューロンが高レート
        クラス1: 最後の'pattern_size'ニューロンが高レート
        """
        rates = np.ones((n_samples, self.n_input)) * self.base_rate
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            cls = i % 2
            labels[i] = cls
            if cls == 0:
                rates[i, :self.pattern_size] = self.pat_rate
            else:
                rates[i, -self.pattern_size:] = self.pat_rate
                
        return rates, labels

# --- 修正: Objective 1.1 に基づき CIFAR10Loader を追加 ---
class CIFAR10Loader:
    """
    CIFAR-10データセットをダウンロードし、Numpy配列としてロードします。
    (Objective 1.1)
    """
    def __init__(self, root_dir: str = "./data"):
        if not TORCH_AVAILABLE:
            raise ImportError("torchvision is required for CIFAR10Loader. Please install it.")
        
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -1から1の範囲に正規化
        ])
        
        # データセットのダウンロード
        self.train_dataset = self._get_dataset(train=True)
        self.test_dataset = self._get_dataset(train=False)

    def _get_dataset(self, train: bool):
        try:
            return torchvision.datasets.CIFAR10(
                root=self.root_dir, 
                train=train,
                download=True, 
                transform=self.transform
            )
        except Exception as e:
            print(f"Failed to download CIFAR-10: {e}")
            print("Please check your internet connection and permissions.")
            return None

    def load_data(self, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        CIFAR-10データをNumpy配列としてロードします。
        
        Args:
            n_samples (Optional[int]): ロードするサンプル数（Noneの場合は全件）

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (train_images, train_labels, test_images, test_labels)
                画像は (N, C, H, W) のNumpy配列
        """
        if self.train_dataset is None or self.test_dataset is None:
            raise RuntimeError("CIFAR-10 dataset not loaded. Download failed.")

        # 訓練データ
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)
        train_images_tensor, train_labels_tensor = next(iter(train_loader))
        
        # テストデータ
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)
        test_images_tensor, test_labels_tensor = next(iter(test_loader))

        # Numpyに変換
        train_images = train_images_tensor.numpy()
        train_labels = train_labels_tensor.numpy()
        test_images = test_images_tensor.numpy()
        test_labels = test_labels_tensor.numpy()

        if n_samples is not None:
            train_images = train_images[:n_samples]
            train_labels = train_labels[:n_samples]
            test_images = test_images[:n_samples]
            test_labels = test_labels[:n_samples]
            
        return train_images, train_labels, test_images, test_labels

    def load_data_flattened(self, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        CIFAR-10データをフラット化されたNumpy配列 (N, 3072) としてロードします。
        """
        tr_img, tr_lbl, te_img, te_lbl = self.load_data(n_samples)
        
        tr_img_flat = tr_img.reshape(tr_img.shape[0], -1) # (N, 3*32*32)
        te_img_flat = te_img.reshape(te_img.shape[0], -1)
        
        return tr_img_flat, tr_lbl, te_img_flat, te_lbl
