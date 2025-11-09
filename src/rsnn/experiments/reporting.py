# ./src/rsnn/experiments/reporting.py
# タイトル: レポーティングモジュール
# 機能説明: 実験結果をJSONおよびMarkdownファイルとして保存します。
from __future__ import annotations
import os
import json
import numpy as np # 修正: 統計計算のため
from typing import Any, Dict, List

class ResultReporter:
    """実験結果のシリアライズと保存"""
    
    def __init__(self, output_dir: str, summary_filename: str, readme_filename: str):
        self.output_dir = output_dir
        self.summary_path = os.path.join(output_dir, summary_filename)
        self.readme_path = os.path.join(output_dir, readme_filename)
        os.makedirs(self.output_dir, exist_ok=True)

    def save_json_summary(self, summary_data: Dict[str, Any]):
        """
        サマリデータをJSONファイルに保存します。
        
        Args:
            summary_data (Dict[str, Any]): 保存するデータ
        """
        try:
            # 修正: numpy値をネイティブ型に変換するハンドラ
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.floating, np.bool_)):
                        return obj.item()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            with open(self.summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            print(f"JSON summary saved to: {self.summary_path}")
        except IOError as e:
            print(f"Error saving JSON: {e}")

    def generate_readme(self, summary_data: Dict[str, Any]):
        """
        サマリデータからREADME.mdを自動生成します。
        
        Args:
            summary_data (Dict[str, Any]): サマリデータ
        """
        lines = []
        lines.append("# RSNN DI + LangChain 実験まとめ\n")
        lines.append("このドキュメントは、DIコンテナとLangChainを使用して再構築されたRSNN実験のまとめです。\n")
        
        if 'run_timestamp' in summary_data:
            lines.append(f"**実行日時**: {summary_data['run_timestamp']}\n")
            
        lines.append("## 主な結果 (詳細)\n")
        
        result_headers = ['seed', 'acc', 'mean_rate', 'mean_total_spikes']
        all_stats = [] # 修正: 統計サマリー用

        homeo_results = summary_data.get('homeo_poisson_results', [])
        if homeo_results:
            self._append_results_table(lines, "Homeo (Poisson)", homeo_results, result_headers)
            all_stats.append(("Homeo (Poisson)", self._calculate_stats(homeo_results, result_headers[1:])))

        latency_results = summary_data.get('homeo_latency_results', [])
        if latency_results:
            self._append_results_table(lines, "Homeo (Latency)", latency_results, result_headers)
            all_stats.append(("Homeo (Latency)", self._calculate_stats(latency_results, result_headers[1:])))

        ei_results = summary_data.get('ei_poisson_results', [])
        if ei_results:
            self._append_results_table(lines, "E/I (Poisson)", ei_results, result_headers)
            all_stats.append(("E/I (Poisson)", self._calculate_stats(ei_results, result_headers[1:])))

        # 修正: 統計サマリーセクションの追加 (Objective 1.3)
        if all_stats:
            lines.append("\n## 統計サマリー (複数シード実行)\n")
            lines.append(f"実行シード数: {len(homeo_results) if homeo_results else 'N/A'}\n")
            
            stat_headers = ['Metric', 'Mean', 'Std.Dev', 'Min', 'Max']
            
            for title, stats_dict in all_stats:
                lines.append(f"\n### 🔹 {title}\n")
                lines.append(f"| {' | '.join(stat_headers)} |")
                lines.append(f"|{'---|' * len(stat_headers)}")
                for key in result_headers[1:]: # 'seed' を除くメトリクス
                    stats = stats_dict.get(key, {})
                    cols = [
                        key,
                        f"{stats.get('mean', 0):.4f}",
                        f"{stats.get('std', 0):.4f}",
                        f"{stats.get('min', 0):.4f}",
                        f"{stats.get('max', 0):.4f}",
                    ]
                    lines.append(f"| {' | '.join(cols)} |")


        lines.append("\n## パラメータ概要\n")
        lines.append("```json")
        config_data = summary_data.get('config', {})
        lines.append(json.dumps(config_data, indent=2, ensure_ascii=False))
        lines.append("```\n")

        txt = "\n".join(lines)
        try:
            with open(self.readme_path, 'w', encoding='utf-8') as f:
                f.write(txt)
            print(f"README saved to: {self.readme_path}")
        except IOError as e:
            print(f"Error saving README: {e}")

    def _calculate_stats(self, results: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
        """複数シード結果の統計（平均、標準偏差など）を計算"""
        stats_summary = {}
        if not results:
            return stats_summary
        
        for key in keys:
            values = [res.get(key, np.nan) for res in results]
            values = [v for v in values if v is not None and not np.isnan(v)]
            
            if values:
                stats_summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        return stats_summary

    def _append_results_table(self, lines: List[str], title: str, results: List[Dict], headers: List[str]):
        """テーブル形式で結果を追記（ヘルパー）"""
        lines.append(f"\n### 🔹 {title}\n")
        
        lines.append(f"| {' | '.join(headers)} |")
        lines.append(f"|{'---|' * len(headers)}")
        
        for res in results:
            cols = []
            for h in headers:
                val = res.get(h)
                if isinstance(val, float):
                    cols.append(f"{val:.4f}")
                elif val is None:
                    cols.append("N/A")
                else:
                    cols.append(str(val))
            lines.append(f"| {' | '.join(cols)} |")
