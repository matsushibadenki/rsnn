# ./src/rsnn/experiments/reporting.py
# タイトル: レポーティングモジュール
# 機能説明: 実験結果をJSONおよびMarkdownファイルとして保存します。
from __future__ import annotations
import os
import json
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
            with open(self.summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
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
            
        lines.append("## 主な結果\n")
        
        # 計測するヘッダー (Objective.md フェーズ2.4対応)
        result_headers = ['seed', 'acc', 'mean_rate', 'mean_total_spikes']

        homeo_results = summary_data.get('homeo_poisson_results', [])
        if homeo_results:
            self._append_results_table(lines, "Homeo (Poisson)", homeo_results, 
                                       result_headers)

        latency_results = summary_data.get('homeo_latency_results', [])
        if latency_results:
            self._append_results_table(lines, "Homeo (Latency)", latency_results, 
                                       result_headers)

        ei_results = summary_data.get('ei_poisson_results', [])
        if ei_results:
            self._append_results_table(lines, "E/I (Poisson)", ei_results,
                                       result_headers)

        lines.append("\n## パラメータ概要\n")
        lines.append("```json")
        # configはLangChainの入力から取得することを想定
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

    def _append_results_table(self, lines: List[str], title: str, results: List[Dict], headers: List[str]):
        """テーブル形式で結果を追記（ヘルパー）"""
        lines.append(f"\n### 🔹 {title}\n")
        
        # ヘッダー
        lines.append(f"| {' | '.join(headers)} |")
        lines.append(f"|{'---|' * len(headers)}")
        
        # データ行
        for res in results:
            cols = []
            for h in headers:
                val = res.get(h) # 存在しない場合 (古い実行結果など) は None
                if isinstance(val, float):
                    # mean_total_spikes は小数点以下が必要ない場合もあるが、統一
                    cols.append(f"{val:.4f}")
                elif val is None:
                    cols.append("N/A") # ヘッダーにあってもデータがない場合
                else:
                    cols.append(str(val))
            lines.append(f"| {' | '.join(cols)} |")
