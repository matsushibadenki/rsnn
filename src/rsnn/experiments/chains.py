# ./src/rsnn/experiments/chains.py
# タイトル: LangChain 実験パイプライン
# 機能説明: LangChain (LCEL) を使用して、実験の実行シーケンス
#           (設定 -> 実行 -> レポート) を定義します。
from __future__ import annotations
import datetime
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from ..di.containers import ApplicationContainer
from ..core import encoding

# DIコンテナから必要なコンポーネントを取得するためのヘルパー関数群
# (LangChainのRunnableLambdaから呼び出される)

def _get_homeo_model(container: ApplicationContainer):
    return container.models.rsnn_homeo()

def _get_ei_model(container: ApplicationContainer):
    return container.models.rsnn_ei()

def _get_poisson_encoder(container: ApplicationContainer):
    return container.core.poisson_encoder()

def _get_latency_encoder(container: ApplicationContainer):
    return container.core.latency_encoder()

def _get_experiment_service(container: ApplicationContainer):
    return container.services.experiment_service()

def _get_reporter(container: ApplicationContainer):
    return container.experiments.reporter()

# --- LangChainのChain定義 ---

class ExperimentChains:
    """LangChain (LCEL) を使った実験実行Chain"""
    
    def __init__(self, container: ApplicationContainer):
        self.container = container
        self.config = container.config() # type: ignore[attr-defined]

    def _run_homeo_poisson(self, input_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chain 1: Homeo + Poisson の実行"""
        service = _get_experiment_service(self.container)
        model = _get_homeo_model(self.container)
        encoder = _get_poisson_encoder(self.container)
        
        sim_params = {'T': self.config['simulation_params']['T_poisson'],
                      'epochs': self.config['simulation_params']['epochs']}
        
        # 修正: poisson_encoding に必要な 'dt' を渡す
        poisson_params = {
            'dt': self.config['simulation_params']['dt']
        }
        
        return service.run_experiment(
            rsnn_model=model,
            encoding_fn=encoder,
            encoding_params=poisson_params, # 修正: 以前は {} だった
            dataset_params=self.config['dataset_params'],
            sim_params=sim_params,
            seeds=[self.config['model_params']['rng_seed']] # 簡略化
        )

    def _run_homeo_latency(self, input_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chain 2: Homeo + Latency の実行"""
        service = _get_experiment_service(self.container)
        model = _get_homeo_model(self.container) # Homeoモデルを再利用
        encoder = _get_latency_encoder(self.container)
        
        sim_params = {'T': self.config['simulation_params']['T_latency'],
                      'epochs': self.config['simulation_params']['epochs']}
        
        return service.run_experiment(
            rsnn_model=model,
            encoding_fn=encoder,
            encoding_params=self.config['latency_params'],
            dataset_params=self.config['dataset_params'],
            sim_params=sim_params,
            seeds=[self.config['model_params']['rng_seed']]
        )

    def _run_ei_poisson(self, input_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chain 3: EI + Poisson の実行"""
        service = _get_experiment_service(self.container)
        model = _get_ei_model(self.container) # EIモデルを使用
        encoder = _get_poisson_encoder(self.container)

        sim_params = {'T': self.config['simulation_params']['T_poisson'],
                      'epochs': self.config['simulation_params']['epochs']}

        # 修正: poisson_encoding に必要な 'dt' を渡す
        poisson_params = {
            'dt': self.config['simulation_params']['dt']
        }

        return service.run_experiment(
            rsnn_model=model,
            encoding_fn=encoder,
            encoding_params=poisson_params, # 修正: 以前は {} だった
            dataset_params=self.config['dataset_params'],
            sim_params=sim_params,
            seeds=[self.config['model_params']['rng_seed']]
        )

    def _generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Chain 4: 最終レポートの生成"""
        reporter = _get_reporter(self.container)
        
        summary_data = {
            "run_timestamp": datetime.datetime.now().isoformat(),
            "config": self.config,
            **results 
        }
        
        reporter.save_json_summary(summary_data)
        reporter.generate_readme(summary_data)
        
        return summary_data

    def get_full_experiment_chain(self):
        """
        全ての実験を実行し、レポートを生成するLCEL Chainを構築します。
        
        入力: config (Dict)
        出力: 最終サマリ (Dict)
        """
        
        # RunnablePassthroughは入力をそのまま後続に渡しつつ、
        # 並列実行（RunnableParallel）のキーとして結果を保持するために使います。
        
        chain = RunnablePassthrough.assign(
            homeo_poisson_results=RunnableLambda(self._run_homeo_poisson),
            homeo_latency_results=RunnableLambda(self._run_homeo_latency),
            ei_poisson_results=RunnableLambda(self._run_ei_poisson)
        ) | RunnableLambda(self._generate_report)
        
        return chain