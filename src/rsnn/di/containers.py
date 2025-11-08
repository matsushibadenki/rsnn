# ./src/rsnn/di/containers.py
# タイトル: DIコンテナ
# 機能説明: dependency_injectorを使用して、プロジェクトの依存関係（サービス、モデル、設定）を
#           一元管理し、注入可能にします。
from __future__ import annotations
from dependency_injector import containers, providers
from ..core import encoding, learning_rules, models
from ..experiments import dataset, evaluation, reporting
from ..services import experiment_service

class CoreContainer(containers.DeclarativeContainer):
    """コアコンポーネント（ルール、エンコーダ）のコンテナ"""
    
    config = providers.Configuration()
    
    # 学習ルール (STDP)
    stdp_rule = providers.Factory(
        learning_rules.STDP,
        eta=config.homeo_params.eta,
        tau_pre=config.homeo_params.tau_pre,
        tau_post=config.homeo_params.tau_post,
        dt=config.simulation_params.dt
    )
    
    # 学習ルール (Homeostasis)
    homeostasis_rule = providers.Factory(
        learning_rules.Homeostasis,
        homeo_lr=config.homeo_params.homeo_lr,
        homeo_target=config.homeo_params.homeo_target
    )
    
    # エンコーディング
    poisson_encoder = providers.Object(encoding.poisson_encoding)
    latency_encoder = providers.Object(encoding.latency_burst_encoding)


class ModelsContainer(containers.DeclarativeContainer):
    """RSNNモデルのコンテナ"""
    
    config = providers.Configuration()
    core = providers.DependenciesContainer()
    
    # ベースパラメータ (共通)
    # ( 削除: providers.Singleton() で ** 展開すると TypeError が発生するため、
    #   各モデルに直接パラメータを展開します )
    # base_model_params = providers.Factory(
    #     dict,
    #     n_input=config.dataset_params.n_input,
    #     n_hidden=config.model_params.n_hidden,
    #     dt=config.simulation_params.dt,
    #     tau_m=config.model_params.tau_m,
    #     v_th=config.model_params.v_th,
    #     v_reset=config.model_params.v_reset,
    #     rec_delay=config.model_params.rec_delay,
    #     rng_seed=config.model_params.rng_seed
    # )
    
    # RSNN_Homeo (Singleton: 1インスタンスを使い回す)
    # (シードを変える場合は Factory にする)
    rsnn_homeo = providers.Singleton(
        models.RSNN_Homeo,
        # **base_model_params, # <- 修正: TypeErrorのため展開
        
        # base_model_params の内容をここに展開
        n_input=config.dataset_params.n_input,
        n_hidden=config.model_params.n_hidden,
        dt=config.simulation_params.dt,
        tau_m=config.model_params.tau_m,
        v_th=config.model_params.v_th,
        v_reset=config.model_params.v_reset,
        rec_delay=config.model_params.rec_delay,
        rng_seed=config.model_params.rng_seed,
        
        stdp_rule=core.stdp_rule,
        homeostasis_rule=core.homeostasis_rule
    )
    
    # RSNN_EI (Singleton)
    rsnn_ei = providers.Singleton(
        models.RSNN_EI,
        # **base_model_params, # <- 修正: TypeErrorのため展開
        
        # base_model_params の内容をここに展開
        n_input=config.dataset_params.n_input,
        n_hidden=config.model_params.n_hidden,
        dt=config.simulation_params.dt,
        tau_m=config.model_params.tau_m,
        v_th=config.model_params.v_th,
        v_reset=config.model_params.v_reset,
        rec_delay=config.model_params.rec_delay,
        rng_seed=config.model_params.rng_seed,

        stdp_rule=core.stdp_rule,
        homeostasis_rule=core.homeostasis_rule,
        excitatory_ratio=config.ei_params.excitatory_ratio,
        # 修正: 欠落していたパラメータと閉じる括弧を追加
        inh_strength=config.ei_params.inh_strength,
        k_winners=config.ei_params.k_winners
    )

class ExperimentContainer(containers.DeclarativeContainer):
    """実験関連（データ、評価）のコンテナ"""
    
    config = providers.Configuration()
    
    # データセットジェネレータ
    dataset_generator = providers.Singleton(
        dataset.DatasetGenerator,
        n_input=config.dataset_params.n_input,
        pattern_size=config.dataset_params.pattern_size,
        base_rate=config.dataset_params.base_rate,
        pat_rate=config.dataset_params.pat_rate,
        rng_seed=config.model_params.rng_seed # データ生成も同じシードで固定
    )
    
    # 評価器
    evaluator = providers.Factory(
        evaluation.ReadoutEvaluator,
        alpha=0.1 # 固定値
    )
    
    # レポーター
    reporter = providers.Factory(
        reporting.ResultReporter,
        output_dir=config.output_paths.output_dir,
        summary_filename=config.output_paths.summary_json,
        readme_filename=config.output_paths.readme_md
    )


class ServicesContainer(containers.DeclarativeContainer):
    """サービス（ビジネスロジック）のコンテナ"""
    
    experiments = providers.DependenciesContainer()
    
    # 実験サービス
    experiment_service = providers.Factory(
        experiment_service.ExperimentService,
        dataset_generator=experiments.dataset_generator,
        evaluator=experiments.evaluator
    )

class ApplicationContainer(containers.DeclarativeContainer):
    """
    アプリケーション全体のDIコンテナ
    """
    config = providers.Configuration()
    
    core = providers.Container(
        CoreContainer,
        config=config
    )
    
    models = providers.Container(
        ModelsContainer,
        config=config,
        core=core
    )
    
    experiments = providers.Container(
        ExperimentContainer,
        config=config
    )
    
    services = providers.Container(
        ServicesContainer,
        experiments=experiments
    )
