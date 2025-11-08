# RSNN DI + LangChain 実験まとめ

このドキュメントは、DIコンテナとLangChainを使用して再構築されたRSNN実験のまとめです。

**実行日時**: 2025-11-08T23:07:07.756996

## 主な結果


### 🔹 Homeo (Poisson)

| seed | acc | mean_rate |
|---|---|---|
| 2028 | 1.0000 | 0.0933 |

### 🔹 Homeo (Latency)

| seed | acc | mean_rate |
|---|---|---|
| 2028 | 0.8500 | 0.0558 |

### 🔹 E/I (Poisson)

| seed | acc | mean_rate |
|---|---|---|
| 2028 | 0.7333 | 0.1197 |

## パラメータ概要

```json
{
  "dataset_params": {
    "n_train": 100,
    "n_test": 60,
    "n_input": 40,
    "pattern_size": 6,
    "base_rate": 2.0,
    "pat_rate": 40.0
  },
  "simulation_params": {
    "T_poisson": 100,
    "T_latency": 80,
    "dt": 0.001,
    "epochs": 3
  },
  "model_params": {
    "n_hidden": 80,
    "v_th": 1.0,
    "v_reset": 0.0,
    "tau_m": 0.02,
    "rec_delay": 1,
    "rng_seed": 2028
  },
  "homeo_params": {
    "eta": 0.0005,
    "homeo_lr": 0.0005,
    "homeo_target": 0.08,
    "tau_pre": 0.02,
    "tau_post": 0.02
  },
  "ei_params": {
    "k_winners": 8,
    "excitatory_ratio": 0.8,
    "inh_strength": 1.0
  },
  "latency_params": {
    "burst_prob": 0.6,
    "burst_len": 2
  },
  "output_paths": {
    "output_dir": "./outputs",
    "summary_json": "rsnn_summary_di_lc.json",
    "readme_md": "README_rsnn_di_lc.md"
  }
}
```
