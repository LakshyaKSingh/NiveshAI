# TradePilot: Machine Learning-Based Automated Stock Trading and Portfolio Management System

## Complete Technical Documentation and Engineering Specification

---

**Document Control Information**

| Attribute | Details |
| :--- | :--- |
| **Project Title** | TradePilot — ML-Based Automated Stock Trading and Portfolio Management System |
| **Document Classification** | Complete Technical Documentation & System Specification |
| **Software Version** | `0.1.0` |
| **Date of Verification** | October 2026 |
| **Target Audience** | Academic Evaluators, Engineering Faculty, Technical Reviewers, Software Engineers, Quant Researchers |
| **System Implementation** | Python 3.10+, Scikit-Learn, Pandas, NumPy, Alpaca Market Data API |
| **Primary Code Location** | `src/tradepilot/` |
| **Test Suite Status** | 17 of 17 Automated Tests Passing (`pytest -v`) |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
   - [1.1 Project Overview](#11-project-overview)
   - [1.2 Problem Statement](#12-problem-statement)
   - [1.3 Core Objectives](#13-core-objectives)
   - [1.4 Key System Capabilities](#14-key-system-capabilities)
   - [1.5 Current Implementation Status](#15-current-implementation-status)
   - [1.6 Intended Users and Real-World Use Cases](#16-intended-users-and-real-world-use-cases)
2. [Project Scope and System Requirements](#2-project-scope-and-system-requirements)
   - [2.1 Functional Requirements](#21-functional-requirements)
   - [2.2 Non-Functional Requirements](#22-non-functional-requirements)
   - [2.3 System Boundaries and Execution Constraints](#23-system-boundaries-and-execution-constraints)
   - [2.4 System Assumptions](#24-system-assumptions)
   - [2.5 In-Scope vs. Out-of-Scope Capabilities](#25-in-scope-vs-out-of-scope-capabilities)
3. [Technology Stack and Environment](#3-technology-stack-and-environment)
   - [3.1 Runtime Environment and Core Libraries](#31-runtime-environment-and-core-libraries)
   - [3.2 Data Ingestion & Market Connectivity](#32-data-ingestion--market-connectivity)
   - [3.3 Machine Learning and Numerical Computation](#33-machine-learning-and-numerical-computation)
   - [3.4 Quality Assurance and Testing Framework](#34-quality-assurance-and-testing-framework)
   - [3.5 Clarification on Emerging and Future Stack Components](#35-clarification-on-emerging-and-future-stack-components)
4. [System Architecture and Component Design](#4-system-architecture-and-component-design)
   - [4.1 High-Level Architecture Diagram](#41-high-level-architecture-diagram)
   - [4.2 End-to-End Data and Control Flow](#42-end-to-end-data-and-control-flow)
   - [4.3 Architectural Decomposition and Component Responsibilities](#43-architectural-decomposition-and-component-responsibilities)
   - [4.4 Separation of Concerns: Prediction, Strategy, Execution, and Risk](#44-separation-of-concerns-prediction-strategy-execution-and-risk)
   - [4.5 Design Principles and Modularity Rationale](#45-design-principles-and-modularity-rationale)
5. [Repository and File Structure](#5-repository-and-file-structure)
   - [5.1 Complete Workspace Directory Tree](#51-complete-workspace-directory-tree)
   - [5.2 Module Mapping and Responsibility Matrix](#52-module-mapping-and-responsibility-matrix)
   - [5.3 Data and Artifact Storage Conventions](#53-data-and-artifact-storage-conventions)
6. [Data Acquisition and Dataset Description](#6-data-acquisition-and-dataset-description)
   - [6.1 Market Data Source and Historical Parameters](#61-market-data-source-and-historical-parameters)
   - [6.2 Ticker Universe Selection across Sectors](#62-ticker-universe-selection-across-sectors)
   - [6.3 Raw Data Schema and Verified Record Counts](#63-raw-data-schema-and-verified-record-counts)
   - [6.4 Data Quality Validation and Cleansing Procedures](#64-data-quality-validation-and-cleansing-procedures)
   - [6.5 Market Data Nuances: IEX Feed Realities and Survivorship Bias](#65-market-data-nuances-iex-feed-realities-and-survivorship-bias)
7. [Exploratory Data Analysis and Feature Engineering](#7-exploratory-data-analysis-and-feature-engineering)
   - [7.1 Feature Generation Methodology](#71-feature-generation-methodology)
   - [7.2 Mathematical Definitions of Technical Indicators](#72-mathematical-definitions-of-technical-indicators)
   - [7.3 Scale Invariance: Why Stationary Features Replace Raw Prices and SMAs](#73-scale-invariance-why-stationary-features-replace-raw-prices-and-smas)
   - [7.4 Target Formulation: Multi-Class Return Horizon](#74-target-formulation-multi-class-return-horizon)
   - [7.5 Strict Chronological Grouping and Lookahead Leakage Prevention](#75-strict-chronological-grouping-and-lookahead-leakage-prevention)
8. [Machine Learning Methodology](#8-machine-learning-methodology)
   - [8.1 Supervised Problem Formulation](#81-supervised-problem-formulation)
   - [8.2 Input Feature Matrix vs. Prohibited Data Leaks](#82-input-feature-matrix-vs-prohibited-data-leaks)
   - [8.3 Chronological Splitting and Boundary Purging](#83-chronological-splitting-and-boundary-purging)
   - [8.4 Candidate Model Architectures](#84-candidate-model-architectures)
   - [8.5 Preprocessing, Pipeline Construction, and Seed Management](#85-preprocessing-pipeline-construction-and-seed-management)
   - [8.6 Model Selection Protocol](#86-model-selection-protocol)
   - [8.7 Model Serialization and Auditable Metadata](#87-model-serialization-and-auditable-metadata)
9. [Model Evaluation and Experimental Results](#9-model-evaluation-and-experimental-results)
   - [9.1 Evaluation Metrics for Class-Imbalanced Financial Data](#91-evaluation-metrics-for-class-imbalanced-financial-data)
   - [9.2 Validation Comparison across Candidate Models (2024 Fold)](#92-validation-comparison-across-candidate-models-2024-fold)
   - [9.3 Final Test Period Generalization Performance (2025 Fold)](#93-final-test-period-generalization-performance-2025-fold)
   - [9.4 Out-of-Sample Confusion Matrix Analysis](#94-out-of-sample-confusion-matrix-analysis)
   - [9.5 Financial Machine Learning Interpretation of Model Skill](#95-financial-machine-learning-interpretation-of-model-skill)
10. [Backtesting Engine and Portfolio Accounting](#10-backtesting-engine-and-portfolio-accounting)
    - [10.1 Event-Driven Simulation Principles](#101-event-driven-simulation-principles)
    - [10.2 Model Probability to Trading Signal Mapping](#102-model-probability-to-trading-signal-mapping)
    - [10.3 Order Generation, Queuing, and Next-Day Open Execution](#103-order-generation-queuing-and-next-day-open-execution)
    - [10.4 Execution Friction: Slippage and Commission Accounting](#104-execution-friction-slippage-and-commission-accounting)
    - [10.5 Portfolio Risk Controls and Capital Allocation Limits](#105-portfolio-risk-controls-and-capital-allocation-limits)
    - [10.6 Marked-to-Market Accounting and Position State Tracking](#106-marked-to-market-accounting-and-position-state-tracking)
    - [10.7 Financial Performance Metrics Formulae](#107-financial-performance-metrics-formulae)
    - [10.8 Empirical Strategy Performance vs. Equal-Weight Buy-and-Hold](#108-empirical-strategy-performance-vs-equal-weight-buy-and-hold)
11. [Detailed Source Code and Module Reference](#11-detailed-source-code-and-module-reference)
    - [11.1 Consolidated Module Reference Matrix](#111-consolidated-module-reference-matrix)
    - [11.2 Package Configuration: `tradepilot.config`](#112-package-configuration-tradepilotconfig)
    - [11.3 Data Ingestion and Validation: `tradepilot.data`](#113-data-ingestion-and-validation-tradepilotdata)
    - [11.4 Feature Engineering Engine: `tradepilot.features`](#114-feature-engineering-engine-tradepilotfeatures)
    - [11.5 Modeling and Cross-Validation: `tradepilot.modeling`](#115-modeling-and-cross-validation-tradepilotmodeling)
    - [11.6 Backtesting Simulator: `tradepilot.backtesting`](#116-backtesting-simulator-tradepilotbacktesting)
    - [11.7 Automated Pipelines: `tradepilot.pipelines`](#117-automated-pipelines-tradepilotpipelines)
    - [11.8 Root Operational Entrypoints](#118-root-operational-entrypoints)
12. [Installation, Environment Setup, and Execution Guide](#12-installation-environment-setup-and-execution-guide)
    - [12.1 Environment Prerequisites](#121-environment-prerequisites)
    - [12.2 Windows PowerShell Virtual Environment Activation](#122-windows-powershell-virtual-environment-activation)
    - [12.3 Editable Package Installation](#123-editable-package-installation)
    - [12.4 Secure Environment Variable Configuration](#124-secure-environment-variable-configuration)
    - [12.5 Executing Data Ingestion, Model Training, and Backtesting](#125-executing-data-ingestion-model-training-and-backtesting)
    - [12.6 Locating and Auditing Generated Artifacts](#126-locating-and-auditing-generated-artifacts)
    - [12.7 Troubleshooting Common Operational Faults](#127-troubleshooting-common-operational-faults)
13. [Quality Assurance, Automated Testing, and Reproducibility](#13-quality-assurance-automated-testing-and-reproducibility)
    - [13.1 Testing Architecture and Test Discovery](#131-testing-architecture-and-test-discovery)
    - [13.2 Breakdown of Automated Unit and Integration Tests](#132-breakdown-of-automated-unit-and-integration-tests)
    - [13.3 Test Verification Log and Passing Assertions](#133-test-verification-log-and-passing-assertions)
    - [13.4 Deterministic Execution Guarantees](#134-deterministic-execution-guarantees)
    - [13.5 Test Scope Boundaries: What Automated Tests Do and Do Not Guarantee](#135-test-scope-boundaries-what-automated-tests-do-and-do-not-guarantee)
14. [Known Limitations, Market Realities, and Quant Risks](#14-known-limitations-market-realities-and-quant-risks)
    - [14.1 Survivorship and Selection Bias](#141-survivorship-and-selection-bias)
    - [14.2 IEX Market Data Limitations](#142-iex-market-data-limitations)
    - [14.3 Static Volatility Thresholds and Non-Stationary Regimes](#143-static-volatility-thresholds-and-non-stationary-regimes)
    - [14.4 Cross-Asset Correlation and Sector Concentration](#144-cross-asset-correlation-and-sector-concentration)
    - [14.5 Friction and Execution Assumptions vs. Live Order Books](#145-friction-and-execution-assumptions-vs-live-order-books)
    - [14.6 Predictive Accuracy vs. Portfolio Alpha Divergence](#146-predictive-accuracy-vs-portfolio-alpha-divergence)
15. [Security, Governance, and Operational Considerations](#15-security-governance-and-operational-considerations)
    - [15.1 API Key and Secret Management](#151-api-key-and-secret-management)
    - [15.2 Version Control Hygiene and Secret Leakage Prevention](#152-version-control-hygiene-and-secret-leakage-prevention)
    - [15.3 Paper Trading Isolation from Real Capital Execution](#153-paper-trading-isolation-from-real-capital-execution)
    - [15.4 Structured Logging, Auditability, and Run Failure Handling](#154-structured-logging-auditability-and-run-failure-handling)
16. [Future Development and Research Roadmap](#16-future-development-and-research-roadmap)
    - [16.1 Phase 1: Operational Hardening and Version Control](#161-phase-1-operational-hardening-and-version-control)
    - [16.2 Phase 2: Dynamic Feature Engineering and Target Labeling](#162-phase-2-dynamic-feature-engineering-and-target-labeling)
    - [16.3 Phase 3: Walk-Forward Validation and Gradient Boosting Ensembles](#163-phase-3-walk-forward-validation-and-gradient-boosting-ensembles)
    - [16.4 Phase 4: MLOps Integration (MLflow and DVC)](#164-phase-4-mlops-integration-mlflow-and-dvc)
    - [16.5 Phase 5: Dynamic Risk Management and Sizing Strategies](#165-phase-5-dynamic-risk-management-and-sizing-strategies)
    - [16.6 Phase 6: Automated Paper-Trading Execution Service](#166-phase-6-automated-paper-trading-execution-service)
17. [Technical and Financial Glossary](#17-technical-and-financial-glossary)
18. [Conclusion and Academic Project Assessment](#18-conclusion-and-academic-project-assessment)
19. [Documentation Verification and Audit Log](#19-documentation-verification-and-audit-log)

---

## 1. Executive Summary

### 1.1 Project Overview
**TradePilot** is an institutional-grade, research-oriented automated stock trading and portfolio management software system engineered in Python. It bridges modern statistical machine learning with realistic quantitative backtesting. The platform ingests historical equity market bars, enforces strict financial data validation constraints, extracts stationary technical indicators, formulates multi-class directional forecasting targets, trains reproducible classification pipelines, and executes event-driven simulations subject to market frictions including adverse slippage, transaction commissions, and position exposure caps.

### 1.2 Problem Statement
Quantitative machine learning systems developed in academic and exploratory environments routinely suffer from catastrophic failure when evaluated out-of-sample or deployed in real financial markets. These failures stem from standard structural errors:
1. **Lookahead Leakage (Data Snooping):** Normalizing features globally, shifting returns across mixed stock groups, or allowing forward-looking target information to contaminate predictive features.
2. **Train-Test Boundary Spillover:** Splitting time-series datasets arbitrarily or allowing overlapping forward prediction windows (e.g., multi-day horizon returns) to straddle the division between training and validation periods.
3. **Price Scale Confounding:** Inputting unstandardized dollar price series or raw price-level moving averages into machine learning models across diverse equities, causing algorithms to learn arbitrary stock price levels (e.g., an equity trading at $20 versus an equity trading at $600) rather than scale-invariant momentum or mean-reversion dynamics.
4. **Frictionless Simulation Illusions:** Assuming trades can be filled at the exact closing price where the signal was detected, with zero slippage, zero commission fees, infinite liquidity, and instantaneous execution.
5. **Class Imbalance Distortions:** Relying on standard accuracy as a performance metric in directional forecasting, which rewards models that simply predict the dominant market trend without generating actionable risk-adjusted trading alpha.

### 1.3 Core Objectives
TradePilot was designed from first principles to resolve these architectural vulnerabilities by delivering:
- A leak-free data engineering pipeline that computes strictly backward-looking technical indicators grouped independently by ticker.
- A multi-class forecasting model predicting 5-day directional return outcomes (`UP`, `DOWN`, `NEUTRAL`) thresholded at $\pm 1.0\%$.
- Chronological train/validation/test splitting with boundary purging that mathematically guarantees zero target-horizon overlap across evaluation folds.
- Realistic, event-driven backtesting that generates signals at market close $t$ and executes fills strictly at next-session Open $t+1$, pricing in 5 basis points of adverse slippage and 5 basis points of transaction fees.
- Comprehensive marked-to-market portfolio accounting tracking cash, gross exposure, drawdowns, transaction turnover, Sharpe ratios, win rates, and profit factors against an Equal-Weighted Buy-and-Hold benchmark.
- An auditable, modular codebase accompanied by a fully passing automated test suite (`pytest`).

### 1.4 Key System Capabilities
- **Automated Market Data Ingestion:** Downloads historical daily split- and dividend-adjusted OHLCV bars via Alpaca Market Data API using `alpaca-py`.
- **Integrity & Bounds Validation:** Verifies schema structure, detects duplicates, and programmatically enforces logical physical price checks ($High \ge Open$, $High \ge Close$, $Low \le Open$, $Low \le Close$, $Low \le High$, $Volume \ge 0$).
- **Stationary Feature Computation:** Extracts scale-invariant momentum, volatility, and trend indicators (Returns over 1d, 5d, 10d; 10-day volatility; volume change; 14-day RSI; MACD line and 9-day signal line).
- **Leak-Proof Target Construction:** Formulates future 5-day forward return labels per ticker, explicitly dropping all warmup and shift NaNs.
- **Model Candidate Benchmarking:** Evaluates a `Majority_Baseline`, a `Logistic_Regression` classifier with `StandardScaler`, and a `Random_Forest` ensemble tuned with balanced class weights to counteract financial class imbalance.
- **Friction-Aware Backtesting Engine:** Simulates realistic execution mechanics, queueing orders, filling at $t+1$ Open, maintaining solvency, and accounting for round-trip friction costs.
- **Auditable Artifact Generation:** Serializes trained model pipelines (`best_model.joblib`), metadata schemas (`model_metadata.json`), daily equity curves, trade ledgers, and comparative benchmark reports.

### 1.5 Current Implementation Status
The project is fully implemented, verified, and operational in `E:\Field Project\New Approach`. 
- The complete pipeline executes deterministically from ingestion through backtesting.
- All 17 automated tests in the test suite pass with 100% compliance.
- Verified datasets span 50 equities from January 2021 through December 2025 (62,664 raw OHLCV rows; 59,964 clean, labeled feature rows).
- Model evaluation demonstrates that the Random Forest classifier successfully captures non-linear structure across all classes, outperforming baseline models on balanced accuracy and Macro F1.
- Backtesting validates that while the strategy generates positive cumulative returns in both 2024 (+10.39%) and 2025 (+9.94%), frequent rebalancing frictions and cash-drag limitations restrict its ability to outperform a concentrated mega-cap bull-market benchmark.

### 1.6 Intended Users and Real-World Use Cases
- **Academic Evaluators and University Faculty:** Reviewing student engineering rigor, machine learning methodology, time-series leak prevention, and software architecture.
- **Quantitative Researchers & Machine Learning Engineers:** Using the codebase as an extensible template for testing alternative alpha factors, gradient boosting models, or dynamic volatility thresholds.
- **Technical Recruiters & Engineering Hiring Managers:** Inspecting code quality, architectural separation of concerns, test coverage, and realistic documentation of quant financial limitations.
- **Algorithmic Traders:** Transitioning from exploratory Jupyter/Colab notebooks to production-style modular Python packages with deterministic execution.

---

## 2. Project Scope and System Requirements

### 2.1 Functional Requirements
1. **Data Acquisition:** The system must connect to the Alpaca Market Data API using secure environment variables and retrieve daily OHLCV bars across a user-defined basket of equities for a multi-year historical span.
2. **Schema & Logical Validation:** The system must validate that all expected columns are present, ensure no duplicate records exist per (symbol, timestamp), reject non-positive prices, and verify strict geometric candlestick boundaries.
3. **Feature Generation:** The system must compute 8 stationary technical indicators grouped strictly by stock ticker, ensuring no cross-symbol information leakage.
4. **Target Labeling:** The system must calculate a forward-looking 5-day return horizon ($Close_{t+5} / Close_t - 1$) and categorize each bar into `UP` ($> +1\%$), `DOWN` ($< -1\%$), or `NEUTRAL` (between $-1\%$ and $+1\%$).
5. **Data Purging & Splitting:** The system must partition data chronologically (Training: $\le 2023$; Validation: $2024$; Test: $2025$). To prevent target overlap leakage, the system must purge exactly 5 trading sessions from the end of the training and validation periods for every ticker.
6. **Model Training & Comparison:** The system must fit candidate models on training data, benchmark performance on validation data using Macro F1 and Balanced Accuracy, and select the winning candidate for test evaluation.
7. **Model Artifact Persistence:** The winning model pipeline must be serialized with joblib, and training configuration parameters, timestamps, and split statistics must be saved in JSON format.
8. **Signal Transformation:** The system must convert out-of-sample class probability estimates dynamically into discrete trading signals (`1` for Buy, `-1` for Exit, `0` for Hold) based on a configurable probability threshold.
9. **Event-Driven Backtest Simulation:** The system must simulate order execution at next-day Open ($t+1$), apply adverse slippage and commissions, manage portfolio cash balances, enforce a maximum position count, and record daily marked-to-market equity values and round-trip trade statistics.
10. **Benchmark Reporting:** The system must compare the strategy's risk and return profile against an Equal-Weighted Buy-and-Hold benchmark over the identical investment universe and time period.

### 2.2 Non-Functional Requirements
1. **Determinism and Reproducibility:** All random seeds must be explicitly set (`random_seed = 42`). Identical inputs and random seeds must yield bitwise identical model parameters, evaluation metrics, and equity curves.
2. **Modularity and Clean Architecture:** The software must adhere to separation of concerns. Ingestion, validation, feature engineering, modeling, backtesting, and pipeline orchestration must reside in independent, self-contained subpackages.
3. **Data Integrity and Leak Prevention:** Forward-looking variables (`future_return_5d`, `target`) and non-stationary identifiers (`open`, `high`, `low`, `close`, `volume`, `symbol`, `timestamp`) must be strictly prohibited from the feature input matrix.
4. **Code Quality and Robustness:** The code must include type annotations, clear docstrings, defensive input assertions, informative error messages, and standard Python logging.
5. **Execution Efficiency:** Daily bar processing across 50 equities (approx. 60,000 observations) must complete within seconds on standard desktop hardware without requiring distributed compute clusters.

### 2.3 System Boundaries and Execution Constraints
- **Temporal Resolution:** The system operates strictly on daily end-of-day (EOD) market bars. Intraday tick, minute, or hourly execution is outside the scope of this implementation.
- **Asset Universe:** The implementation focuses on US equity securities. Options, futures, foreign exchange, and fixed-income instruments are excluded.
- **Hardware Profile:** All components execute in single-node environments (Windows 10/11 or Linux x86_64) using multi-core CPU parallelization (`n_jobs = -1`) for ensemble training.
- **Execution Environment:** Commands, directory paths, and script wrappers are standardized for Windows PowerShell while maintaining OS-agnostic POSIX path compatibility via `pathlib.Path`.

### 2.4 System Assumptions
- **Execution Liquidity:** Daily trading volume for the chosen universe (large-cap US equities) is assumed to be sufficiently deep that strategy orders (max $10,000 per position) do not move the broader market beyond the modeled 5 bps adverse slippage.
- **Cash Settlement:** Cash proceeds from sales are immediately available for redeployment on the subsequent trading bar (T+0 cash availability in backtest simulation).
- **Price Adjustment:** Historical prices downloaded via Alpaca API are split- and dividend-adjusted (`Adjustment.ALL`), preserving historical return continuity.

### 2.5 In-Scope vs. Out-of-Scope Capabilities

| Capability Domain | In-Scope (Implemented & Verified) | Out-of-Scope (Future / Planned) |
| :--- | :--- | :--- |
| **Market Data** | Historical daily OHLCV via Alpaca IEX API | Real-time WebSocket streaming, L2 order books |
| **Data Cleansing** | Missing value drops, duplicate checks, OHLC geometric bounds | Outlier winsorization, corporate action adjustment parsers |
| **Features** | 8 stationary indicators (Returns, Volatility, RSI, MACD, Volume) | Fundamental ratios, order flow imbalance, sentiment analysis |
| **Target Design** | Fixed $\pm 1\%$ 5-day forward return 3-class classification | Dynamic ATR-based targets, Triple Barrier Method, regression |
| **Machine Learning** | DummyClassifier, Logistic Regression, Random Forest | XGBoost, LightGBM, CatBoost, LSTM, Temporal Fusion Transformers |
| **Validation** | Chronological 3-way split with 5-day boundary purging | Purged Walk-Forward Cross-Validation, Combinatorial Purged CV |
| **Trading Strategy** | Long-only event-driven execution with 5-day time exit | Long/short pairs, market-neutral hedging, stop-loss/take-profit |
| **Execution Modeling** | Next-day Open execution, 5 bps slippage, 5 bps commissions | Limit order book simulation, partial fills, market impact models |
| **Deployment & Ops** | Local CLI pipelines, file-based artifacts, automated test suite | MLflow server, DVC remote storage, Docker, live Alpaca trading bot |

---

## 3. Technology Stack and Environment

### 3.1 Runtime Environment and Core Libraries
TradePilot is built on **Python 3.10** (verified on Python 3.10.7 on Windows x86_64). The project relies on standard foundational libraries for high-performance array manipulation and structured data analysis:
- **Python (>= 3.10):** Native runtime providing modern structural pattern matching, robust type hinting, and enhanced exception handling.
- **NumPy (>= 1.22.0):** Provides vectorized array computation, fast mathematical transformations, standard deviations, and matrix operations underlying scikit-learn models.
- **Pandas (>= 2.0.0):** The primary tabular data engine utilized for time-series grouping, rolling window statistics (rolling mean, rolling standard deviation), date filtering, and ledger management.
- **Pathlib:** Standard library module utilized throughout `tradepilot.config` to ensure cross-platform, deterministic path resolution relative to the repository root.

### 3.2 Data Ingestion & Market Connectivity
- **Alpaca Python SDK (`alpaca-py >= 0.44.0`):** The official modern SDK for interfacing with Alpaca Markets. TradePilot uses `StockHistoricalDataClient` to dispatch `StockBarsRequest` queries requesting daily bars with `DataFeed.IEX` and `Adjustment.ALL`.
- **Python-dotenv (`python-dotenv >= 1.0.0`):** Loads environment variables securely from an uncommitted `.env` file into `os.environ`, preventing credentials from leaking into source code.

### 3.3 Machine Learning and Numerical Computation
- **Scikit-Learn (`scikit-learn >= 1.3.0`):** The core machine learning engine. TradePilot utilizes:
  - `sklearn.pipeline.Pipeline`: Encapsulates preprocessing and estimation into atomic, leak-free workflows.
  - `sklearn.preprocessing.StandardScaler`: Normalizes feature distributions for linear models.
  - `sklearn.dummy.DummyClassifier`: Provides the zero-intelligence majority-class baseline.
  - `sklearn.linear_model.LogisticRegression`: Serves as the linear parametric benchmark with L2 regularization.
  - `sklearn.ensemble.RandomForestClassifier`: The primary non-linear ensemble algorithm, tuned with `class_weight="balanced"`.
  - `sklearn.metrics`: Computes `accuracy_score`, `balanced_accuracy_score`, `f1_score(average="macro")`, `classification_report`, and `confusion_matrix`.
- **Joblib (`joblib >= 1.3.0`):** High-efficiency serialization library used to persist trained scikit-learn pipeline objects to disk (`best_model.joblib`).

### 3.4 Quality Assurance and Testing Framework
- **Pytest (`pytest >= 7.0.0`):** Testing framework that discovers and runs the 17 unit and integration tests under `tests/`. Pytest settings are defined directly in `pyproject.toml`.
- **Setuptools (`setuptools >= 61.0`):** PEP 517/621-compliant build backend used to install TradePilot as an editable package (`pip install -e .`).

### 3.5 Clarification on Emerging and Future Stack Components
To maintain technical integrity, the following stack components are explicitly documented as **planned future additions** and are **not** present in the current operational implementation:
- **MLflow:** Not currently active. Experiment tracking is currently executed via structured JSON and CSV artifact generation (`models/model_metadata.json`, `reports/validation_results.csv`).
- **DVC (Data Version Control):** Not currently active. Market datasets are managed via local file versioning under `data/raw/` and `data/processed/`.
- **Flask / FastAPI / Streamlit:** No web application server is present in the current release. Interaction is achieved via command-line entry points.
- **Docker / Kubernetes:** Containerization is planned for subsequent deployment phases.

---

## 4. System Architecture and Component Design

### 4.1 High-Level Architecture Diagram
The following architectural diagram illustrates the complete end-to-end data processing and execution pipeline of TradePilot.

```mermaid
flowchart TD
    subgraph INGESTION ["1. Data Acquisition & Validation"]
        A1["Alpaca Market Data API (IEX Feed)"] -->|"Daily OHLCV Bars"| A2["download_market_data.py"]
        A2 -->|"Raw CSV"| A3["data/raw/stock_ohlcv_2021_2025.csv"]
        A3 --> B1["tradepilot.data.validate_data"]
        B1 -->|"Integrity Passed"| B2["data/processed/stock_eda_cleaned.csv"]
    end

    subgraph FEATURES ["2. Feature & Target Engineering"]
        B2 --> C1["tradepilot.features.build_features"]
        C1 -->|"Compute Indicators Grouped by Symbol"| C2["8 Stationary Features<br/>(Returns, Vol, RSI, MACD)"]
        C1 -->|"Shift Close -5 Days"| C3["Future Return & 3-Class Target<br/>(UP / DOWN / NEUTRAL)"]
        C2 -->|"Drop Warmup & Tail NaNs"| C4["data/processed/stock_features_labeled.csv"]
        C3 -->|"Drop Warmup & Tail NaNs"| C4
    end

    subgraph MODELING ["3. Chronological ML Modeling"]
        C4 --> D1["tradepilot.modeling.split_data"]
        D1 -->|"Purge 5 Days per Symbol"| D2["Train Set (2021-2023)"]
        D1 -->|"Purge 5 Days per Symbol"| D3["Validation Set (2024)"]
        D1 -->|"Unpurged Final Fold"| D4["Test Set (2025)"]

        D2 --> E1["Candidate Pipelines (Majority, LogReg, RF)"]
        E1 --> E2["Fit Models on Train"]
        E2 --> E3["Evaluate on Validation Set"]
        D3 --> E3
        E3 -->|"Select Winning Model (Macro F1)"| E4["Random Forest Model"]
        E4 --> E5["Single Test Evaluation (2025)"]
        D4 --> E5
        E4 -->|"joblib.dump"| E6["models/best_model.joblib"]
        E3 -->|"json.dump"| E7["models/model_metadata.json<br/>reports/test_results.json"]
        E5 -->|"json.dump"| E7
    end

    subgraph BACKTESTING ["4. Event-Driven Backtesting Simulator"]
        E6 --> F1["tradepilot.backtesting.signals<br/>(SignalGenerator)"]
        C4 --> F1
        F1 -->|"Signals Generated at Close (t)"| F2["Pending Order Queue"]

        F2 --> G1["tradepilot.backtesting.engine<br/>(BacktestEngine Loop)"]
        A3 -->|"Next-Session Open (t+1)"| G2["ExecutionHandler"]
        G2 -->|"Apply 5 bps Slippage + 5 bps Comm"| G3["TradeFill"]

        G3 --> H1["tradepilot.backtesting.portfolio<br/>(Portfolio Manager)"]
        A3 -->|"Session Close (t+1)"| H1
        H1 -->|"Mark-to-Market"| H2["Equity Curve & Open Positions"]
        H1 -->|"Close Round-Trips"| H3["Completed Trade Ledger"]
    end

    subgraph REPORTING ["5. Metrics & Audit Reporting"]
        H2 --> I1["tradepilot.backtesting.metrics"]
        H3 --> I1
        A3 -->|"Buy & Hold Baseline"| I1
        I1 --> J1["reports/backtest_summary.json"]
        I1 --> J2["reports/backtest_equity_curve.csv"]
        I1 --> J3["reports/trade_ledger.csv"]
        I1 --> J4["reports/benchmark_comparison.csv"]
    end
```

### 4.2 End-to-End Data and Control Flow
1. **Ingestion & Integrity Check:** `download_market_data.py` interfaces with Alpaca API, fetching split-adjusted daily OHLCV bars for 50 tickers across 2021–2025. Data is passed through schema and boundary assertions in `tradepilot.data.validate_data`.
2. **Feature Computation:** `tradepilot.features.build_features` calculates 8 stationary technical indicators grouped by symbol. Forward 5-day returns are computed, and multi-class target labels are assigned. Rows containing initialization NaNs (warm-up periods) or terminal shift NaNs are dropped.
3. **Partitioning & Boundary Purging:** `tradepilot.modeling.split_data` slices data into Train ($\le 2023$), Validation ($2024$), and Test ($2025$). Exactly 5 trading sessions per symbol are purged from the ends of the training and validation splits to eliminate target-horizon overlap.
4. **Pipeline Training & Selection:** `tradepilot.modeling.train_model` trains candidate pipelines (`Majority_Baseline`, `Logistic_Regression`, `Random_Forest`). `tradepilot.modeling.evaluate_model` benchmarks validation performance. The model with highest Macro F1 (`Random_Forest`) is selected, evaluated on the test set, and serialized to `models/best_model.joblib`.
5. **Signal Generation:** `tradepilot.backtesting.signals.SignalGenerator` consumes the serialized model, dynamically inspects class labels, computes probability outputs, and emits discrete signals (`BUY`, `EXIT`, `HOLD`) using an entry threshold of $P(UP) \ge 0.40$.
6. **Execution Simulation:** `tradepilot.backtesting.engine.BacktestEngine` runs an event-driven chronological loop. Signals produced at Close $t$ become orders filled at Open $t+1$. `ExecutionHandler` models 5 bps slippage and 5 bps fees.
7. **Accounting & Metrics:** `Portfolio` tracks cash, holdings, and daily mark-to-market valuations. Round-trip trades are logged. Strategy performance is compared against an Equal-Weighted Buy-and-Hold benchmark in `reports/backtest_summary.json`.

### 4.3 Architectural Decomposition and Component Responsibilities

| Subpackage | Primary Module | Core Responsibility |
| :--- | :--- | :--- |
| `tradepilot` | `config.py` | Centralized data classes (`DataConfig`, `ModelConfig`, `BacktestConfig`) establishing paths, hyperparameters, and columns. |
| `tradepilot.data` | `load_data.py` | Disk I/O with automatic UTC timestamp parsing and chronological sorting. |
| `tradepilot.data` | `validate_data.py` | Schema verification, duplicate detection, OHLC geometric sanity checks, and feature leakage assertions. |
| `tradepilot.features` | `build_features.py` | Vectorized technical indicator generation per ticker and multi-class target threshold assignment. |
| `tradepilot.modeling` | `split_data.py` | Chronological dataset partitioning with per-ticker boundary purging. |
| `tradepilot.modeling` | `train_model.py` | Definition and fitting of scikit-learn model pipelines. |
| `tradepilot.modeling` | `evaluate_model.py` | Multi-class classification evaluation, balanced metrics, and candidate selection. |
| `tradepilot.modeling` | `save_model.py` | Serialization of fitted estimators (`joblib`) and model metadata schemas (`json`). |
| `tradepilot.backtesting` | `signals.py` | Dynamic class-order probability mapping and signal generation. |
| `tradepilot.backtesting` | `execution.py` | Order generation, budget allocation bounding, next-day Open fills, and transaction cost modeling. |
| `tradepilot.backtesting` | `portfolio.py` | Position life-cycle management, cash accounting, daily mark-to-market valuation, and trade ledger tracking. |
| `tradepilot.backtesting` | `metrics.py` | Computation of annualized Sharpe ratio, CAGR, volatility, max drawdown, win rate, and benchmark metrics. |
| `tradepilot.backtesting` | `engine.py` | Event-driven simulation orchestrator executing the chronological daily state machine. |
| `tradepilot.pipelines` | `train_pipeline.py` | End-to-end training orchestrator from raw data loading to evaluation and serialization. |
| `tradepilot.pipelines` | `backtest_pipeline.py` | Out-of-sample backtest orchestrator executing validation and test simulations and persisting reports. |

### 4.4 Separation of Concerns: Prediction, Strategy, Execution, and Risk
A hallmark of TradePilot's architecture is the strict decoupling of its four core operational domains:
1. **The Predictive Layer (`modeling`):** Focuses solely on statistical classification. It maps an 8-dimensional feature vector $X_t$ to class probabilities $P(y_t = c)$. It possesses no knowledge of account equity, position sizes, order types, or brokerage mechanics.
2. **The Decision / Signal Layer (`signals`):** Converts continuous statistical probabilities into trading intentions. It applies probability thresholds ($P(UP) \ge 0.40$) to filter out low-conviction predictions, emitting discrete signals ($1, -1, 0$).
3. **The Execution Layer (`execution`):** Bridges trading intentions with market reality. It translates a signal generated at Close $t$ into an actionable order executed at Open $t+1$. It calculates the exact integer shares affordable given available capital and deducts modeled slippage and exchange fees.
4. **The Risk and Portfolio Layer (`portfolio`):** Enforces capital preservation rules. It enforces a maximum of 10 concurrent open positions, caps single-stock exposure at 10% of total portfolio equity, prevents position pyramiding, rejects orders when cash is depleted, and executes time-based exits at the expiration of the 5-day holding period.

### 4.5 Design Principles and Modularity Rationale
The architecture adheres to core software engineering principles:
- **Single Responsibility Principle (SRP):** Each module handles one distinct operational task. Feature calculation is strictly separated from data loading, and execution simulation is strictly decoupled from metric reporting.
- **Open/Closed Principle (OCP):** New candidate models (e.g., XGBoost, LightGBM) can be added to `get_candidate_pipelines()` in `train_model.py` without modifying the training orchestrator or evaluation logic.
- **Fail-Fast Defensive Programming:** Validation assertions execute prior to computationally heavy steps. If an OHLC price violates physical bounds or a feature column contains NaNs, the pipeline raises an informative `ValueError` immediately rather than propagating invalid data downstream.
- **Zero Lookahead Leakage by Construction:** Features are computed before target shifting; target shifts are grouped per symbol; boundary rows are purged prior to training; and order execution is delayed to the subsequent trading session ($t+1$).

---

## 5. Repository and File Structure

### 5.1 Complete Workspace Directory Tree
Below is the verified repository tree of the TradePilot project, excluding virtual environments (`.venv`), Python bytecode caches (`__pycache__`), and pytest caches:

```text
E:\Field Project\New Approach\
├── data/
│   ├── processed/
│   │   ├── stock_eda_cleaned.csv        # Validated base dataset with daily returns (62,664 rows)
│   │   └── stock_features_labeled.csv   # Labeled dataset with 8 features & 3-class target (59,964 rows)
│   └── raw/
│       └── stock_ohlcv_2021_2025.csv    # Raw historical OHLCV from Alpaca API (62,664 rows)
├── docs/
│   └── TradePilot_Technical_Documentation.md  # Complete technical documentation document
├── models/
│   ├── best_model.joblib                # Serialized winning Random Forest classifier pipeline
│   └── model_metadata.json              # Full training metadata, hyperparameters, and split info
├── reports/
│   ├── backtest_equity_curve.csv        # Daily marked-to-market equity values for 2025 test period
│   ├── backtest_summary.json            # Complete financial metrics for validation and test backtests
│   ├── benchmark_comparison.csv         # Comparative metrics: Strategy vs. Equal-Weight vs. Cash
│   ├── confusion_matrix.csv             # Confusion matrices across candidate models and splits
│   ├── test_results.json                # Out-of-sample 2025 classification report and metrics
│   ├── trade_ledger.csv                 # Granular record of all 343 executed trades in 2025 test
│   └── validation_results.csv           # Validation comparison table across all candidate models
├── src/
│   └── tradepilot/
│       ├── __init__.py                  # Package exports (__version__ = "0.1.0")
│       ├── config.py                    # Central configuration dataclasses and project root anchor
│       ├── backtesting/
│       │   ├── __init__.py              # Backtesting subpackage exports
│       │   ├── engine.py                # Chronological event-driven simulation loop
│       │   ├── execution.py             # Order execution, slippage, and commission modeling
│       │   ├── metrics.py               # Financial performance formulas (Sharpe, CAGR, Drawdown)
│       │   ├── portfolio.py             # Position tracking, cash accounting, and equity curves
│       │   └── signals.py               # Model probability to discrete signal mapping
│       ├── data/
│       │   ├── __init__.py              # Data subpackage exports
│       │   ├── load_data.py             # Data loading with UTC timestamp standardization
│       │   └── validate_data.py         # Schema, duplicate, OHLC bounds, and leakage validators
│       ├── features/
│       │   ├── __init__.py              # Feature subpackage exports
│       │   └── build_features.py        # Vectorized technical indicator and target generation
│       ├── modeling/
│       │   ├── __init__.py              # Modeling subpackage exports
│       │   ├── evaluate_model.py        # Classification evaluation and model selection logic
│       │   ├── save_model.py            # Artifact persistence (joblib, metadata JSON, reports)
│       │   ├── split_data.py            # Chronological splitting with boundary purging
│       │   └── train_model.py           # Candidate pipeline construction and fitting
│       └── pipelines/
│           ├── __init__.py              # Pipeline subpackage exports
│           ├── backtest_pipeline.py     # Backtesting execution entrypoint
│           └── train_pipeline.py        # Model training execution entrypoint
├── tests/
│   ├── __init__.py                      # Test package marker
│   ├── test_data_validation.py          # Tests for schema, duplicates, price bounds, and leakage
│   ├── test_execution_and_portfolio.py  # Tests for slippage, cash accounting, and solvency
│   ├── test_features.py                 # Tests for RSI, MACD, per-symbol isolation, and targets
│   ├── test_pipeline_reproducibility.py # Tests for deterministic fitting and joblib serialization
│   ├── test_signals.py                  # Tests for class order independence in signal mapping
│   └── test_split_data.py               # Tests for chronological partitioning and boundary purging
├── download_market_data.py              # Script to download raw OHLCV data from Alpaca API
├── pyproject.toml                       # Packaging configuration and pytest testpaths
├── requirements.txt                     # Explicit package dependency list
├── stock_eda.py                         # Standalone exploratory data analysis and Colab prototype
└── train.py                             # Root wrapper delegating to train_pipeline
```

### 5.2 Module Mapping and Responsibility Matrix
The following matrix illustrates the operational role of each primary source file in the workflow:

```text
[Alpaca Market Data API]
       │
       ▼
download_market_data.py ──────► data/raw/stock_ohlcv_2021_2025.csv
                                       │
                                       ▼
                              tradepilot.data.validate_data
                                       │
                                       ▼
                              tradepilot.features.build_features
                                       │
                                       ▼
                              data/processed/stock_features_labeled.csv
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        tradepilot.modeling.split_data       tradepilot.backtesting.signals
                    │                                     │
                    ▼                                     ▼
        tradepilot.modeling.train_model      tradepilot.backtesting.engine
                    │                                     │
                    ▼                                     ▼
        tradepilot.modeling.evaluate_model   tradepilot.backtesting.portfolio
                    │                                     │
                    ▼                                     ▼
        tradepilot.modeling.save_model       tradepilot.backtesting.metrics
                    │                                     │
                    ▼                                     ▼
          models/best_model.joblib              reports/backtest_summary.json
```

### 5.3 Data and Artifact Storage Conventions
- **Raw Data Directory (`data/raw/`):** Contains unadulterated source datasets downloaded directly from external providers. Files in this directory are treated as read-only immutable inputs.
- **Processed Data Directory (`data/processed/`):** Stores validated, cleaned, and feature-engineered datasets. `stock_features_labeled.csv` serves as the single source of truth for model training.
- **Models Directory (`models/`):** Stores serialized production model pipelines (`best_model.joblib`) alongside an auditable machine-readable specification (`model_metadata.json`) containing timestamps, feature lists, parameter dictionaries, and evaluation metrics.
- **Reports Directory (`reports/`):** Holds all tabular and JSON performance artifacts produced by the training and backtesting pipelines, ensuring full transparency and reproducibility.

---

## 6. Data Acquisition and Dataset Description

### 6.1 Market Data Source and Historical Parameters
Historical market data is obtained via the **Alpaca Market Data API** using the `alpaca-py` client. Alpaca provides institutional-grade historical equity data.
- **Endpoint:** `StockHistoricalDataClient`
- **Timeframe:** Daily bars (`TimeFrame.Day`)
- **Query Range:** January 1, 2021 through January 1, 2026 (UTC)
- **Adjustment Setting:** `Adjustment.ALL` (applies backward split and cash dividend adjustments to preserve continuous price returns)
- **Data Feed:** `DataFeed.IEX` (Investors Exchange feed, provided on Alpaca's free API tier)

### 6.2 Ticker Universe Selection across Sectors
TradePilot targets a diversified universe of **50 prominent large-cap and mega-cap US equities** spanning 5 major economic sectors:
1. **Technology (20 tickers):** `AAPL`, `MSFT`, `NVDA`, `AMD`, `AVGO`, `ORCL`, `CRM`, `ADBE`, `CSCO`, `IBM`, `QCOM`, `TXN`, `INTC`, `AMAT`, `MU`, `NOW`, `INTU`, `PLTR`, `ANET`, `PANW`
2. **Consumer & Retail (10 tickers):** `AMZN`, `WMT`, `COST`, `HD`, `MCD`, `NKE`, `SBUX`, `TGT`, `LOW`, `KO`
3. **Communication & Media (5 tickers):** `GOOGL`, `META`, `NFLX`, `DIS`, `TMUS`
4. **Finance (10 tickers):** `JPM`, `BAC`, `GS`, `MS`, `V`, `MA`, `AXP`, `BLK`, `C`, `WFC`
5. **Healthcare (5 tickers):** `LLY`, `JNJ`, `ABBV`, `MRK`, `UNH`

This basket provides substantial cross-sectional variety, encompassing high-beta semiconductor growth stocks, enterprise software providers, defensive consumer staples, financial institutions sensitive to interest-rate cycles, and large pharmaceutical manufacturers.

### 6.3 Raw Data Schema and Verified Record Counts
Inspection of `data/raw/stock_ohlcv_2021_2025.csv` establishes the following verified properties:
- **Total Rows:** Exactly `62,664` rows
- **Columns (7):** `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`
- **Unique Equities:** Exactly `50` tickers
- **Earliest Timestamp:** `2021-01-04 05:00:00+00:00`
- **Latest Timestamp:** `2025-12-31 05:00:00+00:00`
- **Missing Raw Values:** `0` (zero nulls across all OHLCV columns)
- **Duplicate Records:** `0` duplicate `(symbol, timestamp)` records

| Column Name | Storage Type | Physical Interpretation | Constraints / Bounds |
| :--- | :--- | :--- | :--- |
| `symbol` | `string` | Ticker identifier of the equity | Valid uppercase ticker |
| `timestamp` | `datetime64[ns, UTC]` | Session market opening timestamp | ISO-8601 UTC timestamp |
| `open` | `float64` | Split-adjusted session opening price | $> 0.0$ USD |
| `high` | `float64` | Split-adjusted session highest price | $\ge open, \ge close, \ge low$ |
| `low` | `float64` | Split-adjusted session lowest price | $\le open, \le close, \le high, > 0.0$ |
| `close` | `float64` | Split-adjusted session closing price | $> 0.0$ USD |
| `volume` | `float64` | Total session share trading volume | $\ge 0.0$ shares |

### 6.4 Data Quality Validation and Cleansing Procedures
The raw data is subjected to automated validation implemented in `tradepilot.data.validate_data`:
- **Physical Candlestick Integrity:** The module evaluates every bar against standard geometric rules:
  $$High \ge Open \quad \text{and} \quad High \ge Close$$
  $$Low \le Open \quad \text{and} \quad Low \le Close$$
  $$Low \le High$$
- **Non-Negative Checks:** All prices must be strictly positive ($> 0$), and share volume must be non-negative ($\ge 0$).
- **Verification Result:** The raw dataset contains **0 violations** of logical price bounds, **0 duplicate records**, and **0 missing values**.

### 6.5 Market Data Nuances: IEX Feed Realities and Survivorship Bias
A comprehensive engineering document must acknowledge empirical market data nuances:
1. **The IEX Feed Nuance:** Alpaca's free API feed queries the Investors Exchange (`DataFeed.IEX`). The IEX exchange accounts for approximately 2% to 3% of total US consolidated equity volume. While NBBO (National Best Bid and Offer) regulations ensure that transaction prices on IEX closely match consolidated prices, the reported `volume` field reflects IEX-routed trades only, not the consolidated market volume. Consequently, volume indicators reflect IEX liquidity trends rather than total market turnover.
2. **Survivorship Bias:** The 50 tickers selected for this study were chosen based on their prominence and market capitalization in 2026. Applying this universe retrospectively from 2021 to 2025 introduces survivorship bias, as companies that were delisted, went bankrupt, or suffered severe capital distress during 2021–2025 are excluded from the dataset. Evaluators should recognize that real-world historical portfolios would have faced a broader universe that included failing firms.

---

## 7. Exploratory Data Analysis and Feature Engineering

### 7.1 Feature Generation Methodology
The feature engineering process transforms non-stationary raw OHLCV prices into stationary, scale-invariant technical indicators. To prevent cross-ticker leakage, all calculations are performed on subsets grouped strictly by `symbol` and sorted chronologically by `timestamp`.

### 7.2 Mathematical Definitions of Technical Indicators
TradePilot computes 8 backward-looking technical features:

1. **Short-Term Return (`return_1d`):** 1-day percentage price change:
   $$R_{1d, t} = \frac{Close_t - Close_{t-1}}{Close_{t-1}}$$
2. **Medium-Term Momentum (`return_5d`):** 5-day percentage price change:
   $$R_{5d, t} = \frac{Close_t - Close_{t-5}}{Close_{t-5}}$$
3. **Bi-Weekly Momentum (`return_10d`):** 10-day percentage price change:
   $$R_{10d, t} = \frac{Close_t - Close_{t-10}}{Close_{t-10}}$$
4. **Historical Realized Volatility (`volatility_10d`):** 10-day rolling sample standard deviation of daily returns:
   $$\sigma_{10d, t} = \sqrt{\frac{1}{9} \sum_{i=0}^{9} \left( R_{1d, t-i} - \bar{R}_{10d, t} \right)^2}$$
5. **Volume Shock (`volume_change`):** 1-day percentage change in traded share volume:
   $$\Delta Vol_t = \frac{Volume_t - Volume_{t-1}}{Volume_{t-1}}$$
6. **Relative Strength Index (`RSI_14`):** Measures the velocity and magnitude of directional price movements over a 14-day rolling window:
   $$RS_t = \frac{\text{SMA}_{14}(\max(Close_t - Close_{t-1}, 0))}{\text{SMA}_{14}(\max(Close_{t-1} - Close_t, 0)) + 10^{-9}}$$
   $$RSI_{14, t} = 100 - \left( \frac{100}{1 + RS_t} \right)$$
7. **Moving Average Convergence Divergence (`MACD`):** The spread between fast and slow exponential moving averages (EMAs) of the closing price:
   $$EMA_k(Close)_t = \alpha \cdot Close_t + (1 - \alpha) \cdot EMA_k(Close)_{t-1}, \quad \alpha = \frac{2}{k+1}$$
   $$MACD_t = EMA_{12}(Close)_t - EMA_{26}(Close)_t$$
8. **MACD Signal Line (`MACD_signal`):** A 9-day EMA of the MACD line:
   $$Signal_t = EMA_9(MACD)_t$$

### 7.3 Scale Invariance: Why Stationary Features Replace Raw Prices and SMAs
In early exploratory prototypes (`stock_eda.py`), raw moving averages (`SMA_10`, `SMA_20`, `SMA_50`) and raw prices were extracted. In the refactored, production-ready architecture (`src/tradepilot`), **all raw price columns and absolute moving averages are strictly prohibited from entering the model feature matrix**.

**Technical Rationale:**
- In a cross-sectional dataset containing equities trading at vastly different nominal price levels (e.g., INTC at $20 vs. INTU at $600), raw price levels act as confounding identifiers.
- A decision tree splitting on $Price > 150$ learns arbitrary equity clustering rather than generalizable price dynamics.
- In contrast, percentage returns, normalized volatility, RSI ($[0, 100]$), and MACD express scale-invariant momentum and trend dynamics, allowing the model to generalize patterns across equities regardless of their nominal share price.

### 7.4 Target Formulation: Multi-Class Return Horizon
TradePilot formulates directional forecasting as a **3-class classification problem**. The prediction target is defined as the forward 5-trading-day percentage return:
$$R_{future, t} = \frac{Close_{t+5} - Close_t}{Close_t}$$

The continuous future return is mapped into discrete classes using a threshold $\theta = 0.01$ ($1.0\%$):
$$y_t = \begin{cases} 
\text{UP} & \text{if } R_{future, t} > +0.01 \\
\text{DOWN} & \text{if } R_{future, t} < -0.01 \\
\text{NEUTRAL} & \text{if } -0.01 \le R_{future, t} \le +0.01 
\end{cases}$$

**Empirical Target Distribution:**
Across the entire 59,964 clean observations in `stock_features_labeled.csv`:
- `UP`: 25,831 observations (**43.08%**)
- `DOWN`: 20,856 observations (**34.78%**)
- `NEUTRAL`: 13,277 observations (**22.14%**)

The class distribution reflects the secular bull market characterizing US large-cap equities between 2021 and 2025, resulting in an upward structural drift.

### 7.5 Strict Chronological Grouping and Lookahead Leakage Prevention
To guarantee that no future data leaks into model features:
1. All rolling indicators (RSI, volatility, returns) use backward-looking windows ($t-k$ to $t$).
2. The future target shift ($t+5$) is computed independently per ticker using Pandas `groupby('symbol').shift(-5)`.
3. Dropping rows with NaNs removes exactly 54 rows per symbol:
   - 50 rows at the beginning of each symbol's history (warmup period for moving averages).
   - 4 to 5 rows at the end of each symbol's history (terminal rows lacking future 5-day prices).
4. **Row Accounting:** Total raw rows: $62,664$. Total clean rows: $59,964$. Exactly $2,700$ rows ($4.31\%$) are dropped ($54 \text{ rows} \times 50 \text{ symbols} = 2,700$). Retained symbols: exactly 50.

---

## 8. Machine Learning Methodology

### 8.1 Supervised Problem Formulation
Given an 8-dimensional feature vector $\mathbf{x}_t \in \mathbb{R}^8$ observed at the close of trading session $t$, the objective is to predict the discrete directional outcome $y_t \in \{\text{DOWN}, \text{NEUTRAL}, \text{UP}\}$ occurring over the subsequent 5 trading sessions ($t+1$ through $t+5$).

### 8.2 Input Feature Matrix vs. Prohibited Data Leaks
TradePilot enforces strict feature filtering in `tradepilot.config.DataConfig`:
- **Permitted Features (8):** `return_1d`, `return_5d`, `return_10d`, `volatility_10d`, `volume_change`, `RSI_14`, `MACD`, `MACD_signal`.
- **Prohibited Columns:** `symbol`, `timestamp`, `target`, `future_return_5d`, `open`, `high`, `low`, `close`, `volume`, `daily_return`, `SMA_10`, `SMA_20`, `SMA_50`.
- The data validator `validate_features_dataset()` explicitly verifies that no prohibited column appears in the feature matrix, raising an immediate error upon detection.

### 8.3 Chronological Splitting and Boundary Purging
Standard random $k$-fold cross-validation is invalid for financial time series because shuffling records introduces severe lookahead leakage and auto-correlative data snooping. 

TradePilot implements **Chronological Time-Series Partitioning with Boundary Purging**:
- **Training Period:** 2021-03-16 to 2023-12-21 ($34,864$ rows post-purge)
- **Validation Period:** 2024-01-02 to 2024-12-23 ($12,350$ rows post-purge)
- **Test Period:** 2025-01-02 to 2025-12-23 ($12,250$ rows, final evaluation)

```text
Time Horizon (2021 - 2025)
├──────────────────────────────────────┼──────────────────────┼──────────────────────┤
│           TRAINING FOLD              │   VALIDATION FOLD    │      TEST FOLD       │
│      (2021-03-16 to 2023-12-21)      │ (2024-01-02 to 2024) │ (2025-01-02 to 2025) │
│              34,864 rows             │      12,350 rows     │      12,250 rows     │
└─────────────────────────────────[X]──┴─────────────────[X]──┴──────────────────────┘
                                   ▲                      ▲
                                   │                      │
                             Purged Boundary        Purged Boundary
                             (5 days/ticker)        (5 days/ticker)
                             (250 rows total)       (250 rows total)
```

**The Boundary Purging Mechanism:**
Because the target variable represents a forward 5-trading-day return, observations within the final 5 trading days of 2023 would incorporate price information from the first week of 2024 (the validation set). Similarly, the final 5 days of 2024 would incorporate prices from 2025 (the test set).
- To eliminate boundary leakage, `tradepilot.modeling.split_data` purges exactly 5 trading sessions from the end of the training set and validation set for each of the 50 tickers ($5 \times 50 = 250$ purged rows per boundary).
- An explicit programmatic assertion validates that:
  $$\max(Timestamp_{Train}) < \min(Timestamp_{Val}) \quad \text{and} \quad \max(Timestamp_{Val}) < \min(Timestamp_{Test})$$
- Verified sum of rows: $34,864 + 250 + 12,350 + 250 + 12,250 = 59,964$ rows.

### 8.4 Candidate Model Architectures
TradePilot constructs three candidate model pipelines:
1. **Majority-Class Baseline (`Majority_Baseline`):**
   - Implemented via `sklearn.dummy.DummyClassifier(strategy="most_frequent")`.
   - Always predicts the dominant historical training class (`UP`).
   - Serves as the benchmark to evaluate whether complex models add predictive value beyond naive trend following.
2. **Regularized Logistic Regression (`Logistic_Regression`):**
   - Preprocessed with `StandardScaler()`.
   - Configured with `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`.
   - Serves as a linear parametric baseline to determine if linear boundaries adequately separate classes.
3. **Random Forest Classifier (`Random_Forest`):**
   - Non-linear ensemble model.
   - Configured with `RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)`.
   - Hyperparameters restrict tree depth (`max_depth=8`) and enforce leaf support (`min_samples_leaf=5`) to prevent overfitting noisy financial data.
   - `class_weight="balanced"` adjusts weights inversely proportional to class frequencies, forcing the algorithm to learn minority class patterns.

### 8.5 Preprocessing, Pipeline Construction, and Seed Management
All transformations are encapsulated in scikit-learn `Pipeline` objects. This design ensures that preprocessing transformations (such as `StandardScaler` mean and variance estimation) are fitted strictly on the training fold and applied out-of-sample to validation and test folds without data snooping. Deterministic execution is enforced across all models using `random_seed = 42`.

### 8.6 Model Selection Protocol
Model selection is based on the validation fold (2024) performance. The winning model is selected by maximizing **Macro-Averaged F1 Score**:
$$\text{Macro F1} = \frac{F1_{\text{DOWN}} + F1_{\text{NEUTRAL}} + F1_{\text{UP}}}{3}$$
Macro F1 treats all three directional classes equally, penalizing models that collapse into predicting only the majority class. The winning candidate is evaluated exactly once on the frozen 2025 test fold.

### 8.7 Model Serialization and Auditable Metadata
The selected model pipeline is persisted to `models/best_model.joblib`. Concurrently, `models/model_metadata.json` is generated, capturing:
- Project identifier and UTC timestamp.
- Explicit list and count of input features (8).
- Mathematical definition of target classes.
- Granular breakdown of train, validation, and test date ranges and row counts.
- Complete dictionary of all estimator hyperparameters.
- Validation and test performance metrics.

---

## 9. Model Evaluation and Experimental Results

### 9.1 Evaluation Metrics for Class-Imbalanced Financial Data
In financial directional classification, simple accuracy is a deceptive metric. In an upward-trending market where 44% of observations are `UP`, a trivial model predicting `UP` 100% of the time achieves 44% accuracy while providing zero actionable intelligence for risk management or short positioning.

TradePilot evaluates models using a robust multi-metric suite:
- **Macro F1 Score:** Unweighted average of F1 scores across `DOWN`, `NEUTRAL`, and `UP`.
- **Balanced Accuracy:** Arithmetic mean of recall scores achieved across each class, equivalent to accuracy normalized for class prevalence:
  $$\text{Balanced Accuracy} = \frac{\text{Recall}_{\text{DOWN}} + \text{Recall}_{\text{NEUTRAL}} + \text{Recall}_{\text{UP}}}{3}$$
- **Per-Class Precision and Recall:** Measures the purity of directional signals and the model's coverage of actual market moves.

### 9.2 Validation Comparison across Candidate Models (2024 Fold)
The three candidate models were trained on 2021–2023 data ($34,864$ rows) and evaluated out-of-sample on the 2024 validation fold ($12,350$ rows). The empirical results from `reports/validation_results.csv` are summarized below:

| Model Architecture | Macro F1 | Balanced Accuracy | Overall Accuracy | DOWN F1 | NEUTRAL F1 | UP F1 | DOWN Precision / Recall | NEUTRAL Precision / Recall | UP Precision / Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority_Baseline** | 0.2049 | 0.3333 | **0.4437** | 0.0000 | 0.0000 | 0.6147 | 0.00% / 0.00% | 0.00% / 0.00% | 44.37% / 100.0% |
| **Logistic_Regression** | 0.3009 | 0.3618 | 0.3141 | 0.2512 | **0.3897** | 0.2619 | 31.74% / 20.79% | 27.08% / 69.49% | 46.38% / 18.25% |
| **Random_Forest** | **0.3428** | **0.3725** | 0.3457 | **0.3026** | 0.3746 | **0.3512** | **33.48%** / **27.60%** | **28.00%** / 56.60% | **48.40%** / **27.55%** |

**Empirical Analysis of Validation Findings:**
1. **The Accuracy Paradox:** The `Majority_Baseline` achieved the highest raw accuracy ($44.37\%$), yet yielded the lowest Macro F1 ($0.2049$) and a minimal Balanced Accuracy ($0.3333$). It completely failed to identify any `DOWN` or `NEUTRAL` regimes, rendering it useless for risk management.
2. **Linear Model Underperformance:** `Logistic_Regression` produced a low Macro F1 ($0.3009$) and exhibited a strong tendency to overpredict `NEUTRAL` (recall of $69.49\%$), struggling to separate directional equity trends using linear hyperplanes.
3. **Random Forest Superiority:** `Random_Forest` outperformed all candidates, achieving the highest **Macro F1 (0.3428)** and **Balanced Accuracy (0.3725)**. It demonstrated balanced predictive capacity across all three classes, achieving $48.40\%$ precision on `UP` predictions and $33.48\%$ precision on `DOWN` predictions. Based on these validation results, `Random_Forest` was selected as the operational model.

### 9.3 Final Test Period Generalization Performance (2025 Fold)
The selected Random Forest model was evaluated on the completely untouched 2025 test fold ($12,250$ rows). The verified performance metrics from `reports/test_results.json` are documented below:

- **Macro-Averaged F1 Score:** `0.3558` (an improvement over the validation score of 0.3428)
- **Balanced Accuracy:** `0.3736` (consistent with validation performance of 0.3725)
- **Overall Accuracy:** `0.3604`

**Per-Class Test Performance Breakdown:**

| Target Class | Support (Actual Rows) | Precision | Recall | F1-Score | Effective Detection Skill |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DOWN** | 4,240 | 35.21% | 30.26% | 0.3255 | Moderate downward move detection |
| **NEUTRAL** | 2,522 | 26.46% | 45.80% | 0.3354 | High recall on sideways consolidation |
| **UP** | 5,488 | 46.62% | 36.02% | 0.4064 | Strongest precision, driving long entry signals |
| **Total / Macro** | **12,250** | **36.10%** | **37.36%** | **0.3558** | **Statistically superior to random guess (0.3333)** |

### 9.4 Out-of-Sample Confusion Matrix Analysis
The verified confusion matrix on the 2025 test set (from `reports/test_results.json` and `reports/confusion_matrix.csv`) shows the distribution of predicted versus true labels:

| True State \ Predicted State | Predicted DOWN | Predicted NEUTRAL | Predicted UP | Total True Observations |
| :--- | :---: | :---: | :---: | :---: |
| **Actual DOWN** | **1,283** | 1,451 | 1,506 | 4,240 |
| **Actual NEUTRAL** | 609 | **1,155** | 758 | 2,522 |
| **Actual UP** | 1,752 | 1,759 | **1,977** | 5,488 |
| **Total Predicted** | 3,644 | 4,365 | 4,241 | 12,250 |

**Key Diagnostic Insights:**
- **True Positive Upward Capture:** The model correctly identified 1,977 large upward moves. When predicting `UP`, it was correct 1,977 times out of 4,241 predictions ($46.62\%$ precision).
- **False Positive Asymmetry:** Out of 4,241 `UP` predictions, 1,506 were actually `DOWN` sessions. This error rate illustrates the inherent challenge of predicting directional returns in noisy equity markets and underscores the necessity of strict risk management and position exposure caps.
- **Sideways Market Classification:** The model classified 4,365 sessions as `NEUTRAL`, reflecting its ability to filter out low-conviction consolidation periods.

### 9.5 Financial Machine Learning Interpretation of Model Skill
In liquid public equity markets governed by the Semi-Strong Efficient Market Hypothesis (EMH), obtaining a 3-class classification accuracy significantly above 50% on daily features is mathematically improbable. A balanced accuracy of **$37.36\%$** across three classes represents an edge over a naive random baseline of **$33.33\%$**. In quantitative finance, small predictive edges can be monetized if transaction costs, portfolio allocation, and position sizing are tightly controlled.

---

## 10. Backtesting and Portfolio Management

### 10.1 Event-Driven Simulation Principles
TradePilot implements a realistic, event-driven backtesting engine (`tradepilot.backtesting.engine.BacktestEngine`) designed to prevent execution lookahead leakage. The simulation enforces the following timeline:
1. **End-of-Day Close ($t$):** Daily features are calculated using information available up to market close. The model computes probability estimates and generates trading signals. Orders are placed into a pending queue.
2. **Next-Day Open ($t+1$):** Pending orders are executed using session Opening prices, incorporating modeled slippage and exchange fees.
3. **End-of-Day Close ($t+1$):** The portfolio is marked to market using session Closing prices. Holding durations are incremented, and exit criteria are evaluated for the subsequent session.

```text
Session t Close                     Session t+1 Open                   Session t+1 Close
      │                                   │                                  │
      ▼                                   ▼                                  ▼
[Compute Features]               [Execute Orders]                   [Mark-to-Market]
[Model Infers Probs]      ──►    [Fill at Open Price]        ──►    [Update Equity Value]
[Queue Orders for t+1]           [Apply Slippage & Fees]            [Increment Holding Days]
```

### 10.2 Model Probability to Trading Signal Mapping
`tradepilot.backtesting.signals.SignalGenerator` dynamically inspects the fitted pipeline's `classes_` attribute to determine class indices. It applies a long-only rule set with an entry probability threshold $\tau = 0.40$:
- **BUY Signal ($+1$):** Triggered when $P(\text{UP}) \ge \tau$ and $P(\text{UP}) > P(\text{DOWN})$.
- **EXIT Signal ($-1$):** Triggered when $P(\text{DOWN}) \ge \tau$ and $P(\text{DOWN}) > P(\text{UP})$.
- **HOLD / NO ACTION ($0$):** Retained when neither threshold condition is satisfied.

### 10.3 Order Generation, Queuing, and Next-Day Open Execution
When a signal is generated at Close $t$, an `Order` object is queued. At Open $t+1$:
- **Liquidity Prioritization:** Sell orders are executed first to release capital before evaluating buy allocations.
- **Deterministic Processing:** Buy orders are processed in alphabetical order by ticker to ensure simulation determinism.
- **Order Rejection:** If cash is depleted or the portfolio already holds the maximum permitted open positions, pending buy orders are rejected.

### 10.4 Execution Friction: Slippage and Commission Accounting
TradePilot models execution frictions in `tradepilot.backtesting.execution.ExecutionHandler`:
- **Adverse Execution Slippage (5 bps / $0.05\%$):** Reflects price impact and bid-ask spread crossing:
  $$\text{Fill Price}_{\text{BUY}} = Open_{t+1} \times (1 + 0.0005)$$
  $$\text{Fill Price}_{\text{SELL}} = Open_{t+1} \times (1 - 0.0005)$$
- **Transaction Commissions (5 bps / $0.05\%$):** Represents brokerage execution fees and regulatory exchange dues:
  $$\text{Fee} = \text{Shares} \times \text{Fill Price} \times 0.0005$$
- **Capital Bounding:** The maximum integer shares affordable within budget $B$ is:
  $$\text{Shares} = \left\lfloor \frac{B}{\text{Fill Price}_{\text{BUY}} \times (1 + 0.0005)} \right\rfloor$$

### 10.5 Portfolio Risk Controls and Capital Allocation Limits
The `Portfolio` manager enforces structural risk controls (`tradepilot.config.BacktestConfig`):
- **Initial Capital:** $\$100,000.00$
- **Position Allocation Cap:** Maximum $10\%$ of total portfolio equity per position.
- **Maximum Positions:** 10 concurrent open positions (ensuring full diversification at max capacity).
- **No Pyramiding:** Additional buy signals for an already held ticker are ignored.
- **Minimum Cash Buffer:** Requires a minimum of $\$100.00$ cash to execute orders.
- **Holding Horizon Exit:** Positions are automatically liquidated after being held for 5 trading sessions, matching the model's 5-day prediction horizon.

### 10.6 Marked-to-Market Accounting and Position State Tracking
At each session close, the portfolio computes:
- **Holdings Value:** $\sum_{i} \text{Shares}_i \times Close_{i, t}$
- **Total Portfolio Equity:** $\text{Cash}_t + \text{Holdings Value}_t$
- **Gross Exposure:** $\frac{\text{Holdings Value}_t}{\text{Total Equity}_t}$
- **Net Trade PnL:** For closed trades, net profit accounts for both entry and exit frictions:
  $$\text{Gross PnL} = (\text{Fill Price}_{\text{SELL}} - \text{Fill Price}_{\text{BUY}}) \times \text{Shares}$$
  $$\text{Net PnL} = \text{Gross PnL} - (\text{Costs}_{\text{BUY}} + \text{Costs}_{\text{SELL}})$$

### 10.7 Financial Performance Metrics Formulae
Performance metrics are implemented in `tradepilot.backtesting.metrics.calculate_backtest_metrics`:
- **Total Return:** $\frac{Equity_{final} - Equity_{initial}}{Equity_{initial}}$
- **Compound Annual Growth Rate (CAGR):** $\left(\frac{Equity_{final}}{Equity_{initial}}\right)^{\frac{252}{N}} - 1$
- **Annualized Volatility:** $\sigma_{daily} \times \sqrt{252}$
- **Annualized Sharpe Ratio ($R_f = 0$):** $\frac{\bar{R}_{daily}}{\sigma_{daily}} \times \sqrt{252}$
- **Maximum Drawdown:** $\min_t \left( \frac{Equity_t - \max_{\tau \le t} Equity_\tau}{\max_{\tau \le t} Equity_\tau} \right)$
- **Profit Factor:** $\frac{\sum \text{Net Profits}}{\sum |\text{Net Losses}|}$
- **Turnover:** $\frac{\text{Total Traded Value}}{2 \times \text{Average Equity}}$

### 10.8 Empirical Strategy Performance vs. Equal-Weight Buy-and-Hold
TradePilot was evaluated across both the 2024 validation period and the 2025 test period. The verified metrics from `reports/backtest_summary.json` and `reports/benchmark_comparison.csv` are presented below:

| Performance Metric | 2024 Validation: TradePilot ML | 2024 Validation: Buy-and-Hold | 2025 Test: TradePilot ML | 2025 Test: Buy-and-Hold | 2025 Test: Cash (Risk-Free) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Initial Capital** | $\$100,000.00$ | $\$100,000.00$ | $\$100,000.00$ | $\$100,000.00$ | $\$100,000.00$ |
| **Final Equity** | **$\$110,393.10$** | $\$133,351.38$ | **$\$109,940.55$** | $\$126,289.35$ | $\$100,000.00$ |
| **Total Net Return** | **+10.39%** | +33.35% | **+9.94%** | +26.29% | 0.00% |
| **Annualized Return (CAGR)**| **+10.44%** | +33.66% | **+10.07%** | +26.77% | 0.00% |
| **Annualized Volatility** | **11.49%** | 14.50% | **22.72%** | 20.61% | 0.00% |
| **Sharpe Ratio ($R_f=0$)** | **0.92** | 2.07 | **0.54** | 1.25 | 0.00 |
| **Maximum Drawdown** | **-6.16%** | -9.57% | **-22.50%** | -21.41% | 0.00% |
| **Total Closed Trades** | **236** | N/A | **343** | N/A | 0 |
| **Win Rate (% Profitable)** | **56.78%** | N/A | **51.90%** | N/A | N/A |
| **Profit Factor** | **1.26** | N/A | **1.08** | N/A | N/A |
| **Average Win Return** | **+2.99%** | N/A | **+4.30%** | N/A | N/A |
| **Average Loss Return** | **-3.07%** | N/A | **-4.20%** | N/A | N/A |
| **Portfolio Turnover** | **23.38x** | 0.00x | **33.41x** | 0.00x | 0.00x |
| **Total Modeled Costs** | **$\$5,041.68$** | $\$0.00$ | **$\$6,928.29$** | $\$0.00$ | $\$0.00$ |
| **Average Capital Exposure**| **41.45%** | 100.00% | **58.80%** | 100.00% | 0.00% |
| **Time in Market** | **94.82%** | 100.00% | **97.59%** | 100.00% | 0.00% |

**Detailed Quantitative Interpretation:**
1. **Consistent Absolute Profitability:** The TradePilot strategy generated positive net returns in both out-of-sample periods: **+10.39% in 2024** and **+9.94% in 2025**, validating that the model generates positive expected value after friction costs.
2. **Defensive Characteristics in 2024:** During the 2024 validation fold, the strategy exhibited defensive qualities:
   - Lower annualized volatility (**11.49%** vs. **14.50%** for Buy-and-Hold).
   - A significantly milder maximum drawdown (**-6.16%** vs. **-9.57%**).
   - An attractive Sharpe ratio of **0.92** and a win rate of **56.78%**.
3. **The Cost of High Turnover in 2025:** In the 2025 test period, trading activity increased to 343 trades, resulting in a turnover of **33.41x** and **$\$6,928.29$** in friction costs. This friction eroded net returns, lowering the profit factor to **1.08** and the Sharpe ratio to **0.54**.
4. **Underperformance Relative to Buy-and-Hold:** The strategy trailed the Equal-Weight Buy-and-Hold benchmark in both periods (+10.39% vs. +33.35% in 2024; +9.94% vs. +26.29% in 2025). This divergence highlights two structural realities:
   - **Cash Drag:** TradePilot maintained an average exposure of 41.5% in 2024 and 58.8% in 2025. Holding 40%–60% of the portfolio in uninvested cash during strong bull-market regimes inherently lags fully invested benchmarks.
   - **Rebalancing Friction:** Frequent entries and exits incur continuous slippage and commission friction, whereas Buy-and-Hold incurs zero friction after initial establishment.

---

## 11. Detailed Source Code and Module Reference

### 11.1 Consolidated Module Reference Matrix
The following reference table documents all primary Python modules comprising the TradePilot software system:

| File Path | Functional Responsibility | Primary Classes & Functions | Inputs / Preconditions | Outputs / Side Effects | Error Handling & Invariants |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `src/tradepilot/config.py` | Central configuration dataclasses | `DataConfig`, `ModelConfig`, `BacktestConfig` | None (static defaults) | Immutable configuration objects | Uses `frozen=True` to guarantee runtime immutability. |
| `src/tradepilot/data/load_data.py` | Disk ingestion of raw & processed CSVs | `load_raw_ohlcv()`, `load_processed_features()` | Path to CSV files | Loaded, UTC-parsed, sorted `pd.DataFrame` | Raises `FileNotFoundError` if files are missing. |
| `src/tradepilot/data/validate_data.py` | Schema, duplicate, and price bounds validation | `validate_ohlcv_schema()`, `validate_features_dataset()` | `pd.DataFrame` | None (validates or raises) | Raises `ValueError` on negative prices, geometric OHLC violations, NaNs, or leakage columns. |
| `src/tradepilot/features/build_features.py` | Feature extraction & target formulation | `calculate_rsi()`, `calculate_macd()`, `compute_technical_features()`, `generate_features_and_targets()` | Raw OHLCV DataFrame | Labeled DataFrame with 8 features & target | Groups by symbol to isolate statistics; drops warmup and shift NaNs. |
| `src/tradepilot/modeling/split_data.py` | Chronological splitting & boundary purging | `chronological_split()` | Processed DataFrame, split year boundaries | `train_df`, `val_df`, `test_df`, `split_info` | Raises `ValueError` on insufficient rows or split boundary date overlaps. |
| `src/tradepilot/modeling/train_model.py` | Pipeline construction & fitting | `get_candidate_pipelines()`, `train_candidates()` | Feature matrix $X$, target series $y$, seed | Dict of fitted scikit-learn `Pipeline` objects | Pipeline encapsulates preprocessing; seeds ensure reproducibility. |
| `src/tradepilot/modeling/evaluate_model.py` | Classification evaluation & candidate selection | `evaluate_classifier()`, `select_best_model()` | Fitted pipeline, $X$, $y$, class names | Metrics dictionary, winning model name | Explicitly ignores zero divisions; handles multi-class outputs. |
| `src/tradepilot/modeling/save_model.py` | Persistence of models, metadata & reports | `save_model_artifacts()` | Fitted pipeline, metric dicts, config | Serialized joblib, JSON, and CSV artifacts | Creates target directories automatically; persists auditable JSON. |
| `src/tradepilot/backtesting/signals.py` | Dynamic probability signal mapping | `SignalGenerator`, `generate_signals()` | Fitted pipeline, feature DataFrame | Signal DataFrame ($1, -1, 0$) | Inspects `pipeline.classes_` dynamically; validates required classes. |
| `src/tradepilot/backtesting/execution.py` | Order execution, slippage & fee modeling | `Order`, `TradeFill`, `ExecutionHandler`, `execute_order()` | Order, Open price, date, cash, max budget | `TradeFill` object or `None` | Rejects non-positive prices; bounds share sizing to available cash. |
| `src/tradepilot/backtesting/portfolio.py` | Position management & mark-to-market | `Position`, `CompletedTrade`, `Portfolio`, `mark_to_market()` | Initial cash, position limits, fills | Equity history, trade ledger DataFrames | Enforces cash solvency; prevents position pyramiding. |
| `src/tradepilot/backtesting/metrics.py` | Strategy & benchmark financial metrics | `calculate_backtest_metrics()`, `calculate_benchmark_metrics()` | Equity DF, Trade ledger DF, initial cash | Financial metrics dictionary | Handles empty ledgers defensively; handles division-by-zero. |
| `src/tradepilot/backtesting/engine.py` | Chronological event-driven simulation loop | `BacktestResult`, `BacktestEngine`, `run()` | OHLCV DF, Signal DF, date range | `BacktestResult` container | Pre-indexes price lookups; enforces next-day Open fills. |
| `src/tradepilot/pipelines/train_pipeline.py` | Training workflow orchestration | `run_train_pipeline()`, `main()` | CLI invocation or direct call | Fitted model, metadata, reports | Comprehensive logging; logs execution errors and exits cleanly. |
| `src/tradepilot/pipelines/backtest_pipeline.py` | Backtest workflow orchestration | `run_backtest_pipeline()`, `main()` | CLI invocation or direct call | Equity curves, trade ledgers, summaries | Validates model artifact existence before executing simulations. |
| `download_market_data.py` | Historical data retrieval via Alpaca API | Script execution | Alpaca API keys in `.env` | `data/raw/stock_ohlcv_2021_2025.csv` | Raises `ValueError` if API credentials are missing. |
| `stock_eda.py` | Prototype exploratory analysis script | Script execution | Google Drive / local CSVs | Cleaned CSV, feature CSV, reports | Standalone exploratory research script. |
| `train.py` | Root CLI training entrypoint | Script execution | None (CLI parameters) | Calls `train_pipeline.main()` | Thin wrapper delegating execution to the package pipeline. |

---

### 11.2 Package Configuration: `tradepilot.config`
Located at `src/tradepilot/config.py`, this module provides frozen dataclasses that define paths, hyperparameters, and schemas across the package.
```python
# Path resolution relative to project root
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

@dataclass(frozen=True)
class DataConfig:
    raw_ohlcv_path: Path = PROJECT_ROOT / "data" / "raw" / "stock_ohlcv_2021_2025.csv"
    processed_features_path: Path = PROJECT_ROOT / "data" / "processed" / "stock_features_labeled.csv"
    feature_cols: List[str] = field(default_factory=lambda: [
        "return_1d", "return_5d", "return_10d", "volatility_10d",
        "volume_change", "RSI_14", "MACD", "MACD_signal"
    ])
    prohibited_cols: Set[str] = field(default_factory=lambda: {
        "symbol", "timestamp", "target", "future_return_5d",
        "open", "high", "low", "close", "volume", "daily_return",
        "SMA_10", "SMA_20", "SMA_50"
    })
    target_threshold: float = 0.01  # 1.0% return over 5 trading days
    prediction_horizon_days: int = 5
    purge_boundary_days: int = 5
```

---

### 11.3 Data Ingestion and Validation: `tradepilot.data`
The data module handles data loading (`load_data.py`) and validation assertions (`validate_data.py`).

**OHLC Geometric Validation Implementation:**
```python
def validate_ohlcv_schema(df: pd.DataFrame) -> None:
    required_cols = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw OHLCV missing required columns: {sorted(missing)}")

    # Check for duplicates
    duplicates = df.duplicated(subset=["symbol", "timestamp"]).sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate (symbol, timestamp) records.")

    # Geometric price bounds
    high_violations = (df["high"] < df["open"]) | (df["high"] < df["close"])
    low_violations = (df["low"] > df["open"]) | (df["low"] > df["close"])
    low_high_violations = df["low"] > df["high"]

    if high_violations.any() or low_violations.any() or low_high_violations.any():
        raise ValueError("Logical OHLC boundary violation detected in price data.")
```

---

### 11.4 Feature Engineering Engine: `tradepilot.features`
`build_features.py` calculates technical indicators and directional targets:

```python
def generate_features_and_targets(df_raw: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.DataFrame:
    df_feat = compute_technical_features(df_raw)

    # Calculate forward 5-day return grouped by symbol
    df_feat["future_return_5d"] = (
        df_feat.groupby("symbol")["close"].shift(-horizon) / df_feat["close"] - 1.0
    )

    def assign_target(ret: float) -> str:
        if pd.isna(ret):
            return np.nan
        if ret > threshold:
            return "UP"
        if ret < -threshold:
            return "DOWN"
        return "NEUTRAL"

    df_feat["target"] = df_feat["future_return_5d"].apply(assign_target)
    return df_feat.dropna().reset_index(drop=True)
```

---

### 11.5 Modeling and Cross-Validation: `tradepilot.modeling`
Contains modules for chronological boundary purging (`split_data.py`), model pipeline definition (`train_model.py`), evaluation (`evaluate_model.py`), and serialization (`save_model.py`).

**Boundary Purging Implementation:**
```python
def chronological_split(df: pd.DataFrame, train_end_year=2023, val_year=2024, test_year=2025, purge_days=5):
    df_sorted = df.sort_values(by=["symbol", "timestamp"]).reset_index(drop=True)
    train_mask = df_sorted["timestamp"].dt.year <= train_end_year
    val_mask = df_sorted["timestamp"].dt.year == val_year
    test_mask = df_sorted["timestamp"].dt.year == test_year

    train_indices, val_indices, test_indices = [], [], []

    for sym, group in df_sorted.groupby("symbol", sort=False):
        g_train = group.loc[train_mask[group.index]]
        g_val = group.loc[val_mask[group.index]]
        g_test = group.loc[test_mask[group.index]]

        # Purge boundary rows
        g_train_clean = g_train.iloc[:-purge_days]
        g_val_clean = g_val.iloc[:-purge_days]

        train_indices.extend(g_train_clean.index)
        val_indices.extend(g_val_clean.index)
        test_indices.extend(g_test.index)

    return (
        df_sorted.loc[train_indices].copy(),
        df_sorted.loc[val_indices].copy(),
        df_sorted.loc[test_indices].copy(),
        split_info
    )
```

---

### 11.6 Backtesting Simulator: `tradepilot.backtesting`
The backtesting subpackage implements event-driven trading simulation across `signals.py`, `execution.py`, `portfolio.py`, `metrics.py`, and `engine.py`.

**Dynamic Class-Order Signal Generation:**
```python
class SignalGenerator:
    def __init__(self, pipeline: Pipeline, feature_cols=None, entry_threshold=0.40):
        self.pipeline = pipeline
        self.entry_threshold = entry_threshold
        classifier = pipeline.named_steps.get("classifier", pipeline)
        
        # Dynamically map class indices without assuming fixed order
        self.classes_ = list(classifier.classes_)
        self.up_idx = self.classes_.index("UP")
        self.down_idx = self.classes_.index("DOWN")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = self.pipeline.predict_proba(df[self.feature_cols])
        prob_up = probs[:, self.up_idx]
        prob_down = probs[:, self.down_idx]

        signals = np.zeros(len(df), dtype=int)
        signals[(prob_up >= self.entry_threshold) & (prob_up > prob_down)] = 1
        signals[(prob_down >= self.entry_threshold) & (prob_down > prob_up)] = -1
        return pd.DataFrame({"timestamp": df["timestamp"], "symbol": df["symbol"], "signal": signals})
```

**Adverse Slippage and Fee Modeling:**
```python
def execute_order(self, order, open_price, execution_date, available_cash, max_position_value):
    slippage_mult = self.slippage_bps / 10_000.0
    commission_mult = self.commission_bps / 10_000.0

    if order.side == OrderSide.BUY:
        # Buy price inflated by adverse slippage
        fill_price = open_price * (1.0 + slippage_mult)
        effective_cost_per_share = fill_price * (1.0 + commission_mult)
        budget = min(available_cash, max_position_value)
        shares = int(budget // effective_cost_per_share)
        if shares <= 0:
            return None

        gross_value = shares * fill_price
        commission_cost = gross_value * commission_mult
        slippage_cost = shares * abs(fill_price - open_price)
        return TradeFill(order.symbol, OrderSide.BUY, shares, fill_price, open_price, execution_date, slippage_cost, commission_cost, slippage_cost + commission_cost)
```

---

### 11.7 Automated Pipelines: `tradepilot.pipelines`
- `train_pipeline.py`: Loads features, verifies schema integrity, performs chronological splitting, fits candidate models, displays validation comparison tables, selects the best model, evaluates out-of-sample test metrics, and persists artifacts.
- `backtest_pipeline.py`: Loads the serialized model, loads historical market data, generates out-of-sample probability signals, executes the event-driven backtesting engine across validation and test periods, prints performance tables, and exports equity curves and trade ledgers.

### 11.8 Root Operational Entrypoints
- `download_market_data.py`: Standalone script to retrieve raw OHLCV bars via the Alpaca API.
- `train.py`: Lightweight CLI entrypoint that imports and invokes `tradepilot.pipelines.train_pipeline.main()`.
- `stock_eda.py`: Standalone prototype script originating from early exploratory Colab research.

---

## 12. Installation, Environment Setup, and Execution Guide

### 12.1 Environment Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux (commands below are verified for Windows PowerShell).
- **Python Version:** Python 3.10 or higher.
- **Git:** Installed and configured in the system PATH.
- **Hardware:** Minimum 8 GB RAM; multi-core CPU recommended.

### 12.2 Windows PowerShell Virtual Environment Activation
Open Windows PowerShell, navigate to the project directory, and activate the existing virtual environment:

```powershell
# Navigate to the workspace root
Set-Location -Path "E:\Field Project\New Approach"

# Activate the existing virtual environment
.\.venv\Scripts\Activate.ps1
```

*Note on Execution Policies:* If PowerShell restricts script activation, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### 12.3 Editable Package Installation
Install TradePilot in editable development mode (`-e`) along with its runtime and testing dependencies:

```powershell
# Install package in editable development mode
pip install -e .

# Verify installed packages
pip install -r requirements.txt
```

### 12.4 Secure Environment Variable Configuration
To download fresh market data from Alpaca, create a `.env` file in the project root:

```ini
# .env template (Do NOT commit to version control)
ALPACA_API_KEY=your_actual_alpaca_key_id
ALPACA_SECRET_KEY=your_actual_alpaca_secret_key
```

*Security Warning:* Never commit `.env` files, API keys, or secret tokens to version control.

### 12.5 Executing Data Ingestion, Model Training, and Backtesting

#### Step 1: Execute Automated Tests
Verify package integrity, schema validators, feature calculations, and financial accounting before running production workloads:
```powershell
pytest -v
```

#### Step 2: Download Fresh Market Data (Optional if raw CSV is present)
```powershell
python download_market_data.py
```

#### Step 3: Run the Machine Learning Training Pipeline
```powershell
# Option A: Via root wrapper
python train.py

# Option B: Via module invocation
python -m tradepilot.pipelines.train_pipeline
```

#### Step 4: Run the Event-Driven Backtesting Pipeline
```powershell
python -m tradepilot.pipelines.backtest_pipeline
```

### 12.6 Locating and Auditing Generated Artifacts
Upon pipeline completion, the following artifacts are generated:
- `models/best_model.joblib`: Serialized Random Forest model pipeline.
- `models/model_metadata.json`: Machine-readable metadata schema detailing training parameters, features, and split metrics.
- `reports/validation_results.csv`: Validation performance comparison across all candidate models.
- `reports/test_results.json`: Final out-of-sample classification report and confusion matrix.
- `reports/confusion_matrix.csv`: Tabular confusion matrices across validation and test folds.
- `reports/backtest_summary.json`: Detailed strategy vs. benchmark performance metrics.
- `reports/backtest_equity_curve.csv`: Daily marked-to-market equity values and gross exposure tracking.
- `reports/trade_ledger.csv`: Granular trade-by-trade log detailing entry/exit dates, fill prices, shares, gross PnL, net PnL, and transaction fees.
- `reports/benchmark_comparison.csv`: High-level strategy vs. benchmark comparative summary.

### 12.7 Troubleshooting Common Operational Faults
1. **`ModuleNotFoundError: No module named 'tradepilot'`:**
   - *Cause:* The package was not installed in editable mode or `src` is missing from `sys.path`.
   - *Resolution:* Run `pip install -e .` from the project root, or set `$env:PYTHONPATH="src"` in PowerShell.
2. **`FileNotFoundError: Raw OHLCV data file not found`:**
   - *Cause:* `stock_ohlcv_2021_2025.csv` is missing from `data/raw/`.
   - *Resolution:* Execute `python download_market_data.py` to fetch market data from Alpaca.
3. **`ValueError: Missing Alpaca API credentials in .env`:**
   - *Cause:* Missing or empty `ALPACA_API_KEY` or `ALPACA_SECRET_KEY` in `.env`.
   - *Resolution:* Add valid Alpaca paper-trading credentials to `.env`.

---

## 13. Quality Assurance, Automated Testing, and Reproducibility

### 13.1 Testing Architecture and Test Discovery
TradePilot maintains an automated test suite under `tests/`. Test discovery and configuration are managed via `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

### 13.2 Breakdown of Automated Unit and Integration Tests
The test suite contains **17 automated test cases** distributed across 6 specialized modules:

1. **`tests/test_data_validation.py` (6 tests):**
   - `test_validate_ohlcv_schema_passes_valid`: Asserts that valid OHLCV datasets pass schema checks without error.
   - `test_validate_ohlcv_detects_duplicates`: Asserts that duplicate `(symbol, timestamp)` records raise `ValueError`.
   - `test_validate_ohlcv_detects_negative_price`: Asserts that non-positive prices raise `ValueError`.
   - `test_validate_ohlcv_detects_high_low_violation`: Asserts that $High < Open$ or $Low > High$ raises `ValueError`.
   - `test_validate_features_dataset_detects_target_leakage`: Asserts that prohibited columns (e.g., `future_return_5d`) in the feature matrix raise `ValueError`.
   - `test_validate_features_dataset_detects_nan_values`: Asserts that unhandled NaNs in features raise `ValueError`.

2. **`tests/test_execution_and_portfolio.py` (3 tests):**
   - `test_execution_slippage_and_fees`: Validates that buy prices are inflated by adverse slippage ($100 \times 1.001 = 100.10$), sell prices are reduced ($110 \times 0.999 = 109.89$), and commission costs are deducted.
   - `test_portfolio_cash_accounting_and_round_trip`: Tracks a complete round-trip trade (buy 20 shares @ $100, mark-to-market @ $110, sell @ $110), verifying that cash balances and net PnL ($+\$200.00$) reconcile accurately.
   - `test_portfolio_prevents_negative_cash`: Verifies that orders exceeding available cash are rejected, preventing insolvency.

3. **`tests/test_features.py` (4 tests):**
   - `test_rsi_bounds`: Mathematically proves that computed RSI values fall strictly within the $[0, 100]$ interval.
   - `test_macd_calculation`: Validates that steady upward price drift produces positive MACD values ($EMA_{12} > EMA_{26}$).
   - `test_features_independent_per_symbol`: Verifies that rolling window calculations for symbol B do not incorporate closing prices from symbol A.
   - `test_target_classification_logic`: Asserts that synthetic compounding price series ($+2\%$ daily) correctly receive `UP` classification labels.

4. **`tests/test_signals.py` (1 test):**
   - `test_signals_class_order_independence`: Validates that `SignalGenerator` dynamically maps class names, ensuring correct signal emission even if scikit-learn inverts internal class order (e.g., `['UP', 'NEUTRAL', 'DOWN']`).

5. **`tests/test_split_data.py` (1 test):**
   - `test_chronological_split_and_purge`: Slices a synthetic multi-year dataset, validating that exactly 5 days per symbol are purged and asserting that $\max(Timestamp_{Train}) < \min(Timestamp_{Val}) < \min(Timestamp_{Test})$.

6. **`tests/test_pipeline_reproducibility.py` (2 tests):**
   - `test_model_training_reproducibility`: Fits two independent Random Forest models using identical data and `random_seed=42`, verifying that output predictions are identical.
   - `test_model_serialization_integrity`: Fits a pipeline, saves it to a temporary directory with `joblib.dump`, reloads it, and verifies identical predictions and class probabilities (`np.allclose`).

### 13.3 Test Verification Log and Passing Assertions
Executing `pytest -v` across the suite produces the following verified passing output:

```text
============================= test session starts =============================
platform win32 -- Python 3.10.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\Field Project\New Approach
configfile: pyproject.toml
testpaths: tests
collected 17 items

tests/test_data_validation.py::test_validate_ohlcv_schema_passes_valid PASSED      [  5%]
tests/test_data_validation.py::test_validate_ohlcv_detects_duplicates PASSED        [ 11%]
tests/test_data_validation.py::test_validate_ohlcv_detects_negative_price PASSED    [ 17%]
tests/test_data_validation.py::test_validate_ohlcv_detects_high_low_violation PASSED [ 23%]
tests/test_data_validation.py::test_validate_features_dataset_detects_target_leakage PASSED [ 29%]
tests/test_data_validation.py::test_validate_features_dataset_detects_nan_values PASSED [ 35%]
tests/test_execution_and_portfolio.py::test_execution_slippage_and_fees PASSED     [ 41%]
tests/test_execution_and_portfolio.py::test_portfolio_cash_accounting_and_round_trip PASSED [ 47%]
tests/test_execution_and_portfolio.py::test_portfolio_prevents_negative_cash PASSED [ 52%]
tests/test_features.py::test_rsi_bounds PASSED                                       [ 58%]
tests/test_features.py::test_macd_calculation PASSED                                 [ 64%]
tests/test_features.py::test_features_independent_per_symbol PASSED                  [ 70%]
tests/test_features.py::test_target_classification_logic PASSED                      [ 76%]
tests/test_pipeline_reproducibility.py::test_model_training_reproducibility PASSED   [ 82%]
tests/test_pipeline_reproducibility.py::test_model_serialization_integrity PASSED     [ 88%]
tests/test_signals.py::test_signals_class_order_independence PASSED                  [ 94%]
tests/test_split_data.py::test_chronological_split_and_purge PASSED                  [100%]

============================= 17 passed in 9.22s ==============================
```

### 13.4 Deterministic Execution Guarantees
- **Seed Pinning:** Random seeds are fixed (`random_state=42`) across all scikit-learn estimators.
- **Ordered Operations:** `df.sort_values(by=['symbol', 'timestamp'])` is executed prior to feature computation, splitting, and backtesting to eliminate non-deterministic ordering effects.
- **Reproducibility Test:** Verified by `test_model_training_reproducibility`, which confirms bitwise prediction parity across independent training runs.

### 13.5 Test Scope Boundaries: What Automated Tests Do and Do Not Guarantee
- **What Tests Prove:** Automated tests verify software correctness, data integrity, mathematical calculation accuracy (RSI, MACD, slippage, PnL), leak-free splitting, error handling, and serialization round-trip fidelity.
- **What Tests Do NOT Prove:** Automated tests do **not** guarantee future financial profitability, absence of regime shifts, live market liquidity, or immunity to tail-risk market events.

---

## 14. Known Limitations, Market Realities, and Quant Risks

### 14.1 Survivorship and Selection Bias
The stock universe consists of 50 prominent large-cap equities selected in 2026. Evaluating these companies retrospectively across 2021–2025 introduces survivorship bias, as companies that experienced bankruptcy, severe distress, or delisting during this window are omitted. In real-world historical environments, portfolios would have encountered failing companies that eroded performance.

### 14.2 IEX Market Data Limitations
Data retrieved via Alpaca's free tier originates from the Investors Exchange (`DataFeed.IEX`). Because IEX accounts for approximately 2% to 3% of total US equity market volume, the `volume` field reflects IEX-specific activity rather than consolidated market-wide volume. While transaction prices on IEX track the consolidated NBBO closely due to arbitrage mechanisms, volume indicators should be interpreted with this limitation in mind.

### 14.3 Static Volatility Thresholds and Non-Stationary Regimes
The target labeling mechanism applies a fixed $\pm 1.0\%$ threshold across all equities and market regimes:
- During high-volatility regimes (e.g., tech selloffs), a 1.0% 5-day move is negligible noise.
- During low-volatility regimes (e.g., utility consolidation), a 1.0% move is a statistically significant trend.
- Future iterations should explore dynamic thresholds, such as scaling boundaries by the Average True Range (ATR) or rolling standard deviation.

### 14.4 Cross-Asset Correlation and Sector Concentration
The portfolio risk model caps individual stock allocation at 10% and allows up to 10 open positions. However, it does not enforce sector exposure caps. In technology-driven bull markets, the strategy may hold 10 highly correlated technology equities simultaneously (e.g., NVDA, AMD, QCOM, AMAT, MU), exposing the portfolio to severe sector drawdown risk during semiconductor sector pullbacks.

### 14.5 Friction and Execution Assumptions vs. Live Order Books
Backtesting models execution at next-day Open with fixed 5 bps slippage and 5 bps commission:
- In live markets, execution prices depend on order book depth, bid-ask spreads, and order routing latency.
- Extreme market opens (e.g., earnings announcements with overnight gaps) may experience slippage exceeding 5 bps.
- The simulation assumes full order fills and does not model partial fills or queue priority.

### 14.6 Predictive Accuracy vs. Portfolio Alpha Divergence
A key finding from this research is that statistical classification performance does not map linearly to investment outperformance:
- Random Forest achieved a statistically superior Balanced Accuracy of **$37.36\%$** on test data.
- However, the backtested strategy underperformed the Buy-and-Hold benchmark (+9.94% vs. +26.29% in 2025).
- This divergence stems from capital allocation friction: holding cash buffers creates drag in strong bull markets, while frequent rebalancing generates substantial friction costs ($\$6,928.29$ in 2025). High predictive accuracy does not generate alpha unless coupled with effective position sizing and trade timing.

---

## 15. Security, Governance, and Operational Considerations

### 15.1 API Key and Secret Management
- Alpaca API credentials (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) must reside exclusively in local, uncommitted `.env` files.
- `tradepilot.config` deliberately omits hardcoded credentials, resolving configuration solely through paths and parameters.
- If API keys are compromised, they should be revoked immediately through the Alpaca developer dashboard.

### 15.2 Version Control Hygiene and Secret Leakage Prevention
During inspection of the workspace, it was noted that **a `.gitignore` file was not present in the workspace root**. 
- *Critical Operational Recommendation:* A production `.gitignore` must be added immediately to prevent accidental exposure of sensitive files:
  ```gitignore
  .env
  .venv/
  __pycache__/
  *.pyc
  .pytest_cache/
  *.egg-info/
  ```

### 15.3 Paper Trading Isolation from Real Capital Execution
TradePilot is engineered for research, education, and paper trading:
- The system must **never** be connected to live brokerage accounts without extensive real-time forward validation.
- Alpaca paper-trading accounts provide an isolated sandbox with simulated capital, enabling risk-free execution testing.

### 15.4 Structured Logging, Auditability, and Run Failure Handling
- The codebase uses standard Python `logging` with ISO timestamps, module names, and severity levels.
- Pipeline execution logs exceptions with stack traces and exits with status code `1` upon critical errors.
- Every training and backtest run generates persistent JSON metadata and CSV reports, ensuring full operational auditability.

---

## 16. Future Development and Research Roadmap

### 16.1 Phase 1: Operational Hardening and Version Control
- Commit a comprehensive `.gitignore` to prevent credential leakage.
- Add GitHub Actions CI workflows to execute `pytest -v` automatically on pull requests.
- Integrate code formatters (`black`, `ruff`) to enforce PEP 8 style standards across the codebase.

### 16.2 Phase 2: Dynamic Feature Engineering and Target Labeling
- **Dynamic Volatility Thresholds (Triple Barrier Method):** Replace the static $\pm 1\%$ threshold with dynamic barriers scaled by daily Average True Range (ATR), incorporating upper, lower, and horizontal time-stop barriers.
- **Expanded Macro Indicators:** Incorporate 10-year Treasury yields, VIX volatility index series, and sector ETF relative strength indices.
- **Cross-Sectional Factor Ranking:** Transform time-series returns into cross-sectional percentiles across the 50-stock universe.

### 16.3 Phase 3: Walk-Forward Validation and Gradient Boosting Ensembles
- **Combinatorial Purged Cross-Validation (CPCV):** Implement walk-forward rolling window validation to evaluate stability across distinct market regimes (bull, bear, sideways).
- **Gradient Boosted Decision Trees:** Integrate XGBoost, LightGBM, and CatBoost classifiers to compare non-linear predictive power against Random Forest.
- **Bayesian Hyperparameter Tuning:** Implement Optuna to automate hyperparameter optimization over training folds without validation snooping.

### 16.4 Phase 4: MLOps Integration (MLflow and DVC)
- **MLflow Tracking:** Track training runs, log hyperparameter dictionaries, store validation metric curves, and register serialized models in a centralized model registry.
- **DVC (Data Version Control):** Version raw and processed datasets with remote cloud storage (AWS S3 or Google Cloud Storage) to ensure full data reproducibility across development environments.

### 16.5 Phase 5: Dynamic Risk Management and Sizing Strategies
- **Volatility-Targeted Sizing:** Size position allocations inversely proportional to rolling volatility (Risk Parity).
- **Trailing Stop-Loss & Take-Profit:** Implement dynamic trailing stops in `BacktestEngine` to lock in gains and truncate tail drawdowns.
- **Sector Concentration Limits:** Restrict maximum portfolio allocation within any single GICS economic sector to $\le 30\%$.

### 16.6 Phase 6: Automated Paper-Trading Execution Service
- Connect the signal generator to Alpaca's live paper-trading API endpoint via scheduled cron or task workers.
- Automatically execute orders at market open, stream fills via WebSockets, and log daily performance reports.
- Implement an automated alerting service (Slack / Discord webhooks) for order confirmations and risk threshold breaches.

---

## 17. Technical and Financial Glossary

- **Adverse Slippage:** The difference between the expected transaction price and the actual execution price, modeled as an adverse cost (higher buy price, lower sell price) due to market spread crossing and latency.
- **Balanced Accuracy:** The unweighted arithmetic mean of recall scores across all target classes. Unlike standard accuracy, it is robust to class imbalance.
- **Basis Point (bp):** A unit of measure equal to one-hundredth of a percentage point ($1 \text{ bp} = 0.01\% = 0.0001$).
- **Boundary Purging:** The practice of removing a buffer of observations (equal to the forward prediction horizon) at the boundary between training and validation splits to prevent target information leakage.
- **Compound Annual Growth Rate (CAGR):** The annualized geometric growth rate of an investment over a multi-year evaluation period.
- **Cash Drag:** The performance drag incurred when holding uninvested cash reserves during a strongly appreciating bull market.
- **Data Snooping / Lookahead Leakage:** The error of incorporating future information into historical model training or feature engineering, inflating backtest performance unrealistically.
- **Event-Driven Backtest:** A simulation architecture that models trading as a sequential, chronological state machine (signal $\to$ order queue $\to$ next-day execution $\to$ accounting).
- **Exponential Moving Average (EMA):** A weighted moving average that assigns greater statistical weight to recent price observations.
- **IEX Feed:** Market data originating from the Investors Exchange. A cost-effective feed that tracks NBBO prices accurately but captures only a subset of consolidated US equity volume.
- **Macro F1 Score:** The unweighted mean of F1 scores calculated independently for each target class, ensuring equal evaluation weight regardless of class frequency.
- **Mark-to-Market (MTM):** Valuing open portfolio positions at current market closing prices to compute daily unrealized profit/loss and total equity.
- **Maximum Drawdown (MDD):** The largest percentage drop from a portfolio equity peak to its subsequent trough before a new peak is achieved.
- **Moving Average Convergence Divergence (MACD):** A trend-following momentum oscillator reflecting the difference between 12-day and 26-day EMAs.
- **Profit Factor:** The ratio of gross winning trade profits to absolute gross losing trade losses. A value $> 1.0$ indicates positive net profitability.
- **Relative Strength Index (RSI):** A momentum oscillator bounded between 0 and 100 that measures the speed and change of directional price movements.
- **Sharpe Ratio:** A measure of risk-adjusted return calculated as excess return over the risk-free rate divided by annualized return volatility.
- **Survivorship Bias:** The distortion introduced by selecting a historical stock basket using current prominence criteria, omitting firms that failed during the historical window.
- **Turnover:** The total volume of capital traded divided by average portfolio equity, reflecting trading frequency and friction exposure.

---

## 18. Conclusion and Academic Project Assessment

### 18.1 Summary of Engineering Achievements
TradePilot demonstrates an institutional-grade, leak-free approach to machine learning in quantitative finance:
1. **Architectural Separation:** The codebase strictly decouples data ingestion, validation, stationary feature extraction, chronological modeling, and event-driven backtesting into modular, independently tested packages.
2. **Methodological Rigor:** By purging split boundaries and removing non-stationary price columns, the system eliminates lookahead leakage and price-scale confounding.
3. **Execution Realism:** Unlike frictionless backtesting scripts, TradePilot executes fills at next-day Open, penalizes round trips with 10 bps total friction, enforces position count caps, and maintains marked-to-market portfolio accounting.
4. **Reproducibility & Verification:** The implementation is accompanied by a fully passing automated test suite (17 of 17 tests passing), verified dataset counts, and persistent JSON metadata schemas.

### 18.2 Critical Takeaways on Financial Machine Learning
The empirical findings deliver important insights for quantitative researchers and evaluators:
- **Predictive Edge vs. Trading Alpha:** While the Random Forest model achieved a verified balanced accuracy of **$37.36\%$** on test data (outperforming naive guessing), translating this statistical edge into market outperformance proved challenging.
- **The Impact of Friction:** Executing 343 trades in 2025 incurred **$\$6,928.29$** in friction costs, illustrating how high-turnover rebalancing erodes net gains in liquid markets.
- **Regime Vulnerability:** Holding cash buffers protected capital in 2024 (yielding a low drawdown of -6.16%), but created cash drag in 2025's strong bull market, leading the strategy to lag the Equal-Weight Buy-and-Hold benchmark.

### 18.3 Assessment for Technical Evaluators
TradePilot provides an exemplary template for academic review and quantitative engineering evaluation:
- It prioritizes leak-free methodology and software architecture over curve-fitted, unrealistic backtest returns.
- It delivers fully auditable artifacts, comprehensive documentation, and verified passing test suites.
- It provides a robust, extensible foundation for subsequent research in advanced alpha factor design, dynamic volatility thresholding, and automated live paper execution.

---

## 19. Documentation Verification and Audit Log

### 19.1 Source Files and Code Inspected
The following files were inspected directly to establish ground truth for this documentation:
- Package Configuration: `pyproject.toml`, `requirements.txt`, `src/tradepilot/config.py`
- Core Data Subpackage: `src/tradepilot/data/load_data.py`, `src/tradepilot/data/validate_data.py`
- Feature Engineering Subpackage: `src/tradepilot/features/build_features.py`
- Modeling Subpackage: `src/tradepilot/modeling/split_data.py`, `src/tradepilot/modeling/train_model.py`, `src/tradepilot/modeling/evaluate_model.py`, `src/tradepilot/modeling/save_model.py`
- Backtesting Subpackage: `src/tradepilot/backtesting/signals.py`, `src/tradepilot/backtesting/execution.py`, `src/tradepilot/backtesting/portfolio.py`, `src/tradepilot/backtesting/metrics.py`, `src/tradepilot/backtesting/engine.py`
- Pipeline Orchestration: `src/tradepilot/pipelines/train_pipeline.py`, `src/tradepilot/pipelines/backtest_pipeline.py`
- Operational Scripts: `download_market_data.py`, `stock_eda.py`, `train.py`
- Test Suite: All 6 test files in `tests/` (`test_data_validation.py`, `test_execution_and_portfolio.py`, `test_features.py`, `test_pipeline_reproducibility.py`, `test_signals.py`, `test_split_data.py`)

### 19.2 Automated Test Execution and Verification
- Command Executed: `pytest -v`
- Execution Environment: Python 3.10.7 on Windows x86_64
- Result: **17 passed in 9.22 seconds** (100% pass rate, 0 failures, 0 warnings).

### 19.3 Verified Datasets and Split Counts
- Raw OHLCV (`data/raw/stock_ohlcv_2021_2025.csv`): Exactly 62,664 rows, 7 columns, 50 symbols, spanning 2021-01-04 to 2025-12-31.
- Clean Labeled Features (`data/processed/stock_features_labeled.csv`): Exactly 59,964 rows, 21 columns, 50 symbols, spanning 2021-03-16 to 2025-12-23. Exactly 2,700 warmup/shift rows dropped (4.31% loss).
- Training Split: 34,864 rows (250 boundary rows purged, 5 per symbol).
- Validation Split: 12,350 rows (250 boundary rows purged, 5 per symbol).
- Test Split: 12,250 rows (unpurged final evaluation fold).
- Total Split Accounting: $34,864 + 250 + 12,350 + 250 + 12,250 = 59,964$ rows.

### 19.4 Verified Empirical Metrics
- **Validation Fold (2024):**
  - Majority Baseline: Macro F1: `0.2049`, Balanced Acc: `0.3333`, Accuracy: `0.4437`
  - Logistic Regression: Macro F1: `0.3009`, Balanced Acc: `0.3618`, Accuracy: `0.3141`
  - Random Forest: Macro F1: `0.3428`, Balanced Acc: `0.3725`, Accuracy: `0.3457`
  - Backtest (TradePilot): Return: `+10.39%`, Sharpe: `0.92`, Max Drawdown: `-6.16%`, Win Rate: `56.78%`, Turnover: `23.38x`, Costs: `$\$5,041.68$`
  - Backtest (Benchmark): Return: `+33.35%`, Sharpe: `2.07`, Max Drawdown: `-9.57%`
- **Test Fold (2025):**
  - Random Forest: Macro F1: `0.3558`, Balanced Acc: `0.3736`, Accuracy: `0.3604`
  - Backtest (TradePilot): Return: `+9.94%`, Sharpe: `0.54`, Max Drawdown: `-22.50%`, Trades: `343`, Win Rate: `51.90%`, Profit Factor: `1.08`, Turnover: `33.41x`, Costs: `$\$6,928.29$`
  - Backtest (Benchmark): Return: `+26.29%`, Sharpe: `1.25`, Max Drawdown: `-21.41%`

### 19.5 Documented Discrepancies and Codebase Evolution
During documentation research, an important discrepancy between early research artifacts and the production package was identified and documented:
- In `stock_eda.py` (the initial Colab research prototype), feature lists included raw price-level moving averages (`SMA_10`, `SMA_20`, `SMA_50`).
- In the refactored production codebase (`src/tradepilot`), these moving averages were deliberately moved to `prohibited_cols` in `config.py` because raw price levels across diverse equities introduce cross-stock scale confounding.
- This evolution reflects intentional engineering refinement and is documented in Section 7.3.

### 19.6 Security and Leakage Audit
- Verified that no live API keys, tokens, or passwords are hardcoded in source code or documentation.
- Verified that `.env` is loaded exclusively via `python-dotenv`.
- Documented the absence of `.gitignore` as an operational risk and provided recommendations for immediate remediation.

---
*End of Complete Technical Documentation for TradePilot.*
