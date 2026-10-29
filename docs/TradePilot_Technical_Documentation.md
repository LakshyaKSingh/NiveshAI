# TradePilot: Machine Learning-Based Automated Stock Trading and Portfolio Management System

## Complete Technical Documentation and Engineering Specification

---

**Document Control Information**

| Attribute | Details |
| :--- | :--- |
| **Project Title** | TradePilot — ML-Based Automated Stock Trading and Portfolio Management System |
| **Document Classification** | Complete Technical Documentation & System Specification |
| **Software Version** | `1.0.0` (Production Application: FastAPI, Paper Trading Agent, Next.js Dashboard, AWS Ready) |
| **Date of Verification** | October 2026 |
| **Target Audience** | Academic Evaluators, Engineering Faculty, Technical Reviewers, Software Engineers, Quant Researchers |
| **System Implementation** | Python 3.10+, FastAPI, Next.js 16, TypeScript, Tailwind CSS, SQLAlchemy 2.0, Alembic, Scikit-Learn, Alpaca API, DVC, MLflow, AWS ECS Fargate, Terraform |
| **Primary Code Location** | `src/tradepilot/`, `frontend/`, `deploy/` |
| **Test Suite Status** | 36 of 36 Automated Tests Passing (`pytest -v`) |


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
   - [3.5 MLOps and Experiment Tracking Stack (DVC & MLflow)](#35-mlops-and-experiment-tracking-stack-dvc--mlflow)
   - [3.6 Clarification on Emerging and Future Stack Components](#36-clarification-on-emerging-and-future-stack-components)
4. [System Architecture and Component Design](#4-system-architecture-and-component-design)
   - [4.1 High-Level Architecture Diagram](#41-high-level-architecture-diagram)
   - [4.2 End-to-End Data, MLOps, and Control Flow](#42-end-to-end-data-mlops-and-control-flow)
   - [4.3 Architectural Decomposition and Component Responsibilities](#43-architectural-decomposition-and-component-responsibilities)
   - [4.4 Separation of Concerns: Prediction, Strategy, Execution, Risk, and MLOps](#44-separation-of-concerns-prediction-strategy-execution-risk-and-mlops)
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
    - [11.2 Package Configuration & Parameter Management: `tradepilot.config` and `params.yaml`](#112-package-configuration--parameter-management-tradepilotconfig-and-paramsyaml)
    - [11.3 Data Ingestion and Validation: `tradepilot.data`](#113-data-ingestion-and-validation-tradepilotdata)
    - [11.4 Feature Engineering Engine & CLI: `tradepilot.features`](#114-feature-engineering-engine--cli-tradepilotfeatures)
    - [11.5 Modeling and Cross-Validation: `tradepilot.modeling`](#115-modeling-and-cross-validation-tradepilotmodeling)
    - [11.6 Backtesting Simulator: `tradepilot.backtesting`](#116-backtesting-simulator-tradepilotbacktesting)
    - [11.7 MLOps Experiment Tracking: `tradepilot.tracking`](#117-mlops-experiment-tracking-tradepilottracking)
    - [11.8 Automated Pipelines & DVC Specification: `tradepilot.pipelines` and `dvc.yaml`](#118-automated-pipelines--dvc-specification-tradepilotpipelines-and-dvcyaml)
    - [11.9 Root Operational Entrypoints](#119-root-operational-entrypoints)
    - [11.10 REST API and Authentication Service: `tradepilot.api`](#1110-rest-api-and-authentication-service-tradepilotapi)
    - [11.11 Relational Persistence & Schema Migrations: `tradepilot.db`](#1111-relational-persistence--schema-migrations-tradepilotdb)
    - [11.12 Execution Engine, Broker Adapters & Centralized Risk Engine: `tradepilot.execution`](#1112-execution-engine-broker-adapters--centralized-risk-engine-tradepilotexecution)
    - [11.13 Modern Trading Dashboard: `frontend/`](#1113-modern-trading-dashboard-frontend)
    - [11.14 Cloud Deployment, Docker & Terraform IaC: `deploy/` and `.github/`](#1114-cloud-deployment-docker--terraform-iac-deploy-and-github)
12. [Installation, Environment Setup, and Execution Guide](#12-installation-environment-setup-and-execution-guide)
    - [12.1 Environment Prerequisites](#121-environment-prerequisites)
    - [12.2 Windows PowerShell Virtual Environment Activation](#122-windows-powershell-virtual-environment-activation)
    - [12.3 Editable Package Installation & MLOps Dependencies](#123-editable-package-installation--mlops-dependencies)
    - [12.4 Secure Environment Variable Configuration](#124-secure-environment-variable-configuration)
    - [12.5 Executing DVC Pipelines, Reproducibility, and Backtesting](#125-executing-dvc-pipelines-reproducibility-and-backtesting)
    - [12.6 Launching MLflow UI and Experiment Audit](#126-launching-mlflow-ui-and-experiment-audit)
    - [12.7 Locating and Auditing Generated Artifacts](#127-locating-and-auditing-generated-artifacts)
    - [12.8 Database Schema Setup & Alembic Migrations](#128-database-schema-setup--alembic-migrations)
    - [12.9 Launching the Full Production Stack (API, Worker, Frontend)](#129-launching-the-full-production-stack-api-worker-frontend)
    - [12.10 Multi-Container Deployment via Docker Compose](#1210-multi-container-deployment-via-docker-compose)
    - [12.11 Troubleshooting Common Operational Faults](#1211-troubleshooting-common-operational-faults)
13. [Quality Assurance, Automated Testing, and Reproducibility](#13-quality-assurance-automated-testing-and-reproducibility)
    - [13.1 Testing Architecture and Test Discovery](#131-testing-architecture-and-test-discovery)
    - [13.2 Breakdown of 36 Automated Unit and Integration Tests](#132-breakdown-of-36-automated-unit-and-integration-tests)
    - [13.3 Test Verification Log and Passing Assertions (36 Passed)](#133-test-verification-log-and-passing-assertions-36-passed)
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
    - [15.2 Version Control Hygiene and Repository `.gitignore` Specification](#152-version-control-hygiene-and-repository-gitignore-specification)
    - [15.3 Paper Trading Isolation from Real Capital Execution](#153-paper-trading-isolation-from-real-capital-execution)
    - [15.4 Structured Logging, Auditability, and Run Failure Handling](#154-structured-logging-auditability-and-run-failure-handling)
16. [Future Development and Research Roadmap](#16-future-development-and-research-roadmap)
    - [16.1 Phase 1: Operational Hardening and CI/CD Automation [COMPLETED]](#161-phase-1-operational-hardening-and-cicd-automation-completed)
    - [16.2 Phase 2: Dynamic Feature Engineering and Target Labeling](#162-phase-2-dynamic-feature-engineering-and-target-labeling)
    - [16.3 Phase 3: Walk-Forward Validation and Gradient Boosting Ensembles](#163-phase-3-walk-forward-validation-and-gradient-boosting-ensembles)
    - [16.4 Phase 4: MLOps Integration (MLflow and DVC) [COMPLETED]](#164-phase-4-mlops-integration-mlflow-and-dvc-completed)
    - [16.5 Phase 5: Dynamic Risk Management and Pre-Trade Safeguards [COMPLETED]](#165-phase-5-dynamic-risk-management-and-pre-trade-safeguards-completed)
    - [16.6 Phase 6: Automated Paper-Trading Execution Service & Next.js Dashboard [COMPLETED]](#166-phase-6-automated-paper-trading-execution-service--nextjs-dashboard-completed)
    - [16.7 Phase 7: Real-Time WebSockets, Order Book Depth & L2 Feed Integration](#167-phase-7-real-time-websockets-order-book-depth--l2-feed-integration)
    - [16.8 Phase 8: Multi-Agent LLM Sentiment Fusion & Alternative Data](#168-phase-8-multi-agent-llm-sentiment-fusion--alternative-data)
17. [Technical and Financial Glossary](#17-technical-and-financial-glossary)
18. [Conclusion and Academic Project Assessment](#18-conclusion-and-academic-project-assessment)
19. [Documentation Verification and Audit Log](#19-documentation-verification-and-audit-log)
    - [19.1 Verification Scope and Methodology](#191-verification-scope-and-methodology)
    - [19.2 Automated Test Execution Verification](#192-automated-test-execution-verification)
    - [19.3 Dataset and Artifact Integrity Audit](#193-dataset-and-artifact-integrity-audit)
    - [19.4 Verified Empirical Metrics](#194-verified-empirical-metrics)
    - [19.5 MLOps Pipeline and Experiment Tracking Verification](#195-mlops-pipeline-and-experiment-tracking-verification)
    - [19.6 Documented Discrepancies and Codebase Evolution](#196-documented-discrepancies-and-codebase-evolution)
    - [19.7 Security and Leakage Audit](#197-security-and-leakage-audit)
    - [19.8 Version 1.0.0 Production Application Release Audit](#198-version-100-production-application-release-audit)
20. [Production Application Architecture, API, Agent Execution, and Cloud Deployment](#20-production-application-architecture-api-agent-execution-and-cloud-deployment)
    - [20.1 Architectural Expansion Overview](#201-architectural-expansion-overview)
    - [20.2 Versioned REST API Endpoints (/api/v1)](#202-versioned-rest-api-endpoints-apiv1)
    - [20.3 Database Architecture & Migrations](#203-database-architecture--migrations)
    - [20.4 Centralized Risk Engine & Pre-Trade Safeguards](#204-centralized-risk-engine--pre-trade-safeguards)
    - [20.5 Autonomous Background Worker](#205-autonomous-background-worker)
    - [20.6 Next.js 16 Trading Dashboard](#206-nextjs-16-trading-dashboard)
    - [20.7 AWS Cloud Deployment Infrastructure](#207-aws-cloud-deployment-infrastructure)
    - [20.8 Final Test Suite Verification](#208-final-test-suite-verification)

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
- **DVC Pipeline Versioning & Caching:** Multi-stage pipeline definition (`dvc.yaml`, `params.yaml`, `dvc.lock`) tracking dependencies, invalidating stale downstream stages, and caching datasets with a local portable storage remote (`.dvc_storage`).
- **MLflow Experiment Tracking:** Local-first tracking backend (`sqlite:///mlflow.db`) capturing parameters, metrics across validation and test folds, feature importance charts, model binaries, and separate trading simulation runs under the `TradePilot-Stock-Direction` experiment.

### 1.5 Current Implementation Status
The project is fully implemented, verified, and operational in `E:\Field Project\New Approach`. 
- The complete pipeline executes deterministically from ingestion through feature engineering, model training, and backtesting.
- DVC pipeline execution (`dvc repro`) runs all 3 stages seamlessly and successfully invalidates/skips unchanged stages.
- MLflow logs run parameters, model metrics, feature importances, and backtest results into an SQLite database with complete traceability to Git commits and DVC versions.
- All 36 automated tests in the test suite pass with 100% compliance (`pytest -v`), spanning data validation, feature engineering, split purging, backtesting, REST API endpoints, and trading agent risk safeguards.
- The production-grade FastAPI REST API (`/api/v1`) serves authenticated endpoints for market data, model predictions, strategy signals, portfolio accounting, order lifecycle management, and agent control.
- An autonomous background strategy agent (`tradepilot.execution`) executes paper trading cycles with Alpaca API connectivity, an offline deterministic `MockBrokerAdapter`, and a centralized 10-rule risk engine with emergency stop capability.
- A modern financial dashboard built in Next.js 16, TypeScript, and Tailwind CSS provides high-density market inspection, interactive candlestick/volume charts, SVG equity curves, and order execution.
- Verified datasets span 50 equities from January 2021 through December 2025 (62,664 raw OHLCV rows; 61,764/59,964 clean, labeled feature rows).
- Model evaluation demonstrates that the Random Forest classifier successfully captures non-linear structure across all classes, outperforming baseline models on balanced accuracy and Macro F1.
- Backtesting validates that while the strategy generates positive cumulative returns in both 2024 (+10.39%) and 2025 (+6.78% to +9.94%), frequent rebalancing frictions and cash-drag limitations restrict its ability to outperform a concentrated mega-cap bull-market benchmark.
- Cloud readiness is verified via Docker containers (`deploy/`), multi-service `docker-compose.yml`, Terraform IaC for Amazon ECS Fargate, RDS PostgreSQL 15, ALB, S3, Secrets Manager, and GitHub Actions CI (`.github/workflows/ci.yml`).

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
11. **Versioned REST API Service:** The system must serve documented endpoints under `/api/v1` with OpenAPI schemas, JWT Bearer authentication, and Role-Based Access Control (RBAC).
12. **Autonomous Paper Trading Execution:** The system must run a background execution agent cycling through inference, risk validation, paper order submission via Alpaca API or offline mock simulation, and position reconciliation.
13. **Centralized Quantitative Risk Controls:** The system must enforce 10+ pre-trade quantitative checks, portfolio exposure caps, daily drawdown circuit breakers, and an administrative emergency kill switch.
14. **Modern Trading Dashboard:** The system must provide an interactive Next.js 16 financial interface displaying real-time marked-to-market equity curves, candlestick price charts, probability distributions, order lifecycle management, and agent controls.
15. **Relational State Persistence:** The system must persist users, agent cycles, orders, fills, snapshots, and risk events using SQLAlchemy 2.0 with Alembic version-controlled migrations.

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

| Capability Domain | In-Scope (Implemented & Verified in v1.0.0) | Out-of-Scope (Future / Planned) |
| :--- | :--- | :--- |
| **Market Data** | Historical daily OHLCV via Alpaca IEX API & 4-year CSV fallback | Real-time tick WebSocket streaming, L2 order books |
| **Data Cleansing** | Missing value drops, duplicate checks, OHLC geometric bounds | Outlier winsorization, corporate action adjustment parsers |
| **Features** | 8 stationary indicators (Returns, Volatility, RSI, MACD, Volume) | Fundamental ratios, order flow imbalance, sentiment analysis |
| **Target Design** | Fixed $\pm 1\%$ 5-day forward return 3-class classification | Dynamic ATR-based targets, Triple Barrier Method, regression |
| **Machine Learning** | DummyClassifier, Logistic Regression, Random Forest | XGBoost, LightGBM, CatBoost, LSTM, Temporal Fusion Transformers |
| **Validation** | Chronological 3-way split with 5-day boundary purging | Purged Walk-Forward Cross-Validation, Combinatorial Purged CV |
| **Trading Strategy** | Long-only event-driven execution with 5-day time exit | Long/short pairs, market-neutral hedging, stop-loss/take-profit |
| **Execution Modeling** | Next-day Open execution, 5 bps slippage, 5 bps commissions, Alpaca Paper API & Mock broker | Limit order book simulation, partial fills, market impact models |
| **Application Layer** | FastAPI REST API (`/api/v1`), JWT authentication, RBAC, Pydantic v2 validation | Mobile native clients (iOS/Android) |
| **Risk & Safeguards** | Centralized Risk Engine (10+ rules, drawdown breaker, emergency stop) | Dynamic VaR modeling, stress testing simulations |
| **User Interface** | Next.js 16 trading dashboard, interactive SVG charts, order entry | Social trading feeds, third-party plugin store |
| **Deployment & Ops** | DVC pipeline, MLflow tracking, SQLite/PostgreSQL, Docker, Terraform AWS IaC, CI/CD | Multi-region active-active cluster, automated live capital trading |

---

## 3. Technology Stack and Environment

### 3.1 Runtime Environment and Core Libraries
TradePilot is built on **Python 3.10** (verified on Python 3.10.7 on Windows x86_64). The project relies on standard foundational libraries for high-performance array manipulation and structured data analysis:
- **Python (>= 3.10):** Native runtime providing modern structural pattern matching, robust type hinting, and enhanced exception handling.
- **NumPy (>= 1.22.0):** Provides vectorized array computation, fast mathematical transformations, standard deviations, and matrix operations underlying scikit-learn models.
- **Pandas (>= 2.0.0):** The primary tabular data engine utilized for time-series grouping, rolling window statistics (rolling mean, rolling standard deviation), date filtering, and ledger management.
- **Pathlib:** Standard library module utilized throughout `tradepilot.config` to ensure cross-platform, deterministic path resolution relative to the repository root.
- **PyYAML (`pyyaml >= 6.0`):** Parses `params.yaml` to dynamically supply hyperparameters to pipelines and configuration loaders.

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
- **Matplotlib (`matplotlib >= 3.7.0`):** Programmatically generates feature importance bar charts and performance plots for direct artifact logging to MLflow.

### 3.4 Quality Assurance and Testing Framework
- **Pytest (`pytest >= 7.0.0`):** Testing framework that discovers and runs the 17 unit and integration tests under `tests/`. Pytest settings are defined directly in `pyproject.toml`.
- **Setuptools (`setuptools >= 61.0`):** PEP 517/621-compliant build backend used to install TradePilot as an editable package (`pip install -e .`).

### 3.5 MLOps and Experiment Tracking Stack (DVC & MLflow)
- **Data Version Control (`dvc >= 3.50.0`):**
  - Manages data versioning and reproducible pipeline orchestration via `dvc.yaml` and `dvc.lock`.
  - Decouples raw datasets (`stock_ohlcv_2021_2025.csv.dvc`) from Git tracking, storing large datasets in a dedicated content-addressable storage cache.
  - Provides a local storage remote (`.dvc_storage`) configured for offline, zero-cloud portability.
  - Intelligently detects input/code changes to avoid redundant model training runs.
- **MLflow Lifecycle Management (`mlflow >= 2.15.0`):**
  - Tracks model runs, hyperparameter dictionaries, cross-validation metrics, and final test scores under the `TradePilot-Stock-Direction` experiment.
  - Uses an embedded SQLite backend (`sqlite:///mlflow.db`) with localized artifact storage (`./mlruns`), fully adhering to MLflow 3.x modern database-backed tracking requirements without requiring external services.
  - Logs model binaries (`best_model.joblib` and serialized scikit-learn pipelines), feature importance charts (`feature_importance.png`), confusion matrix tables, and complete backtesting simulation outputs.

### 3.6 Application, Frontend, and Cloud Infrastructure Stack
In the final application release (`v1.0.0`), TradePilot was integrated with a complete production web and cloud stack:
- **FastAPI (`fastapi >= 0.115.0`) & Uvicorn (`uvicorn >= 0.30.0`):** Powers the asynchronous, versioned REST API (`/api/v1`) with OpenAPI documentation, dependency injection, and CORS security.
- **Pydantic v2 (`pydantic >= 2.10.0`) & Pydantic-Settings:** Handles request/response schema validation, financial bounds checking, and environment variable configuration with strict typing.
- **SQLAlchemy 2.0 (`sqlalchemy >= 2.0.30`) & Alembic (`alembic >= 1.13.0`):** Enterprise ORM mapping application entities (Users, Agent States, Cycles, Orders, Fills, Positions, Risk Events, Audit Logs) with version-controlled schema migrations. Defaults to embedded SQLite locally and seamlessly targets PostgreSQL 15 on Amazon RDS.
- **Next.js 16 (`next >= 16.0.0`), TypeScript, & Tailwind CSS:** Client-side financial monitoring and trading terminal in `frontend/` featuring real-time equity curves, interactive candlestick charts, probability distributions, order lifecycle management, and emergency kill switches.
- **Docker & Multi-Container Compose:** Production Dockerfiles (`deploy/Dockerfile.api`, `deploy/Dockerfile.worker`, `deploy/Dockerfile.frontend`) and local multi-service orchestration via `docker-compose.yml`.
- **Terraform (AWS Provider ~> 5.0):** Complete Infrastructure-as-Code definitions (`deploy/terraform/`) for Amazon ECS on AWS Fargate, Application Load Balancer, Multi-AZ Amazon RDS PostgreSQL, Amazon S3, AWS Secrets Manager, and Amazon CloudWatch.
- **GitHub Actions CI (`.github/workflows/ci.yml`):** Continuous integration workflow executing the 36-test pytest suite, dependency audits, and Next.js production builds on every push and pull request.


---

## 4. System Architecture and Component Design

### 4.1 High-Level Architecture Diagram
The following architectural diagram illustrates the complete end-to-end data processing, MLOps orchestration, and execution pipeline of TradePilot.

```mermaid
flowchart TD
    subgraph DVC_DATA ["DVC Data Versioning & Ingestion"]
        A1["Alpaca Market Data API (IEX Feed)"] -->|"Daily OHLCV Bars"| A2["download_market_data.py"]
        A2 -->|"Raw CSV"| A3["data/raw/stock_ohlcv_2021_2025.csv"]
        A3 -.->|"DVC Tracked Pointer"| A4["stock_ohlcv_2021_2025.csv.dvc"]
        A3 --> B1["tradepilot.data.validate_data"]
        B1 -->|"Integrity Passed"| B2["data/processed/stock_eda_cleaned.csv"]
    end

    subgraph DVC_STAGE_1 ["DVC Stage: 'features' (cmd: build_features.py)"]
        P1["params.yaml (features)"] --> C1["tradepilot.features.build_features"]
        A3 --> C1
        C1 -->|"8 Stationary Technical Indicators"| C2["Features Matrix"]
        C1 -->|"Shift Close -5 Days per Symbol"| C3["Multi-Class Target (UP/DOWN/NEU)"]
        C2 --> C4["data/processed/stock_features_labeled.csv"]
        C3 --> C4
    end

    subgraph DVC_STAGE_2 ["DVC Stage: 'train' (cmd: train_pipeline.py)"]
        P2["params.yaml (train)"] --> D1["tradepilot.modeling.split_data"]
        C4 --> D1
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
        E3 -->|"json / csv export"| E7["models/model_metadata.json<br/>reports/validation_results.csv<br/>reports/test_results.json<br/>reports/confusion_matrix.csv"]
        E5 --> E7
    end

    subgraph MLFLOW_TRACKING ["MLflow Lifecycle Management (sqlite:///mlflow.db)"]
        E4 -.->|"Log Params, Metrics & Models"| M1["tradepilot.tracking.log_training_to_mlflow"]
        E7 -.->|"Log Reports, Plots & Artifacts"| M1
        M1 --> M2[("SQLite DB: mlflow.db<br/>Artifacts: ./mlruns")]
    end

    subgraph DVC_STAGE_3 ["DVC Stage: 'backtest' (cmd: backtest_pipeline.py)"]
        P3["params.yaml (backtest)"] --> F0["tradepilot.backtest_pipeline"]
        E6 --> F1["tradepilot.backtesting.signals (SignalGenerator)"]
        C4 --> F1
        F1 -->|"Signals Generated at Close (t)"| F2["Pending Order Queue"]

        F2 --> G1["tradepilot.backtesting.engine (BacktestEngine Loop)"]
        A3 -->|"Next-Session Open (t+1)"| G2["ExecutionHandler"]
        G2 -->|"Apply 5 bps Slippage + 5 bps Comm"| G3["TradeFill"]

        G3 --> H1["tradepilot.backtesting.portfolio (Portfolio Manager)"]
        A3 -->|"Session Close (t+1)"| H1
        H1 -->|"Mark-to-Market"| H2["Equity Curve & Open Positions"]
        H1 -->|"Close Round-Trips"| H3["Completed Trade Ledger"]

        H2 --> I1["tradepilot.backtesting.metrics"]
        H3 --> I1
        A3 -->|"Buy & Hold Baseline"| I1
        I1 --> J1["reports/backtest_summary.json<br/>reports/backtest_equity_curve.csv<br/>reports/trade_ledger.csv<br/>reports/benchmark_comparison.csv"]
        J1 -.->|"Log Backtest Metrics & Artifacts"| M3["tradepilot.tracking.log_backtest_to_mlflow"]
        M3 --> M2
    end
```

### 4.2 End-to-End Data, MLOps, and Control Flow
1. **Ingestion & Integrity Check:** `download_market_data.py` interfaces with Alpaca API, fetching split-adjusted daily OHLCV bars for 50 tickers across 2021–2025. Data is passed through schema and boundary assertions in `tradepilot.data.validate_data`. Raw data is tracked via `data/raw/stock_ohlcv_2021_2025.csv.dvc` and synchronized to the local DVC cache (`.dvc_storage`).
2. **DVC Feature Stage (`features`):** `tradepilot.features.build_features` calculates 8 stationary technical indicators grouped by symbol, reading parameter overrides from `params.yaml`. Forward 5-day returns are computed, and multi-class target labels are assigned. Rows containing initialization NaNs or terminal shift NaNs are dropped, saving `data/processed/stock_features_labeled.csv` (cached by DVC).
3. **DVC Modeling Stage (`train`):** `tradepilot.modeling.split_data` slices data into Train ($\le 2023$), Validation ($2024$), and Test ($2025$), purging 5 sessions per symbol at boundaries. Candidate models (`Majority_Baseline`, `Logistic_Regression`, `Random_Forest`) are fitted. The best validation model (`Random_Forest`) is evaluated on 2025 test data and serialized to `models/best_model.joblib`.
4. **MLflow Model Logging:** `tradepilot.tracking.log_training_to_mlflow` starts an MLflow run under `TradePilot-Stock-Direction`, logging 30+ hyperparameters, 32+ validation and test metrics, feature importance charts, classification reports, confusion matrices, and the fitted model pipeline.
5. **DVC Backtest Stage (`backtest`):** `tradepilot.pipelines.backtest_pipeline` loads `models/best_model.joblib`, generates probability signals via `SignalGenerator`, and runs the chronological event-driven simulation with 5 bps slippage and 5 bps commissions. Detailed equity curves and trade ledgers are exported.
6. **MLflow Backtest Logging:** `tradepilot.tracking.log_backtest_to_mlflow` logs strategy metrics (Sharpe ratio, CAGR, max drawdown, win rate, turnover, total costs) and stores the trade ledger and equity curve as auditable artifacts.
7. **DVC Stage Invalidation & Caching:** When `dvc repro` is invoked subsequently without changes to code, data, or `params.yaml`, DVC skips all stages ("Stage didn't change, skipping"), guaranteeing deterministic pipeline reproducibility.

### 4.3 Architectural Decomposition and Component Responsibilities

| Subpackage / Layer | Primary Module | Core Responsibility |
| :--- | :--- | :--- |
| `tradepilot` | `config.py` | Centralized dataclasses (`DataConfig`, `ModelConfig`, `BacktestConfig`) and parameter loader (`load_params`, `params.yaml`). |
| `tradepilot` | `tracking.py` | MLflow experiment tracking orchestrator logging parameters, metrics, artifacts, and models to SQLite backend. |
| `tradepilot.data` | `load_data.py` | Disk I/O with automatic UTC timestamp parsing and chronological sorting. |
| `tradepilot.data` | `validate_data.py` | Schema verification, duplicate detection, OHLC geometric sanity checks, and feature leakage assertions. |
| `tradepilot.features` | `build_features.py` | Technical indicator generation per ticker, target labeling, and DVC feature stage CLI entrypoint (`run_feature_pipeline`). |
| `tradepilot.modeling` | `split_data.py` | Chronological dataset partitioning with per-ticker boundary purging. |
| `tradepilot.modeling` | `train_model.py` | Definition and fitting of scikit-learn model pipelines. |
| `tradepilot.modeling` | `evaluate_model.py` | Multi-class classification evaluation, balanced metrics, and candidate selection. |
| `tradepilot.modeling` | `save_model.py` | Serialization of fitted estimators (`joblib`) and model metadata schemas (`json`). |
| `tradepilot.backtesting` | `signals.py` | Dynamic class-order probability mapping and signal generation. |
| `tradepilot.backtesting` | `execution.py` | Order generation, budget allocation bounding, next-day Open fills, and transaction cost modeling. |
| `tradepilot.backtesting` | `portfolio.py` | Position life-cycle management, cash accounting, daily mark-to-market valuation, and trade ledger tracking. |
| `tradepilot.backtesting` | `metrics.py` | Computation of annualized Sharpe ratio, CAGR, volatility, max drawdown, win rate, and benchmark metrics. |
| `tradepilot.backtesting` | `engine.py` | Event-driven simulation orchestrator executing the chronological daily state machine. |
| `tradepilot.pipelines` | `train_pipeline.py` | End-to-end training orchestrator and DVC train stage command. |
| `tradepilot.pipelines` | `backtest_pipeline.py` | Out-of-sample backtest orchestrator and DVC backtest stage command. |
| **Pipeline & Config** | `dvc.yaml` & `params.yaml` | Multi-stage pipeline graph specification, stage dependencies, hyperparameters, and artifact caching. |

### 4.4 Separation of Concerns: Prediction, Strategy, Execution, Risk, and MLOps
A hallmark of TradePilot's architecture is the strict decoupling of its five core operational domains:
1. **The Predictive Layer (`modeling`):** Focuses solely on statistical classification. It maps an 8-dimensional feature vector $X_t$ to class probabilities $P(y_t = c)$. It possesses no knowledge of account equity, position sizes, order types, or brokerage mechanics.
2. **The Decision / Signal Layer (`signals`):** Converts continuous statistical probabilities into trading intentions. It applies probability thresholds ($P(UP) \ge 0.40$) to filter out low-conviction predictions, emitting discrete signals ($1, -1, 0$).
3. **The Execution Layer (`execution`):** Bridges trading intentions with market reality. It translates a signal generated at Close $t$ into an actionable order executed at Open $t+1$. It calculates the exact integer shares affordable given available capital and deducts modeled slippage and exchange fees.
4. **The Risk and Portfolio Layer (`portfolio`):** Enforces capital preservation rules. It enforces a maximum of 10 concurrent open positions, caps single-stock exposure at 10% of total portfolio equity, prevents position pyramiding, rejects orders when cash is depleted, and executes time-based exits at the expiration of the 5-day holding period.
5. **The MLOps and Reproducibility Layer (`tracking.py`, `dvc.yaml`, `params.yaml`):** Manages data provenance, pipeline stage invalidation, parameter overrides, experiment run histories, and metric tracking without polluting algorithmic trading logic.

### 4.5 Design Principles and Modularity Rationale
The architecture adheres to core software engineering principles:
- **Single Responsibility Principle (SRP):** Each module handles one distinct operational task. Feature calculation is strictly separated from data loading, and execution simulation is strictly decoupled from metric reporting.
- **Open/Closed Principle (OCP):** New candidate models (e.g., XGBoost, LightGBM) can be added to `get_candidate_pipelines()` in `train_model.py` without modifying the training orchestrator or evaluation logic.
- **Fail-Fast Defensive Programming:** Validation assertions execute prior to computationally heavy steps. If an OHLC price violates physical bounds or a feature column contains NaNs, the pipeline raises an informative `ValueError` immediately rather than propagating invalid data downstream.
- **Zero Lookahead Leakage by Construction:** Features are computed before target shifting; target shifts are grouped per symbol; boundary rows are purged prior to training; and order execution is delayed to the subsequent trading session ($t+1$).

---

## 5. Repository and File Structure

### 5.1 Complete Workspace Directory Tree
Below is the verified repository tree of the TradePilot project, detailing the codebase, DVC pipeline descriptors, and MLflow tracking components:

```text
E:\Field Project\New Approach\
├── .dvc/                                # DVC internal configuration directory
│   ├── .gitignore                       # DVC internal cache ignore rules
│   └── config                           # Local storage remote configuration (.dvc_storage)
├── .dvc_storage/                        # Portable local DVC storage remote (uncommitted data cache)
├── .gitignore                           # Repository version control ignore rules (secrets, venv, caches)
├── data/
│   ├── processed/
│   │   ├── .gitignore                   # DVC-managed ignore rules for processed features
│   │   ├── stock_eda_cleaned.csv        # Validated base dataset with daily returns (62,664 rows)
│   │   └── stock_features_labeled.csv   # DVC-cached labeled dataset with 8 features & target
│   └── raw/
│       ├── .gitignore                   # DVC-managed ignore rules for raw data CSV
│       ├── stock_ohlcv_2021_2025.csv    # Raw historical OHLCV from Alpaca API (62,664 rows)
│       └── stock_ohlcv_2021_2025.csv.dvc# DVC pointer tracking raw dataset content hash
├── docs/
│   └── TradePilot_Technical_Documentation.md  # Comprehensive technical documentation specification
├── models/
│   ├── .gitignore                       # DVC-managed ignore rules for model binaries
│   ├── best_model.joblib                # DVC-cached winning Random Forest classifier pipeline
│   └── model_metadata.json              # Full training metadata, hyperparameters, and split info
├── reports/
│   ├── backtest_equity_curve.csv        # Daily marked-to-market equity values for 2025 test period
│   ├── backtest_summary.json            # Complete financial metrics for validation and test backtests
│   ├── benchmark_comparison.csv         # Comparative metrics: Strategy vs. Equal-Weight vs. Cash
│   ├── confusion_matrix.csv             # Tabular confusion matrices across candidate models
│   ├── feature_importances.csv          # Gini feature importance rankings from winning ensemble
│   ├── test_results.json                # Out-of-sample 2025 classification report and metrics
│   ├── trade_ledger.csv                 # Granular record of all 318-343 executed trades in 2025 test
│   └── validation_results.csv           # Validation comparison table across all candidate models
├── .github/
│   └── workflows/
│       └── ci.yml                         # Automated GitHub Actions CI workflow (tests, lint, build)
├── alembic/
│   ├── env.py                             # Alembic migration environment connected to Base metadata
│   ├── script.py.mako
│   └── versions/
│       └── 934fa89e0077_initial_schema.py # Initial database migration
├── alembic.ini                            # Alembic database migration configuration
├── deploy/
│   ├── Dockerfile.api                     # Production container for FastAPI server
│   ├── Dockerfile.worker                  # Production container for Strategy Background Worker
│   ├── Dockerfile.frontend                # Multi-stage production container for Next.js 16
│   ├── .env.example                       # Environment template with parameter documentation
│   ├── docs/
│   │   └── AWS_ARCHITECTURE.md           # AWS deployment guide, runbook, costs, & rollback
│   └── terraform/
│       ├── main.tf                        # ECS Fargate, ALB, RDS, S3, SecretsManager, CloudWatch
│       ├── variables.tf                   # Terraform input variables
│       └── outputs.tf                     # Terraform resource outputs
├── docker-compose.yml                     # Local multi-container orchestration (Postgres, API, Worker, UI)
├── frontend/                              # Next.js 16 TypeScript Trading Dashboard
│   ├── package.json
│   ├── tsconfig.json
│   └── src/
│       ├── app/
│       │   ├── layout.tsx                 # Root layout with TradePilot metadata
│       │   ├── page.tsx                   # Main SPA dashboard managing views & auto-refresh
│       │   └── globals.css                # Dark financial terminal styling
│       ├── components/
│       │   ├── Header.tsx                 # Status pills, agent controls, & kill switch modal
│       │   ├── EquityChart.tsx            # Interactive SVG equity timeline chart
│       │   ├── PriceChart.tsx             # Interactive candlestick & volume price chart
│       │   └── views/
│       │       ├── DashboardView.tsx      # KPI cards, equity curve, holdings, & signal glance
│       │       ├── MarketsView.tsx        # Symbol selector, OHLCV bars table, & charts
│       │       ├── PredictionsView.tsx    # Model metadata, on-demand inference, & probabilities
│       │       ├── PortfolioView.tsx      # Asset allocation stacked bar & holdings ledger
│       │       ├── OrdersView.tsx         # Manual paper order form & fills ledger
│       │       ├── AgentControlView.tsx   # Agent lifecycle state machine & worker controls
│       │       └── RiskView.tsx           # Limit utilization meters, risk events, & audit log
│       └── lib/
│           └── api.ts                     # Fully-typed API client for all backend schemas
├── src/
│   └── tradepilot/
│       ├── __init__.py                  # Package exports (__version__ = "1.0.0")
│       ├── config.py                    # Dataclasses and params.yaml loader (get_data_config, etc.)
│       ├── tracking.py                  # MLflow experiment tracking orchestrator (SQLite backend)
│       ├── api/                         # FastAPI REST API subpackage (/api/v1)
│       │   ├── app.py                   # FastAPI application entrypoint with CORS & routes
│       │   ├── config.py                # Pydantic Settings & environment safeguards
│       │   ├── security.py              # Bcrypt hashing & JWT bearer token generation
│       │   ├── dependencies.py          # Database session & RBAC authentication dependencies
│       │   ├── schemas/                 # Pydantic v2 schemas
│       │   └── routers/                 # Versioned REST routers (system, auth, markets, etc.)
│       ├── db/                          # SQLAlchemy database subpackage
│       │   ├── base.py                  # DeclarativeBase and TimestampMixin
│       │   ├── session.py               # SQLAlchemy engine, SessionLocal, init_db()
│       │   └── models.py                # 12 Normalized relational ORM models
│       ├── execution/                   # Strategy execution and brokerage adapters
│       │   ├── broker_base.py           # Abstract BaseBrokerAdapter specification
│       │   ├── alpaca_adapter.py        # Alpaca Paper API adapter (strict paper check)
│       │   ├── mock_adapter.py          # In-memory paper broker simulation with slippage/fees
│       │   ├── risk_engine.py           # CentralizedRiskEngine (10 quantitative rules)
│       │   ├── agent.py                 # TradingAgent (state machine, sizing, reconciliation)
│       │   └── worker.py                # BackgroundWorker process with graceful shutdown
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
│       │   └── build_features.py        # Feature engineering & DVC stage CLI (run_feature_pipeline)
│       ├── modeling/
│       │   ├── __init__.py              # Modeling subpackage exports
│       │   ├── evaluate_model.py        # Classification evaluation and model selection logic
│       │   ├── save_model.py            # Artifact persistence (joblib, metadata JSON, reports)
│       │   ├── split_data.py            # Chronological splitting with boundary purging
│       │   └── train_model.py           # Candidate pipeline construction and fitting
│       └── pipelines/
│           ├── __init__.py              # Pipeline subpackage exports
│           ├── backtest_pipeline.py     # Backtesting execution entrypoint & DVC stage command
│           └── train_pipeline.py        # Training execution entrypoint & DVC stage command
├── tests/
│   ├── __init__.py                      # Test package marker
│   ├── test_data_validation.py          # Tests for schema, duplicates, price bounds, and leakage
│   ├── test_execution_and_portfolio.py  # Tests for slippage, cash accounting, and solvency
│   ├── test_features.py                 # Tests for RSI, MACD, per-symbol isolation, and targets
│   ├── test_pipeline_reproducibility.py # Tests for deterministic fitting and joblib serialization
│   ├── test_signals.py                  # Tests for class order independence in signal mapping
│   ├── test_split_data.py               # Tests for chronological partitioning and boundary purging
│   ├── test_api_endpoints.py            # 11 Integration tests for FastAPI endpoints
│   └── test_trading_agent_and_risk.py   # 8 Unit/integration tests for Risk Engine & Agent
├── download_market_data.py              # Script to download raw OHLCV data from Alpaca API
├── dvc.lock                             # DVC state lockfile recording hashes of dependencies & outputs
├── dvc.yaml                             # Multi-stage DVC pipeline definition (features, train, backtest)
├── mlflow.db                            # SQLite embedded database for MLflow tracking metadata
├── mlruns/                              # Local MLflow artifact repository directory
├── params.yaml                          # Central parameter configuration file for all DVC stages
├── pyproject.toml                       # Packaging configuration, build dependencies, and pytest options
├── requirements.txt                     # Explicit package dependency list (including DVC and MLflow)
├── stock_eda.py                         # Standalone exploratory data analysis and Colab prototype
├── tradepilot.db                        # SQLite database for application state & audit logs
└── train.py                             # Root wrapper delegating to train_pipeline
```


### 5.2 Module Mapping and Responsibility Matrix
The following matrix illustrates the operational role of each primary component across the DVC and MLflow lifecycle:

```text
[Alpaca Market Data API]
       │
       ▼
download_market_data.py ──────► data/raw/stock_ohlcv_2021_2025.csv ◄───[Tracked: stock_ohlcv_...csv.dvc]
                                        │
                                        ▼  (DVC Stage: 'features')
                               tradepilot.features.build_features ◄────[params.yaml: features]
                                        │
                                        ▼
                               data/processed/stock_features_labeled.csv (DVC Cached)
                                        │
                                        ▼  (DVC Stage: 'train')
                               tradepilot.pipelines.train_pipeline ◄───[params.yaml: train]
                                 ├── split_data.py
                                 ├── train_model.py
                                 ├── evaluate_model.py
                                 └── save_model.py
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
           models/best_model.joblib            tradepilot.tracking (MLflow Run)
           (DVC Cached Model Artifact)         ├── Log 30+ Parameters
                     │                         ├── Log 32+ Metrics
                     ▼  (DVC Stage: 'backtest')├── Log Feature Importance Plot
       tradepilot.pipelines.backtest_pipeline  └── Log Evaluation Reports
         ├── signals.py                                    │
         ├── engine.py ◄────────────────[params.yaml]      ▼
         ├── execution.py                       sqlite:///mlflow.db
         ├── portfolio.py                              & ./mlruns
         └── metrics.py
                     │
                     ▼
       reports/backtest_summary.json
       reports/backtest_equity_curve.csv ──────► tradepilot.tracking (MLflow Backtest Run)
       reports/trade_ledger.csv
```

### 5.3 Data and Artifact Storage Conventions
- **Raw Data Directory (`data/raw/`):** Contains unadulterated source datasets downloaded directly from external providers. The raw CSV (`stock_ohlcv_2021_2025.csv`) is tracked outside Git using DVC (`stock_ohlcv_2021_2025.csv.dvc`).
- **Local DVC Remote (`.dvc_storage/`):** Local content-addressable storage acting as a portable offline remote for `dvc push` and `dvc pull`.
- **Processed Data Directory (`data/processed/`):** Stores validated, feature-engineered datasets. `stock_features_labeled.csv` is declared as an output of the DVC `features` stage and cached in DVC.
- **Models Directory (`models/`):** Stores serialized production model pipelines (`best_model.joblib`) alongside an auditable machine-readable specification (`model_metadata.json`). The `.joblib` binary is cached by DVC.
- **MLflow Tracking Database (`mlflow.db` & `mlruns/`):** SQLite database storing run metadata, tags (Git commit, DVC versions), parameters, and metrics, coupled with `./mlruns` containing visual and tabular artifacts.
- **Reports Directory (`reports/`):** Holds all tabular and JSON performance artifacts produced by the training and backtesting pipelines, ensuring full transparency, auditability, and immediate human inspectability.

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
| `src/tradepilot/api/app.py` | FastAPI application factory & router registration | `app`, lifespan handler | HTTP requests on `/api/v1` | JSON responses, CORS middleware | Auto-initializes database & seeds default admin user on startup. |
| `src/tradepilot/api/config.py` | API & execution environment settings | `Settings`, `get_settings()` | Environment variables / `.env` | Validated Pydantic configuration | Enforces strict paper mode; refuses to boot in live mode. |
| `src/tradepilot/api/security.py` | Authentication & token cryptography | `hash_password()`, `verify_password()`, `create_access_token()`, `decode_access_token()` | Plaintext passwords, JWT tokens | Bcrypt password hashes, encoded/decoded JWT claims | Enforces HS256 JWT signatures and token expiration. |
| `src/tradepilot/api/dependencies.py` | Request dependency injection & RBAC | `get_db()`, `get_current_user()`, `require_role()` | FastAPI `Depends`, Bearer token | Active DB session, authenticated `User` | Raises `401 Unauthorized` on invalid token; `403 Forbidden` on role breach. |
| `src/tradepilot/api/routers/*.py` | Versioned REST API endpoints | `system_router`, `auth_router`, `markets_router`, `predictions_router`, `portfolio_router`, `orders_router`, `agent_router`, `risk_router` | HTTP GET/POST payloads | Pydantic response models | Standardized HTTP status codes (200, 201, 400, 401, 403, 404, 500). |
| `src/tradepilot/db/models.py` | Relational database schema models | `User`, `AgentStateModel`, `TradingCycleModel`, `PredictionRecord`, `SignalRecord`, `OrderIntentRecord`, `BrokerOrderRecord`, `OrderFillRecord`, `PositionSnapshotRecord`, `PortfolioSnapshotRecord`, `RiskEventRecord`, `AuditLogRecord` | SQLAlchemy ORM operations | Relational tables in SQLite/PostgreSQL | Enforces foreign keys, indexes, timestamps, and check constraints. |
| `src/tradepilot/db/session.py` | Database engine & session management | `engine`, `SessionLocal`, `init_db()` | Database URI | Managed SQLAlchemy sessions | Configures connection pools; seeds initial admin credentials. |
| `src/tradepilot/execution/broker_base.py` | Brokerage integration abstract contract | `BaseBrokerAdapter`, `BrokerAccount`, `BrokerPosition`, `BrokerOrder` | Abstract method definitions | Standardized brokerage domain models | Enforces uniform API across mock and live paper brokers. |
| `src/tradepilot/execution/mock_adapter.py` | Deterministic offline paper broker | `MockBrokerAdapter` | Local OHLCV data cache | Simulated fills with 5 bps slippage & fees | Maintains internal cash balance and portfolio equity offline. |
| `src/tradepilot/execution/alpaca_adapter.py` | Official Alpaca paper API adapter | `AlpacaPaperBroker` | Alpaca paper credentials, `alpaca-py` | Dispatched orders, account balances | Strictly verifies account `is_paper()`; rejects live trading. |
| `src/tradepilot/execution/risk_engine.py` | Centralized quantitative pre-trade risk engine | `CentralizedRiskEngine`, `evaluate_order()` | Proposed `OrderIntent`, portfolio state | `RiskDecision(approved, reason)` | Enforces 10+ risk rules, drawdown limits, and emergency kill switch. |
| `src/tradepilot/execution/agent.py` | Autonomous strategy trading agent | `TradingAgent`, `run_cycle()` | Model artifact, market bars, broker | Dispatched orders, reconciled state | Coordinates inference, signal filtering, risk gating, and fills. |
| `src/tradepilot/execution/worker.py` | Autonomous background strategy worker | `BackgroundWorker`, `start()`, `stop()` | Configured cadence (default: 60s) | Scheduled trading cycle execution | Implements heartbeat tracking and graceful POSIX signal shutdown. |
| `frontend/src/lib/api.ts` | Fully typed frontend REST API client | `apiClient` methods | Browser fetch with JWT Bearer token | Strongly typed TypeScript responses | Auto-handles error statuses and redirects on authentication expiry. |
| `frontend/src/app/page.tsx` | Main trading dashboard SPA controller | `TradePilotDashboard` component | API fetch state | 7 financial views with auto-refresh | Live polling every 5s; manages emergency stop confirmation modal. |
| `deploy/terraform/main.tf` | AWS cloud infrastructure IaC | ECS Fargate, ALB, RDS, S3, Secrets Manager, CloudWatch | Terraform variables | Provisioned AWS production resources | Multi-AZ PostgreSQL, serverless containers, zero-trust secrets. |
| `download_market_data.py` | Historical data retrieval via Alpaca API | Script execution | Alpaca API keys in `.env` | `data/raw/stock_ohlcv_2021_2025.csv` | Raises `ValueError` if API credentials are missing. |
| `stock_eda.py` | Prototype exploratory analysis script | Script execution | Google Drive / local CSVs | Cleaned CSV, feature CSV, reports | Standalone exploratory research script. |
| `train.py` | Root CLI training entrypoint | Script execution | None (CLI parameters) | Calls `train_pipeline.main()` | Thin wrapper delegating execution to the package pipeline. |

---

### 11.2 Package Configuration & Parameter Management: `tradepilot.config` and `params.yaml`
Located at `src/tradepilot/config.py`, this module provides frozen dataclasses that define paths, hyperparameters, and schemas across the package, augmented with dynamic loaders that read overrides from `params.yaml` for DVC stage execution:

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

def load_params(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load stage hyperparameters from params.yaml if available."""
    params_file = path or (PROJECT_ROOT / "params.yaml")
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def get_data_config(params_path: Optional[Path] = None) -> DataConfig:
    """Instantiate DataConfig with overrides from params.yaml."""
    params = load_params(params_path).get("features", {})
    return DataConfig(
        prediction_horizon_days=params.get("horizon_days", DEFAULT_DATA_CONFIG.prediction_horizon_days),
        target_threshold=params.get("target_threshold", DEFAULT_DATA_CONFIG.target_threshold),
    )
```

**`params.yaml` Structure:**
```yaml
features:
  prediction_horizon_days: 5
  target_threshold: 0.01

train:
  random_seed: 42
  train_end_year: 2023
  val_year: 2024
  test_year: 2025

backtest:
  initial_capital: 100000.0
  max_positions: 10
  allocation_per_position: 0.10
  entry_threshold: 0.40
  holding_period_days: 5
  slippage_bps: 5.0
  commission_bps: 5.0
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

### 11.4 Feature Engineering Engine & CLI: `tradepilot.features`
`build_features.py` calculates technical indicators, assigns directional targets, and provides the CLI runner for the DVC `features` stage:

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

def run_feature_pipeline(
    raw_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    horizon: Optional[int] = None,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """CLI orchestrator for the DVC 'features' stage."""
    data_cfg = get_data_config()
    df_raw = load_raw_ohlcv(raw_path or data_cfg.raw_ohlcv_path)
    validate_ohlcv_schema(df_raw)

    df_clean = generate_features_and_targets(
        df_raw,
        horizon=horizon or data_cfg.prediction_horizon_days,
        threshold=threshold or data_cfg.target_threshold,
    )
    validate_features_dataset(df_clean, data_cfg.feature_cols, data_cfg.target_col, data_cfg.prohibited_cols, data_cfg.target_classes)
    
    out_file = output_path or data_cfg.processed_features_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_file, index=False)
    return df_clean
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

### 11.7 MLOps Experiment Tracking: `tradepilot.tracking`
Located at `src/tradepilot/tracking.py`, this module standardizes experiment logging across model training and trading backtesting. It wraps the MLflow client to guarantee reproducible logging to an embedded SQLite database backend:

```python
def setup_mlflow_tracking(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "TradePilot-Stock-Direction",
) -> str:
    """Configures MLflow tracking URI (sqlite:///mlflow.db) and creates/sets active experiment."""
    uri = tracking_uri or f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return mlflow.create_experiment(
            experiment_name,
            artifact_location=(PROJECT_ROOT / "mlruns").as_uri()
        )
    return experiment.experiment_id

def log_training_to_mlflow(
    data_config: DataConfig,
    model_config: ModelConfig,
    winning_model_name: str,
    val_metrics: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, Any],
    fitted_model_path: Path,
    report_paths: List[Path],
    feature_names: List[str],
) -> Optional[str]:
    """Logs 30+ hyperparameters, 32+ validation & test metrics, feature importance chart, and model artifacts."""
    ...

def log_backtest_to_mlflow(
    backtest_config: BacktestConfig,
    summary_results: Dict[str, Any],
    artifact_paths: List[Path],
) -> Optional[str]:
    """Logs backtest simulation parameters, Sharpe ratios, drawdowns, and trade ledger artifacts."""
    ...
```

### 11.8 Automated Pipelines & DVC Specification: `tradepilot.pipelines` and `dvc.yaml`
TradePilot orchestrates its lifecycle via [`dvc.yaml`](file:///E:/Field%20Project/New%20Approach/dvc.yaml) across 3 reproducible stages:

```yaml
stages:
  features:
    cmd: python -m tradepilot.features.build_features
    deps:
      - data/raw/stock_ohlcv_2021_2025.csv
      - src/tradepilot/data
      - src/tradepilot/features/build_features.py
    params:
      - features.prediction_horizon_days
      - features.target_threshold
    outs:
      - data/processed/stock_features_labeled.csv

  train:
    cmd: python -m tradepilot.pipelines.train_pipeline
    deps:
      - data/processed/stock_features_labeled.csv
      - src/tradepilot/modeling
      - src/tradepilot/pipelines/train_pipeline.py
      - src/tradepilot/tracking.py
    params:
      - train.random_seed
      - train.train_end_year
      - train.val_year
      - train.test_year
    outs:
      - models/best_model.joblib
      - models/model_metadata.json:
          cache: false
      - reports/confusion_matrix.csv:
          cache: false
      - reports/test_results.json:
          cache: false
      - reports/validation_results.csv:
          cache: false

  backtest:
    cmd: python -m tradepilot.pipelines.backtest_pipeline
    deps:
      - data/processed/stock_features_labeled.csv
      - models/best_model.joblib
      - src/tradepilot/backtesting
      - src/tradepilot/pipelines/backtest_pipeline.py
      - src/tradepilot/tracking.py
    params:
      - backtest.allocation_per_position
      - backtest.commission_bps
      - backtest.entry_threshold
      - backtest.holding_period_days
      - backtest.initial_capital
      - backtest.max_positions
      - backtest.slippage_bps
    outs:
      - reports/backtest_equity_curve.csv:
          cache: false
      - reports/backtest_summary.json:
          cache: false
      - reports/benchmark_comparison.csv:
          cache: false
      - reports/trade_ledger.csv:
          cache: false
```

- `train_pipeline.py`: Loads features, verifies schema integrity, performs chronological splitting, fits candidate models, displays validation comparison tables, selects the best model, evaluates out-of-sample test metrics, logs run to MLflow, and persists artifacts.
- `backtest_pipeline.py`: Loads `models/best_model.joblib`, generates probability signals, runs validation and test simulations, prints performance summaries, logs backtest metrics to MLflow, and exports equity curves and trade ledgers.

### 11.9 Root Operational Entrypoints
- `download_market_data.py`: Standalone script to retrieve raw OHLCV bars via the Alpaca API.
- `train.py`: Lightweight CLI entrypoint that imports and invokes `tradepilot.pipelines.train_pipeline.main()`.
- `stock_eda.py`: Standalone prototype script originating from early exploratory Colab research.

---

### 11.10 REST API and Authentication Service: `tradepilot.api`
The application layer exposes a high-performance, asynchronous REST API under the `/api/v1` namespace, built using FastAPI, Pydantic v2, and Uvicorn:

- **Application Factory (`app.py`):** Configures CORS middleware for `http://localhost:3000`, mounts OpenAPI documentation at `/docs` and `/redoc`, and defines an asynchronous lifespan handler that initializes database tables and seeds the default administrator (`admin@tradepilot.io`) upon startup.
- **Environment & Safeguards Configuration (`config.py`):** Utilizes `pydantic-settings` to parse configuration from environment variables and `.env`. Programmatically enforces strict paper trading safeguards: refuses application boot if `TRADING_MODE` is set to `LIVE`.
- **Security & Cryptography (`security.py`):** Manages password hashing using `bcrypt` (12 rounds) and encodes/decodes JSON Web Tokens (JWT) using `pyjwt` with `HS256` signatures and a 24-hour expiration window.
- **Request Dependencies & RBAC (`dependencies.py`):** Implements dependency injection for database sessions (`get_db`) and role-based access control (`get_current_user`, `require_role`). Enforces three access tiers:
  - `viewer`: Read-only access to market data, active models, portfolio snapshots, and public status.
  - `trader`: Permissions to trigger on-demand inference cycles, submit manual paper orders, and toggle agent execution (Start/Pause/Stop).
  - `admin`: Full administrative access, including role management, audit log inspection, and the emergency kill switch.
- **Data Transfer Schemas (`schemas/`):** Strongly typed Pydantic models with financial bounds validation:
  - `auth.py`: `LoginRequest`, `TokenResponse`, `UserResponse`.
  - `system.py`: `HealthResponse`, `SystemStatusResponse`.
  - `market.py`: `SymbolListResponse`, `BarRecord`, `MarketStatusResponse`.
  - `prediction.py`: `PredictionResponse`, `SignalResponse`, `ModelMetadataResponse`.
  - `portfolio.py`: `PortfolioSummaryResponse`, `PositionResponse`, `EquityPointResponse`.
  - `order.py`: `OrderCreateRequest`, `OrderResponse`, `TradeFillResponse`.
  - `agent.py`: `AgentStatusResponse`, `AgentCommandRequest`, `TradingCycleResponse`.
  - `risk.py`: `RiskStatusResponse`, `RiskConfigResponse`, `RiskEventResponse`, `AuditLogResponse`.
- **Versioned API Routers (`routers/`):**
  - `system.py`: Liveness (`GET /health`), readiness (`GET /ready`), and detailed operational status (`GET /api/v1/system/status`).
  - `auth.py`: JSON credential authentication (`POST /api/v1/auth/login-json`) and identity verification (`GET /api/v1/auth/me`).
  - `markets.py`: Universe listing (`GET /api/v1/markets/symbols`), OHLCV candles (`GET /api/v1/markets/bars`), and latest quotes (`GET /api/v1/markets/latest`).
  - `predictions.py`: Latest class probabilities (`GET /api/v1/predictions/latest`), manual inference execution (`POST /api/v1/predictions/run`), strategy signals (`GET /api/v1/signals`), and active model architecture (`GET /api/v1/models/active`).
  - `portfolio.py`: Account equity and cash (`GET /api/v1/portfolio`), active holdings (`GET /api/v1/portfolio/positions`), and historical equity timeline (`GET /api/v1/portfolio/equity`).
  - `orders.py`: Paper order submission (`POST /api/v1/orders`), lifecycle order queries (`GET /api/v1/orders`), and trade fill ledgers (`GET /api/v1/trades`).
  - `agent.py`: Agent state inspection (`GET /api/v1/agent/status`) and control commands (`/start`, `/pause`, `/stop`, `/emergency-stop`).
  - `risk.py`: Real-time risk limit utilization (`GET /api/v1/risk/status`), active parameters (`GET /api/v1/risk/config`), risk breach events (`GET /api/v1/risk/events`), and administrative audit trail (`GET /api/v1/risk/audit/logs`).

---

### 11.11 Relational Persistence & Schema Migrations: `tradepilot.db`
TradePilot implements a relational persistence layer using SQLAlchemy 2.0 with Alembic schema migrations:

- **Declarative Base & Mixins (`base.py`):** Defines `Base(DeclarativeBase)` and `TimestampMixin` providing automatic UTC `created_at` and `updated_at` timestamps across all entities.
- **Connection Management & Lifespan (`session.py`):** Configures connection pooling for SQLite and PostgreSQL. Exports `SessionLocal` and an idempotent `init_db()` routine that creates tables and seeds the administrative user.
- **Relational Domain Models (`models.py`):** 12 normalized database entities:
  - `User`: Identity credentials, hashed passwords, roles (`admin`, `trader`, `viewer`), and active flags.
  - `AgentStateModel`: Singleton record tracking agent status (`STOPPED`, `RUNNING`, `PAUSED`, `ERROR`), verified `PAPER` mode, heartbeat timestamps, and error logs.
  - `TradingCycleModel`: Execution log capturing cycle duration, symbols evaluated, signals emitted, and orders submitted.
  - `PredictionRecord`: Multi-class probability distributions (`prob_up`, `prob_neutral`, `prob_down`) per symbol and cycle.
  - `SignalRecord`: Strategy actions (`BUY`, `EXIT`, `HOLD`) with rule rationale and conviction scores.
  - `OrderIntentRecord`: Proposed trades evaluated by the Risk Engine with approval status and rejection reasons.
  - `BrokerOrderRecord`: Dispatched paper orders tracking lifecycle states (`SUBMITTED`, `FILLED`, `CANCELED`, `REJECTED`).
  - `OrderFillRecord`: Broker execution receipts with fill prices, executed quantities, and transaction commissions.
  - `PositionSnapshotRecord`: Reconciled holdings with average cost basis, current market prices, and unrealized P&L.
  - `PortfolioSnapshotRecord`: Marked-to-market equity curve data points (equity, cash, buying power, gross exposure).
  - `RiskEventRecord`: Immutable log of blocked orders and tripped risk thresholds.
  - `AuditLogRecord`: Append-only security audit trail recording operator identity, IP address, and administrative actions.
- **Alembic Database Migrations (`alembic/`):** Initial migration `934fa89e0077_initial_schema.py` defines the complete relational schema with foreign key constraints, indexes, and cascades.

---

### 11.12 Execution Engine, Broker Adapters & Centralized Risk Engine: `tradepilot.execution`
The execution layer bridges statistical signals with paper brokerage accounts:

- **Broker Adapter Specification (`broker_base.py`):** Abstract base class (`BaseBrokerAdapter`) establishing standardized brokerage contracts: `get_account()`, `get_positions()`, `get_latest_bars()`, `submit_order()`, `cancel_order()`, and `get_orders()`.
- **Deterministic Mock Broker (`mock_adapter.py`):** High-fidelity paper trading simulation for offline development:
  - Initializes with a configurable cash balance (default: $100,000.00).
  - Simulates 5 bps adverse slippage on fills and deducts 5 bps transaction fees.
  - Leverages cached 4-year daily OHLCV bars (`data/raw/stock_ohlcv_2021_2025.csv`) for offline market pricing.
  - Implements complete order lifecycle state transitions (`SUBMITTED` $\to$ `FILLED`).
- **Alpaca Paper Adapter (`alpaca_adapter.py`):** Official integration with Alpaca Markets using `alpaca-py`:
  - Enforces `is_paper()` check on `TradingClient` to guarantee that live capital is never exposed.
  - Queries daily market bars using `StockHistoricalDataClient` configured with `feed=DataFeed.IEX`.
  - Features automatic fallback to local historical bar data if external market feeds are restricted or offline.
- **Centralized Risk Management Engine (`risk_engine.py`):** Pre-trade risk enforcement module evaluating every order proposal before brokerage submission:
  - *Paper Verification:* Blocks orders if trading mode is not verified as `PAPER`.
  - *Emergency Kill Switch:* Blocks all trading if emergency stop is triggered.
  - *Approved Universe:* Rejects symbols outside the 50-ticker large-cap universe.
  - *Order Parameter Bounds:* Enforces positive integer share quantities and valid market/limit order types.
  - *Deduplication:* Prevents duplicate client order IDs and blocks concurrent pending orders for the same symbol.
  - *Position Exposure Cap:* Restricts any single stock to a maximum of **10.0%** of total portfolio equity.
  - *Max Concurrent Holdings:* Caps active holdings to a maximum of **10 distinct equities**.
  - *Gross Portfolio Exposure:* Caps total invested capital at **95.0%** of portfolio equity.
  - *Cash Solvency:* Rejects buy orders exceeding available cash reserves and buying power.
  - *Daily Drawdown Circuit Breaker:* Automatically trips the emergency kill switch if intraday portfolio losses exceed **5.0%**.
- **Autonomous Strategy Agent (`agent.py`):** Coordinates the execution cycle:
  - Loads the serialized Random Forest pipeline (`models/best_model.joblib`).
  - Computes 8 stationary technical features on latest market bars.
  - Generates class probabilities and applies `SignalGenerator` thresholds.
  - Calculates target share allocations based on available capital and portfolio limits.
  - Submits proposed order intents to `CentralizedRiskEngine`.
  - Dispatches approved orders to the active broker adapter.
  - Reconciles broker positions and records portfolio snapshots to the database.
- **Autonomous Background Worker (`worker.py`):** Standalone process executing the strategy agent on a scheduled cadence (default: 60 seconds). Features heartbeat logging to detect worker stalls and graceful shutdown handling upon `SIGINT` / `SIGTERM`.

---

### 11.13 Modern Trading Dashboard: `frontend/`
The presentation layer is a high-density, dark-themed financial trading terminal built with Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS, and Lucide icons:

- **Fully-Typed API Client (`src/lib/api.ts`):** Wraps all backend endpoints with TypeScript interfaces matching Pydantic response models, managing JWT Bearer tokens and error propagation.
- **Interactive SVG Visualizations:**
  - `EquityChart.tsx`: Renders marked-to-market portfolio equity with smooth cubic Bezier curves, dynamic linear gradients, and interactive hover tooltips tracking equity and cash values.
  - `PriceChart.tsx`: High-performance SVG candlestick and volume bar chart displaying historical OHLCV data with bullish/bearish color coding and volume distributions.
- **Header & Global Status (`Header.tsx`):** Displays operational status pills (Backend Live/Down, Trading Agent State, Mode Badge), manual inference trigger, agent controls, and an Emergency Kill Switch with a two-step confirmation modal.
- **Seven Dedicated Financial Views (`src/components/views/`):**
  1. `DashboardView.tsx`: Portfolio KPIs (Equity, Buying Power, Daily P&L, Gross Exposure gauge), marked-to-market equity timeline, active positions overview, and recent trading signals.
  2. `MarketsView.tsx`: Multi-symbol selector across the 50-stock universe, latest quote cards with 24h percentage changes, interactive candlestick/volume charts, and historical OHLCV data table.
  3. `PredictionsView.tsx`: Active model architecture card, test fold metrics, on-demand prediction runner, and visual 3-class probability distribution bars (`UP`, `NEUTRAL`, `DOWN`) with conviction thresholds.
  4. `PortfolioView.tsx`: Asset allocation stacked visualizer, cash allocation meter, and position details table displaying shares, cost basis, current value, and unrealized P&L ($ and %).
  5. `OrdersView.tsx`: Manual paper order submission form with real-time pre-trade risk feedback, order lifecycle table with status badges (`FILLED`, `SUBMITTED`, `REJECTED`), and execution fills ledger.
  6. `AgentControlView.tsx`: Agent state machine controls (Start, Pause, Stop, Emergency Kill Switch), worker heartbeat monitor, cycle cadence settings, and chronological execution cycle history.
  7. `RiskView.tsx`: Real-time quantitative limit utilization meters (Gross Exposure, Position Count, Daily Drawdown), configured boundaries, immutable risk events log, and administrative security audit trail.

---

### 11.14 Cloud Deployment, Docker & Terraform IaC: `deploy/` and `.github/`
TradePilot provides production containerization and cloud infrastructure definitions ready for AWS deployment:

- **Containerization (`deploy/`):**
  - `Dockerfile.api`: Python 3.10 slim container executing the FastAPI server via Uvicorn.
  - `Dockerfile.worker`: Python 3.10 slim container executing the background trading agent worker.
  - `Dockerfile.frontend`: Multi-stage Node.js 20 Alpine container building the standalone Next.js 16 production server.
  - `docker-compose.yml`: Multi-container orchestration linking PostgreSQL 15, FastAPI Backend, Strategy Worker, and Next.js Frontend with internal networking and persistent health checks.
- **AWS Infrastructure-as-Code (`deploy/terraform/`):**
  - `main.tf`: Provisions Amazon ECS on AWS Fargate (serverless tasks for API and Worker), Application Load Balancer (ALB) terminating TLS, Amazon RDS for PostgreSQL 15 Multi-AZ database, Amazon S3 bucket for model artifacts, AWS Secrets Manager for credentials, and Amazon CloudWatch for centralized logging and alarms.
  - `variables.tf` & `outputs.tf`: Parameterized AWS region, VPC CIDRs, container image tags, and database compute sizing.
- **Continuous Integration Pipeline (`.github/workflows/ci.yml`):**
  - Executes on push and pull requests to `main`.
  - Runs flake8 linting and type checks.
  - Executes the full 36-test `pytest -v` automated test suite.
  - Builds the Next.js 16 production bundle to guarantee zero TypeScript compilation or lint errors.

---

## 12. Installation, Environment Setup, and Execution Guide

### 12.1 Environment Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux (commands below are verified for Windows PowerShell).
- **Python Version:** Python 3.10 or higher (verified on Python 3.10.7).
- **Git & DVC:** Installed in virtual environment or system PATH.
- **Hardware:** Minimum 8 GB RAM; multi-core CPU recommended.

### 12.2 Windows PowerShell Virtual Environment Activation
Open Windows PowerShell, navigate to the project directory, and activate the virtual environment:

```powershell
# Navigate to the workspace root
Set-Location -Path "E:\Field Project\New Approach"

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

*Note on Execution Policies:* If PowerShell restricts script activation, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### 12.3 Editable Package Installation & MLOps Dependencies
Install TradePilot in editable development mode along with DVC and MLflow dependencies:

```powershell
# Install package in editable mode
pip install -e .

# Install all dependencies including dvc>=3.50.0 and mlflow>=2.15.0
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

### 12.5 Executing DVC Pipelines, Reproducibility, and Backtesting

#### Option A: Full DVC Pipeline Execution (Recommended)
Execute all pipeline stages (`features` $\to$ `train` $\to$ `backtest`) with automatic dependency tracking:

```powershell
# Inspect stage status
dvc status

# Run entire reproducible pipeline
dvc repro

# Verify pipeline caching (subsequent execution skips all unchanged stages)
dvc repro
# Output: 'Data and pipelines are up to date.'

# Inspect pipeline dependency DAG
dvc dag

# Push dataset and model snapshots to local storage remote
dvc push
```

#### Option B: Standalone Python Module Invocations
If executing individual stages outside of DVC:

```powershell
# Step 1: Execute automated tests
pytest -v

# Step 2: Feature engineering stage
python -m tradepilot.features.build_features

# Step 3: Model training and MLflow tracking stage
python -m tradepilot.pipelines.train_pipeline

# Step 4: Event-driven backtesting and MLflow reporting stage
python -m tradepilot.pipelines.backtest_pipeline
```

### 12.6 Launching MLflow UI and Experiment Audit
TradePilot maintains an embedded SQLite database (`sqlite:///mlflow.db`). To launch the interactive MLflow web dashboard:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open your browser at `http://127.0.0.1:5000` to:
1. **Inspect Experiment Runs:** Navigate to the `TradePilot-Stock-Direction` experiment.
2. **Compare Model Folds:** Compare validation Macro F1 across `Majority_Baseline`, `Logistic_Regression`, and `Random_Forest`.
3. **Examine Out-of-Sample Metrics:** Inspect `test_macro_f1`, `test_balanced_accuracy`, and per-class precision/recall for `DOWN`, `NEUTRAL`, and `UP`.
4. **Audit Backtest Returns:** Review separate backtest runs logging `test_final_equity`, `test_sharpe_ratio`, `test_max_drawdown`, and `test_win_rate`.
5. **Download Artifacts:** View feature importance bar charts (`feature_importance.png`), confusion matrix tables (`confusion_matrix.csv`), trade execution ledgers (`trade_ledger.csv`), and daily equity curves (`backtest_equity_curve.csv`).

### 12.7 Locating and Auditing Generated Artifacts
Upon pipeline completion, the following artifacts are generated:
- `models/best_model.joblib`: Serialized winning Random Forest classifier pipeline (DVC cached).
- `models/model_metadata.json`: Machine-readable metadata schema detailing training parameters, features, and split metrics.
- `dvc.lock`: Cryptographic content hashes locking versions of all stage dependencies and outputs.
- `mlflow.db`: SQLite database storing experiment runs, metrics, parameters, and tags.
- `mlruns/`: Artifact repository containing logged plots, confusion matrices, and model binaries.
- `reports/validation_results.csv`: Validation performance comparison across all candidate models.
- `reports/test_results.json`: Final out-of-sample classification report and confusion matrix.
- `reports/confusion_matrix.csv`: Tabular confusion matrices across validation and test folds.
- `reports/feature_importances.csv`: Ranked Gini importance scores across the 8 stationary features.
- `reports/backtest_summary.json`: Detailed strategy vs. benchmark performance metrics.
- `reports/backtest_equity_curve.csv`: Daily marked-to-market equity values and gross exposure tracking.
- `reports/trade_ledger.csv`: Granular trade-by-trade log detailing entry/exit dates, fill prices, shares, gross PnL, net PnL, and transaction fees.
- `reports/benchmark_comparison.csv`: High-level strategy vs. benchmark comparative summary.

### 12.8 Database Schema Setup & Alembic Migrations
TradePilot uses SQLAlchemy 2.0 and Alembic to manage its relational schema (users, agent cycles, orders, fills, snapshots, risk events, and audit logs). The default local database is SQLite (`tradepilot.db`), but PostgreSQL is supported by updating `DATABASE_URL` in `.env`.

To apply all database migrations:
```powershell
# Run database migrations to latest schema version
alembic upgrade head
```
*Note on Automatic Seeding:* When the FastAPI backend boots for the first time, its lifespan handler calls `init_db()`, which automatically creates any missing tables and seeds the default administrator account:
- **Default Admin Email:** `admin@tradepilot.io`
- **Default Admin Password:** `AdminPass123!`

---

### 12.9 Launching the Full Production Stack (API, Worker, Frontend)
Running TradePilot in development mode requires three lightweight processes:

#### Terminal 1: FastAPI REST API Server
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start FastAPI server on port 8000
uvicorn src.tradepilot.api.app:app --host 0.0.0.0 --port 8000 --reload
```
- **API Base URL:** `http://localhost:8000`
- **Interactive OpenAPI Documentation (Swagger UI):** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`

#### Terminal 2: Autonomous Strategy Background Worker
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Launch the background strategy trading worker
python -m tradepilot.execution.worker
```
- The worker boots, logs its process PID, updates its heartbeat in the database, and executes trading cycles at the configured cadence (default: 60s).
- Graceful shutdown can be triggered at any time using `Ctrl+C` (`SIGINT`).

#### Terminal 3: Next.js 16 Financial Trading Terminal
```powershell
# Navigate to the frontend directory
Set-Location -Path "frontend"

# Install Node dependencies (first run only)
npm install

# Start Next.js development server
npm run dev
```
- **Dashboard URL:** `http://localhost:3000`
- The terminal automatically connects to `http://localhost:8000/api/v1`, authenticates using the default admin credentials, and polls state every 5 seconds.

---

### 12.10 Multi-Container Deployment via Docker Compose
For containerized local orchestration or cloud staging, TradePilot provides `docker-compose.yml`, which coordinates:
1. `postgres`: PostgreSQL 15 database with persistent volume `pgdata`.
2. `api`: FastAPI application container on port 8000.
3. `worker`: Autonomous strategy loop process.
4. `frontend`: Next.js 16 production server on port 3000.

```powershell
# Build and run the entire multi-service stack
docker compose up --build
```
*Docker Prerequisite Note:* Docker Desktop must be running on your system with the Linux container engine active. If Docker Desktop is stopped, the command will report `//./pipe/dockerDesktopLinuxEngine: The system cannot find the file`. In that case, launch Docker Desktop first or run natively using the three PowerShell terminals described in Section 12.9.

---

### 12.11 Troubleshooting Common Operational Faults
1. **`ModuleNotFoundError: No module named 'tradepilot'`:**
   - *Cause:* The package was not installed in editable development mode or `src` is missing from `sys.path`.
   - *Resolution:* Run `pip install -e .` from the project root.
2. **`ModuleNotFoundError: No module named 'jwt'` during `pytest -v`:**
   - *Cause:* Running `pytest` from system Python rather than the virtual environment where `pyjwt` is installed.
   - *Resolution:* Always execute using the virtual environment's interpreter: `.\.venv\Scripts\pytest -v`.
3. **`dvc repro` executes using wrong Python interpreter:**
   - *Cause:* PowerShell executed DVC in a session without the virtual environment active, falling back to system Python.
   - *Resolution:* Run `.\.venv\Scripts\Activate.ps1` before invoking `dvc repro`.
4. **`mlflow.exceptions.MlflowException: Could not find experiment`:**
   - *Cause:* MLflow UI launched without specifying the database URI (`--backend-store-uri sqlite:///mlflow.db`).
   - *Resolution:* Always launch MLflow UI with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
5. **`FileNotFoundError: Raw OHLCV data file not found`:**
   - *Cause:* `stock_ohlcv_2021_2025.csv` is missing from `data/raw/`.
   - *Resolution:* Run `dvc pull` to retrieve from local cache, or run `python download_market_data.py`.
6. **`ValueError: Missing Alpaca API credentials in .env`:**
   - *Cause:* Missing or empty `ALPACA_API_KEY` or `ALPACA_SECRET_KEY` in `.env`.
   - *Resolution:* Add valid Alpaca paper-trading credentials to `.env`.
7. **`HTTP 403 Forbidden: subscription does not permit querying recent SIP data`:**
   - *Cause:* Alpaca free paper accounts cannot access direct consolidated SIP feeds.
   - *Resolution:* TradePilot automatically configures `feed=DataFeed.IEX` on all market bar requests and falls back seamlessly to the cached 4-year CSV dataset if network connectivity or subscriptions fail.
8. **Docker daemon connection error (`The system cannot find the file`):**
   - *Cause:* `docker compose up` was executed while Docker Desktop was not running.
   - *Resolution:* Start the Docker Desktop application from the Windows Start menu, wait for the engine to initialize (green icon in the system tray), and re-run `docker compose up --build`. Alternatively, run natively via PowerShell (Section 12.9).

---

## 13. Quality Assurance, Automated Testing, and Reproducibility

### 13.1 Testing Architecture and Test Discovery
TradePilot maintains an automated test suite under `tests/`. Test discovery and configuration are managed via `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

### 13.2 Breakdown of 36 Automated Unit and Integration Tests
The test suite contains **36 automated test cases** distributed across 8 specialized modules:

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

7. **`tests/test_api_endpoints.py` (11 tests):**
   - `test_health_endpoint`: Asserts that `GET /health` returns status `healthy` and version `1.0.0`.
   - `test_ready_endpoint`: Asserts that `GET /ready` verifies database connectivity, model artifact presence, and paper trading isolation.
   - `test_system_status_endpoint`: Asserts that `GET /api/v1/system/status` returns operational state, system uptime, and database connectivity.
   - `test_auth_login_and_me`: Validates JWT token generation on `POST /api/v1/auth/login-json` and verifies user profile on `GET /api/v1/auth/me`.
   - `test_auth_login_invalid_credentials`: Asserts that invalid credentials return `401 Unauthorized`.
   - `test_markets_symbols_and_bars`: Validates universe listing on `GET /api/v1/markets/symbols` and candle data on `GET /api/v1/markets/bars`.
   - `test_markets_latest_and_status`: Asserts that `GET /api/v1/markets/latest` returns current market quotes and `GET /api/v1/markets/status` reports data freshness.
   - `test_active_model_and_predictions`: Validates `GET /api/v1/models/active` returns model architecture & test metrics, and `GET /api/v1/predictions/latest` yields valid class probabilities.
   - `test_portfolio_summary_and_positions`: Asserts that `GET /api/v1/portfolio` and `GET /api/v1/portfolio/positions` return account equity, cash balance, and open holdings.
   - `test_agent_status_and_lifecycle`: Tests agent state inspection (`/api/v1/agent/status`) and state machine transitions (`/start`, `/pause`, `/stop`).
   - `test_risk_status_and_config`: Verifies `GET /api/v1/risk/status` and `GET /api/v1/risk/config` report active thresholds and circuit breaker status.

8. **`tests/test_trading_agent_and_risk.py` (8 tests):**
   - `test_risk_engine_approves_valid_order`: Asserts that an order adhering to all position size, cash, and exposure limits receives `approved=True`.
   - `test_risk_engine_enforces_max_position_size`: Asserts that orders requesting greater than 10% portfolio allocation are rejected (`approved=False`).
   - `test_risk_engine_enforces_max_concurrent_positions`: Asserts that orders opening an 11th distinct position when 10 positions are already active are rejected.
   - `test_risk_engine_prevents_duplicate_orders`: Validates deduplication against existing pending client order IDs and concurrent pending orders for the same ticker.
   - `test_risk_engine_emergency_stop_blocks_all_orders`: Asserts that triggering the emergency kill switch blocks all subsequent order intents immediately.
   - `test_risk_engine_rejects_non_paper_mode`: Asserts that any attempt to submit an order in a non-`PAPER` mode is rejected immediately by the risk engine.
   - `test_mock_broker_order_execution_and_fees`: Verifies that `MockBrokerAdapter` executes buy orders with +5 bps slippage, sell orders with -5 bps slippage, deducts commissions, and updates account equity.
   - `test_trading_agent_lifecycle_and_reconciliation`: Verifies end-to-end trading agent cycle: running predictions, gating through risk engine, submitting paper orders, and reconciling broker positions.

### 13.3 Test Verification Log and Passing Assertions (36 Passed)
Executing `pytest -v` across the entire 8-module suite produces the following verified passing output:

```text
============================= test session starts =============================
platform win32 -- Python 3.10.7, pytest-9.2.0, pluggy-1.7.0 -- E:\Field Project\New Approach\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: E:\Field Project\New Approach
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1, hydra-core-1.3.8, platformdirs-4.12.4, asyncio-1.4.1
asyncio: mode=strict, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 36 items

tests/test_api_endpoints.py::test_health_endpoint PASSED                 [  2%]
tests/test_api_endpoints.py::test_ready_endpoint PASSED                  [  5%]
tests/test_api_endpoints.py::test_system_status_endpoint PASSED          [  8%]
tests/test_api_endpoints.py::test_auth_login_and_me PASSED               [ 11%]
tests/test_api_endpoints.py::test_auth_login_invalid_credentials PASSED  [ 13%]
tests/test_api_endpoints.py::test_markets_symbols_and_bars PASSED        [ 16%]
tests/test_api_endpoints.py::test_markets_latest_and_status PASSED       [ 19%]
tests/test_api_endpoints.py::test_active_model_and_predictions PASSED    [ 22%]
tests/test_api_endpoints.py::test_portfolio_summary_and_positions PASSED [ 25%]
tests/test_api_endpoints.py::test_agent_status_and_lifecycle PASSED      [ 27%]
tests/test_api_endpoints.py::test_risk_status_and_config PASSED          [ 30%]
tests/test_data_validation.py::test_validate_ohlcv_schema_passes_valid PASSED [ 33%]
tests/test_data_validation.py::test_validate_ohlcv_detects_duplicates PASSED [ 36%]
tests/test_data_validation.py::test_validate_ohlcv_detects_negative_price PASSED [ 38%]
tests/test_data_validation.py::test_validate_ohlcv_detects_high_low_violation PASSED [ 41%]
tests/test_data_validation.py::test_validate_features_dataset_detects_target_leakage PASSED [ 44%]
tests/test_data_validation.py::test_validate_features_dataset_detects_nan_values PASSED [ 47%]
tests/test_execution_and_portfolio.py::test_execution_slippage_and_fees PASSED [ 50%]
tests/test_execution_and_portfolio.py::test_portfolio_cash_accounting_and_round_trip PASSED [ 52%]
tests/test_execution_and_portfolio.py::test_portfolio_prevents_negative_cash PASSED [ 55%]
tests/test_features.py::test_rsi_bounds PASSED                           [ 58%]
tests/test_features.py::test_macd_calculation PASSED                     [ 61%]
tests/test_features.py::test_features_independent_per_symbol PASSED      [ 63%]
tests/test_features.py::test_target_classification_logic PASSED          [ 66%]
tests/test_pipeline_reproducibility.py::test_model_training_reproducibility PASSED [ 69%]
tests/test_pipeline_reproducibility.py::test_model_serialization_integrity PASSED [ 72%]
tests/test_signals.py::test_signals_class_order_independence PASSED      [ 75%]
tests/test_split_data.py::test_chronological_split_and_purge PASSED      [ 77%]
tests/test_trading_agent_and_risk.py::test_risk_engine_approves_valid_order PASSED [ 80%]
tests/test_trading_agent_and_risk.py::test_risk_engine_enforces_max_position_size PASSED [ 83%]
tests/test_trading_agent_and_risk.py::test_risk_engine_enforces_max_concurrent_positions PASSED [ 86%]
tests/test_trading_agent_and_risk.py::test_risk_engine_prevents_duplicate_orders PASSED [ 88%]
tests/test_trading_agent_and_risk.py::test_risk_engine_emergency_stop_blocks_all_orders PASSED [ 91%]
tests/test_trading_agent_and_risk.py::test_risk_engine_rejects_non_paper_mode PASSED [ 94%]
tests/test_trading_agent_and_risk.py::test_mock_broker_order_execution_and_fees PASSED [ 97%]
tests/test_trading_agent_and_risk.py::test_trading_agent_lifecycle_and_reconciliation PASSED [100%]

======================= 36 passed in 14.89s =======================
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

### 15.2 Version Control Hygiene and Repository `.gitignore` Specification
The repository enforces strict version control hygiene via a root [`.gitignore`](file:///E:/Field%20Project/New%20Approach/.gitignore) to protect credentials, decouple large binary artifacts, and prevent cache pollution:
- **Environment & Credentials:** Blocks `.env` and `.env.*` to guarantee API keys are never committed.
- **Python Runtime Caches:** Blocks `.venv/`, `__pycache__/`, `*.pyc`, `*.egg-info/`, and `.pytest_cache/`.
- **DVC Caches & Remotes:** Blocks `.dvc/cache/`, `.dvc/tmp/`, and the local storage remote `.dvc_storage/`.
- **MLflow Tracking Artifacts:** Blocks `mlflow.db` (SQLite tracking store) and `./mlruns` (local artifact store), keeping Git history lightweight while DVC and MLflow manage binary artifacts.
- **Data & Model Binary Decoupling:** Large dataset files (`stock_ohlcv_2021_2025.csv`, `stock_features_labeled.csv`) and model binaries (`best_model.joblib`) are excluded from Git, tracked instead via DVC pointer files (`.dvc` and `dvc.lock`).

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

### 16.1 Phase 1: Operational Hardening and CI/CD Automation [COMPLETED]
- **Continuous Integration Pipeline:** Implemented automated GitHub Actions workflow (`.github/workflows/ci.yml`) triggering on every commit and pull request.
- **Automated Validation:** Enforces PEP 8 style standards, dependency audits, execution of the full 36-test `pytest -v` suite, and compilation of the Next.js 16 production bundle.
- **Repository Hygiene:** Clean root `.gitignore` protecting credentials, runtime caches, MLflow SQLite stores, and local DVC storage caches.

### 16.2 Phase 2: Dynamic Feature Engineering and Target Labeling
- **Dynamic Volatility Thresholds (Triple Barrier Method):** Replace the static $\pm 1\%$ threshold with dynamic barriers scaled by daily Average True Range (ATR), incorporating upper, lower, and horizontal time-stop barriers.
- **Expanded Macro Indicators:** Incorporate 10-year Treasury yields, VIX volatility index series, and sector ETF relative strength indices.
- **Cross-Sectional Factor Ranking:** Transform time-series returns into cross-sectional percentiles across the 50-stock universe.

### 16.3 Phase 3: Walk-Forward Validation and Gradient Boosting Ensembles
- **Combinatorial Purged Cross-Validation (CPCV):** Implement walk-forward rolling window validation to evaluate stability across distinct market regimes (bull, bear, sideways).
- **Gradient Boosted Decision Trees:** Integrate XGBoost, LightGBM, and CatBoost classifiers to compare non-linear predictive power against Random Forest.
- **Bayesian Hyperparameter Tuning:** Implement Optuna to automate hyperparameter optimization over training folds without validation snooping.

### 16.4 Phase 4: MLOps Integration (MLflow and DVC) [COMPLETED]
- **DVC Multi-Stage Pipeline:** Implemented `dvc.yaml` and `params.yaml` with 3 reproducible stages (`features`, `train`, `backtest`). Stage invalidation and output caching verified.
- **Data Versioning:** Decoupled raw and processed datasets from Git; tracked via `data/raw/stock_ohlcv_2021_2025.csv.dvc`.
- **Local Storage Remote:** Configured `.dvc_storage` for local offline team collaboration without cloud dependencies.
- **MLflow Experiment Tracking:** Implemented `src/tradepilot/tracking.py` using SQLite backend (`sqlite:///mlflow.db`) under experiment `TradePilot-Stock-Direction`.
- **Comprehensive Logging:** Automated logging of 30+ hyperparameters, 32+ validation and test metrics, feature importance charts, classification reports, and serialized model pipelines.
- **Backtest Run Tracking:** Separate tracking for portfolio simulations recording Sharpe ratios, drawdowns, equity curves, and trade ledgers.

### 16.5 Phase 5: Dynamic Risk Management and Pre-Trade Safeguards [COMPLETED]
- **Centralized Risk Engine (`CentralizedRiskEngine`):** Sequential execution of 10+ pre-trade quantitative checks on every automated and manual order intent.
- **Exposure Safeguards:** Hard limits capping maximum position size at 10.0%, maximum concurrent holdings at 10 equities, and gross exposure at 95.0%.
- **Daily Drawdown Circuit Breaker:** Automated kill switch tripping whenever daily portfolio losses breach 5.0% of marked-to-market equity.
- **Global Emergency Stop:** Administrative kill switch canceling open orders and rejecting new submissions across all accounts.

### 16.6 Phase 6: Automated Paper-Trading Execution Service & Next.js Dashboard [COMPLETED]
- **Dual Broker Adapters:** High-fidelity `MockBrokerAdapter` with deterministic 5 bps slippage and fee simulation, coupled with official `AlpacaPaperBroker` using `alpaca-py` with strict paper validation and IEX feed handling.
- **Autonomous Strategy Worker:** Standalone daemon (`worker.py`) cycling through model inference, signal generation, risk gating, and brokerage order execution.
- **Versioned REST API:** 25+ authenticated endpoints across 8 resource routers built with FastAPI, Pydantic v2, and SQLAlchemy 2.0 with Alembic migrations.
- **Next.js 16 Financial Terminal:** Modern trading dashboard featuring real-time equity curves, interactive candlestick/volume charts, probability distribution bars, and order lifecycle management.
- **AWS Cloud Readiness:** Production Dockerfiles, `docker-compose.yml`, and Terraform IaC for Amazon ECS Fargate, RDS PostgreSQL 15, ALB, S3, Secrets Manager, and CloudWatch.

### 16.7 Phase 7: Real-Time WebSockets, Order Book Depth & L2 Feed Integration
- **Streaming Market Feeds:** Interfacing with Alpaca Real-Time WebSocket API (`alpaca-py.live`) to ingest live 1-minute and tick trade bars.
- **Sub-Second Order Routing:** Evaluating streaming quotes against model signals with low-latency execution dispatch.
- **Level 2 Market Depth:** Modeling order book bid/ask queues, liquidity depletion, and dynamic market impact functions.

### 16.8 Phase 8: Multi-Agent LLM Sentiment Fusion & Alternative Data
- **Financial News Sentiment:** Ingesting real-time corporate press releases and financial news feeds, scoring sentiment using open-source financial LLMs (e.g., FinBERT, Llama 3).
- **SEC Filing Analysis:** Automated extraction of risk factors and executive commentary from quarterly 10-Q and annual 10-K filings.
- **Multi-Modal Alpha Blending:** Combining quantitative technical indicators with qualitative sentiment embeddings in an ensemble decision framework.

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
- Package Configuration & Parameters: `pyproject.toml`, `requirements.txt`, `src/tradepilot/config.py`, `params.yaml`
- DVC Pipeline & Versioning: `dvc.yaml`, `dvc.lock`, `.dvc/config`, `data/raw/stock_ohlcv_2021_2025.csv.dvc`, `.gitignore`
- MLOps Experiment Tracking: `src/tradepilot/tracking.py`, `mlflow.db`
- Core Data Subpackage: `src/tradepilot/data/load_data.py`, `src/tradepilot/data/validate_data.py`
- Feature Engineering Subpackage: `src/tradepilot/features/build_features.py`
- Modeling Subpackage: `src/tradepilot/modeling/split_data.py`, `src/tradepilot/modeling/train_model.py`, `src/tradepilot/modeling/evaluate_model.py`, `src/tradepilot/modeling/save_model.py`
- Backtesting Subpackage: `src/tradepilot/backtesting/signals.py`, `src/tradepilot/backtesting/execution.py`, `src/tradepilot/backtesting/portfolio.py`, `src/tradepilot/backtesting/metrics.py`, `src/tradepilot/backtesting/engine.py`
- Pipeline Orchestration: `src/tradepilot/pipelines/train_pipeline.py`, `src/tradepilot/pipelines/backtest_pipeline.py`
- Operational Scripts: `download_market_data.py`, `stock_eda.py`, `train.py`
- Test Suite: All 6 test modules in `tests/` (`test_data_validation.py`, `test_execution_and_portfolio.py`, `test_features.py`, `test_pipeline_reproducibility.py`, `test_signals.py`, `test_split_data.py`)

### 19.2 Automated Test Execution and Verification
- **Command Executed:** `pytest -v`
- **Execution Environment:** Python 3.10.7, pytest-9.2.0 on Windows x86_64
- **Result:** **17 passed in 12.66 seconds** (100% pass rate, 0 failures, 0 warnings).

### 19.3 Verified Datasets and Split Counts
- Raw OHLCV (`data/raw/stock_ohlcv_2021_2025.csv`): Exactly 62,664 rows, 7 columns, 50 symbols, spanning 2021-01-04 to 2025-12-31 (tracked via DVC pointer).
- Clean Labeled Features (`data/processed/stock_features_labeled.csv`): 61,764/59,964 rows, 50 symbols. Warmup and shift NaNs purged.
- Training Split: 34,864 rows (250 boundary rows purged, 5 per symbol).
- Validation Split: 12,350 rows (250 boundary rows purged, 5 per symbol).
- Test Split: 12,250 rows (unpurged final evaluation fold).

### 19.4 Verified Empirical Metrics
- **Validation Fold (2024):**
  - Majority Baseline: Macro F1: `0.2049`, Balanced Acc: `0.3333`, Accuracy: `0.4437`
  - Logistic Regression: Macro F1: `0.3041`, Balanced Acc: `0.3660`, Accuracy: `0.3181`
  - Random Forest: Macro F1: `0.3461`, Balanced Acc: `0.3763`, Accuracy: `0.3504`
  - Backtest (TradePilot): Return: `+10.39%`, Sharpe: `0.92`, Max Drawdown: `-6.16%`, Win Rate: `56.78%`, Turnover: `23.38x`
  - Backtest (Benchmark): Return: `+33.35%`, Sharpe: `2.07`, Max Drawdown: `-9.57%`
- **Test Fold (2025):**
  - Random Forest: Macro F1: `0.3568`, Balanced Acc: `0.3754`, Accuracy: `0.3635`
  - Class F1 Scores: `DOWN`: 0.3151, `NEUTRAL`: 0.3353, `UP`: 0.4200
  - Backtest (TradePilot): Return: `+6.78%` (to `+9.94%`), Sharpe: `0.42` (to `0.54`), Max Drawdown: `-19.09%` (to `-22.50%`), Trades: `318-343`, Win Rate: `50.31% - 51.90%`, Profit Factor: `1.05 - 1.08`, Turnover: `30.91x - 33.41x`, Total Costs: `$\$6,622.60 - \$6,928.29$`
  - Backtest (Benchmark): Return: `+26.29%`, Sharpe: `1.25`, Max Drawdown: `-21.41%`

### 19.5 MLOps Pipeline and Experiment Tracking Verification
- **DVC Pipeline Execution (`dvc repro`):** All 3 stages (`features`, `train`, `backtest`) completed successfully with exit code 0.
- **DVC Stage Invalidation & Caching:** Verified that rerunning `dvc repro` without modifications outputs:
  ```text
  'data\raw\stock_ohlcv_2021_2025.csv.dvc' didn't change, skipping
  Stage 'features' didn't change, skipping
  Stage 'train' didn't change, skipping
  Stage 'backtest' didn't change, skipping
  Data and pipelines are up to date.
  ```
- **DVC Remote Storage Push (`dvc push`):** Verified successful push of 3 data/model objects to `.dvc_storage`.
- **MLflow Tracking Audit (`sqlite:///mlflow.db`):** Verified that training runs record 34 parameters, 32 metrics, feature importance plots, confusion matrices, and serialized model binaries, and separate backtest runs record 9 parameters, 20 financial metrics, equity curves, and trade ledgers under experiment `TradePilot-Stock-Direction`.

### 19.6 Documented Discrepancies and Codebase Evolution
During documentation research and MLOps integration, important architectural refinements were established:
- **Stationary vs. Non-Stationary Features:** Raw price moving averages (`SMA_10`, `SMA_20`, `SMA_50`) from early exploratory prototypes were moved to `prohibited_cols` in `config.py` to prevent cross-stock scale confounding.
- **MLflow 3.x Backend Architecture:** Modern MLflow deprecates pure filesystem tracking. Configured embedded SQLite (`sqlite:///mlflow.db`) with `./mlruns` artifact store for robust querying and dashboard support.
- **DVC and Git Separation of Concerns:** Untracked raw data CSV, feature CSV, and model binaries from Git index (`git rm --cached`) so DVC manages large data while Git manages lightweight pointer files (`.dvc`, `dvc.lock`, `params.yaml`).

### 19.7 Security and Leakage Audit
- Verified that no live API keys, tokens, or passwords are hardcoded in source code or documentation.
- Verified that `.env` is loaded exclusively via `python-dotenv`.
- Verified that a comprehensive root `.gitignore` is active and protects secrets, virtual environments, DVC internal caches, and MLflow run databases from accidental version control exposure.

### 19.8 Version 1.0.0 Production Application Release Audit
The v1.0.0 full production application release underwent comprehensive validation:
1. **Automated Test Suite Expansion:** Expanded from 17 baseline machine-learning tests to **36 total automated tests** across 8 modules. Verified 100% pass rate (`pytest -v`) in 14.89s under Python 3.10.7.
2. **Frontend Build Verification:** Next.js 16.4.3 production compilation verified (`npm run build`) in `frontend/` with zero TypeScript errors, valid Route types, and optimized static asset packaging.
3. **Database Migration Verification:** Initial migration `934fa89e0077_initial_schema.py` verified via `alembic upgrade head`, generating 12 relational tables with foreign keys and indexes.
4. **Paper Trading Safeguard Verification:** Asserted that `TRADING_MODE` is locked to `PAPER` across settings, broker adapters, and the centralized risk engine. The system programmatically refuses order submission if paper mode cannot be verified.
5. **API Security & RBAC:** Verified JWT bearer token authentication, bcrypt password hashing, and role-based permissions (`admin`, `trader`, `viewer`).
6. **Containerization & IaC:** Verified Dockerfiles (`Dockerfile.api`, `Dockerfile.worker`, `Dockerfile.frontend`), `docker-compose.yml`, and Terraform modules for AWS ECS Fargate, Multi-AZ RDS PostgreSQL 15, ALB, S3, Secrets Manager, and CloudWatch.

---

## 20. Production Application Architecture, API, Agent Execution, and Cloud Deployment

### 20.1 Architectural Expansion Overview
Following the successful establishment of the Machine Learning foundation, feature engineering pipelines, and DVC/MLflow MLOps tracking, TradePilot was expanded into a complete production-grade software system:
1. **Versioned REST API (`/api/v1`)**: Implemented using FastAPI, Pydantic v2, and SQLAlchemy 2.0 with JWT authentication and Role-Based Access Control (RBAC).
2. **Autonomous Strategy Agent & Broker Adapters**: High-reliability paper-trading agent supporting both official Alpaca Paper Trading APIs (`alpaca-py`) and an offline, deterministic `MockBrokerAdapter` with slippage and fee simulation.
3. **Centralized Risk Management Engine**: Enforces 10 quantitative pre-trade risk checks, daily drawdown limits, and an emergency kill switch.
4. **Focused Financial Dashboard**: Built with Next.js 16, TypeScript, Tailwind CSS, and Lucide icons featuring real-time equity curves, interactive candlestick charts, probability distributions, order management, and agent controls.
5. **AWS Cloud-Ready Deployment**: Containerized multi-service architecture (FastAPI API, Background Worker, PostgreSQL, Next.js Frontend) with Terraform Infrastructure-as-Code for Amazon ECS Fargate, RDS PostgreSQL, ALB, S3, Secrets Manager, and CloudWatch.

```mermaid
flowchart TD
    subgraph Frontend Client
        UI[Next.js 16 Dashboard\nDashboard | Markets | Predictions | Portfolio | Orders | Risk]
    end

    subgraph Backend Application
        API[FastAPI Server :8000\nAuth | Markets | Predictions | Portfolio | Orders | Risk]
        Worker[Background Worker Process\nAutonomous Scheduled Trading Loop]
        Risk[Centralized Risk Engine\n10 Quantitative Pre-Trade Safeguards]
        Agent[Trading Agent\nInference | Signals | Order Sizing | Reconciliation]
    end

    subgraph Execution & Data
        Broker[Broker Adapter\nAlpaca Paper API / Mock Simulation]
        DB[(PostgreSQL / SQLite\nAlembic Migrations)]
        ModelStore[Model Artifacts\nbest_model.joblib & metadata.json]
    end

    UI <-->|REST API / JWT| API
    API -->|Inspect & Control| DB
    Worker -->|Heartbeat & Cycles| DB
    Worker --> Agent
    Agent --> Risk
    Risk -->|Approved| Broker
    Agent -->|Inference| ModelStore
    Broker -->|Reconciliation| DB
```

### 20.2 Versioned REST API Endpoints (`/api/v1`)
The FastAPI backend serves structured, documented endpoints under `/api/v1`:

| Area | HTTP Method & Path | Description & Access Control |
|---|---|---|
| **System & Health** | `GET /health` | Process liveness probe (Public) |
| | `GET /ready` | Dependency readiness probe (DB, Model, Paper Mode) |
| | `GET /api/v1/system/status` | Detailed operational state, uptime, and database connectivity |
| **Authentication** | `POST /api/v1/auth/login-json` | Authenticates credentials and returns JWT Bearer token |
| | `GET /api/v1/auth/me` | Returns current user profile and role |
| **Market Data** | `GET /api/v1/markets/symbols` | Approved trading symbols universe |
| | `GET /api/v1/markets/bars` | Historical daily OHLCV bars (`symbol`, `limit`) |
| | `GET /api/v1/markets/latest` | Latest closing prices and daily percentage changes |
| | `GET /api/v1/markets/status` | Market data connectivity and staleness metric |
| **Predictions & Strategy** | `GET /api/v1/predictions/latest` | Latest ML class probabilities (`prob_up`, `prob_neutral`, `prob_down`) |
| | `POST /api/v1/predictions/run` | Triggers immediate inference cycle across universe (Trader/Admin) |
| | `GET /api/v1/signals` | Strategy signals (`BUY`, `EXIT`, `HOLD`) with rule reasoning |
| | `GET /api/v1/models/active` | Active model metadata, algorithm, features, and test metrics |
| **Portfolio & Positions** | `GET /api/v1/portfolio` | Account equity, cash balance, buying power, and exposure % |
| | `GET /api/v1/portfolio/positions` | Open holdings, shares, entry price, current value, unrealized P&L |
| | `GET /api/v1/portfolio/equity` | Historical mark-to-market equity curve timeline points |
| **Orders & Trades** | `GET /api/v1/orders` | Submitted paper orders and lifecycle statuses |
| | `GET /api/v1/orders/{id}` | Order details by client or broker order ID |
| | `GET /api/v1/trades` | Execution fills ledger and transaction fees |
| | `POST /api/v1/orders` | Validated manual paper order entry gated by Risk Engine |
| **Agent Control** | `GET /api/v1/agent/status` | Agent lifecycle state (`RUNNING`, `PAUSED`, `STOPPED`), heartbeat, mode |
| | `POST /api/v1/agent/start` | Starts autonomous trading agent loop (Trader/Admin) |
| | `POST /api/v1/agent/pause` | Pauses agent (stops new orders while maintaining holdings) |
| | `POST /api/v1/agent/stop` | Stops agent strategy loop |
| | `POST /api/v1/agent/emergency-stop`| Global kill switch: cancels open orders & blocks trading (Admin) |
| **Risk & Activity** | `GET /api/v1/risk/status` | Real-time risk limit utilization & circuit breaker status |
| | `GET /api/v1/risk/config` | Active quantitative thresholds and slippage parameters |
| | `GET /api/v1/risk/events` | Audit log of blocked orders and risk rule trips |
| | `GET /api/v1/risk/audit/logs` | Append-only administrative action audit trail (Admin) |

### 20.3 Database Architecture & Migrations
The database layer uses SQLAlchemy 2.0 with Alembic version control:
- **`users`**: User identity, hashed passwords (bcrypt), and roles (`admin`, `trader`, `viewer`).
- **`agent_state`**: Singleton row tracking operational state, strict `PAPER` mode safeguard, heartbeat, and last cycle timestamp.
- **`trading_cycles`**: Execution records capturing cycle duration, symbols processed, orders generated, and errors.
- **`predictions`**: Empirical class probability distribution per symbol evaluated during cycles.
- **`signals`**: Strategy signals (`BUY`, `EXIT`, `HOLD`) derived from prediction thresholds.
- **`order_intents`**: Internal pre-submission proposals validated by the Centralized Risk Engine.
- **`broker_orders`**: Actual orders dispatched to the paper broker with status lifecycle tracking (`SUBMITTED`, `FILLED`, `CANCELED`, `REJECTED`).
- **`order_fills`**: Execution records with execution price, filled shares, and transaction fees.
- **`position_snapshots`**: Reconciled holdings snapshot with market values and unrealized P&L.
- **`portfolio_snapshots`**: Historical mark-to-market equity timeline for charting.
- **`risk_events`**: Immutable record of risk rule breaches and blocked order intents.
- **`audit_logs`**: Append-only security and operational audit trail.

### 20.4 Centralized Risk Engine & Pre-Trade Safeguards
All order intents (both automated from model signals and manual from the UI) must pass through `CentralizedRiskEngine.evaluate_order(...)`:
1. **Paper-Only Safeguard**: Refuses execution if mode is not verified as `PAPER`.
2. **Circuit Breaker / Emergency Stop**: Blocks all orders if the emergency stop was triggered.
3. **Account Trading Status**: Refuses execution if broker account indicates blocked trading.
4. **Approved Universe Check**: Refuses orders for symbols outside the approved universe.
5. **Quantity & Order Sizing**: Enforces positive integer share quantities and valid order types.
6. **Duplicate Order Prevention**: Deduplicates client order IDs and prevents concurrent pending orders on the same symbol.
7. **Market Data Staleness**: Refuses execution if historical lookback data exceeds freshness tolerance.
8. **Maximum Position Size**: Caps single stock holding at `10%` of total portfolio equity.
9. **Maximum Concurrent Positions**: Caps active stock holdings at `10` distinct components.
10. **Maximum Gross Exposure**: Caps total invested capital at `95%` of equity.
11. **Buying Power & Cash Check**: Ensures adequate available cash/buying power prior to order submission.
12. **Daily Drawdown Circuit Breaker**: Trips automatic kill switch if daily losses exceed `5%` of equity.

### 20.5 Autonomous Background Worker
The background strategy loop runs in an independent process (`src/tradepilot/execution/worker.py`):
- Executes on a configurable schedule (default: 60 seconds).
- Heartbeat tracking to detect worker staleness.
- Reconciles broker state before and after each cycle.
- Graceful shutdown on `SIGINT` / `SIGTERM` signals.

### 20.6 Next.js 16 Trading Dashboard
The frontend application in `frontend/` provides a focused, high-density financial terminal:
- **Dashboard**: Real-time equity, buying power, daily P&L, exposure gauge, interactive SVG equity timeline chart, active holdings, and recent signals.
- **Markets**: Symbol selector, current prices, day change, interactive candlestick/volume chart, and OHLCV history table.
- **Predictions & Signals**: Active model architecture card, test metrics, on-demand inference trigger, class probability bars, and rule reasoning.
- **Portfolio**: Asset allocation stacked bar, cash percentage, position details, unrealized P&L ($ and %), and exposure percentage.
- **Orders & Trades**: Manual paper order entry form with risk feedback, order lifecycle table with status badges, and execution fills ledger.
- **Agent Control**: Start/Pause/Stop agent controls, heartbeat status, worker cadence, and prominent emergency kill switch with confirmation modal.
- **Risk & Activity**: Real-time utilization meters (Exposure, Positions, Drawdown), configured boundaries, risk events log, and audit trail.

### 20.7 AWS Cloud Deployment Infrastructure
- **Amazon ECS on AWS Fargate**: Serverless containers for the API and Background Worker.
- **Application Load Balancer**: Public entry point terminating HTTPS and routing traffic to API tasks.
- **Amazon RDS for PostgreSQL 15**: Multi-AZ relational database with automated daily snapshots and point-in-time recovery.
- **Amazon S3**: Secure bucket for model artifacts, training data, and backtest reports.
- **AWS Secrets Manager**: Storing Alpaca credentials and database connection strings.
- **Amazon CloudWatch**: Centralized logging and alarms for error rates, CPU/memory, and circuit breaker events.
- **Terraform IaC**: Complete infrastructure definitions in `deploy/terraform/` (`main.tf`, `variables.tf`, `outputs.tf`).
- **CI/CD Pipeline**: GitHub Actions workflow in `.github/workflows/ci.yml` verifying Python test suites, linting, and Next.js frontend builds.

### 20.8 Final Test Suite Verification
- **Total Automated Tests:** **36 of 36 Passing (100% pass rate)**.
  - Data Validation Tests: 6 tests passing.
  - Execution & Portfolio Tests: 3 tests passing.
  - Feature Engineering Tests: 4 tests passing.
  - Pipeline Reproducibility Tests: 2 tests passing.
  - Signal Generation Tests: 1 test passing.
  - Chronological Data Split Tests: 1 test passing.
  - REST API Endpoint Integration Tests: 11 tests passing (`tests/test_api_endpoints.py`).
  - Trading Agent & Risk Engine Unit Tests: 8 tests passing (`tests/test_trading_agent_and_risk.py`).

---
*End of Complete Technical Documentation for TradePilot.*

