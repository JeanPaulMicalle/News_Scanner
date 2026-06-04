# New Scanner

This folder contains the new thesis-aligned scanner scaffold.

## Goal

Build a research-ready baseline pipeline before adding:

- dynamic retrieval
- persona-aware bias correction
- multimodal verification
- ablation and fairness extensions

## What This Scaffold Includes

- typed data models
- YAML configuration
- corpus loading from article JSON files
- chunking
- reusable embedding service
- FAISS-based retrieval
- structured verdict generation
- CLI entry points for indexing and checking a claim

## Quick Start

From the `New_Scanner` directory:

```powershell
..\venv\Scripts\python.exe -m new_scanner.cli index
..\venv\Scripts\python.exe -m new_scanner.cli check --text "Your claim here"
..\venv\Scripts\python.exe -m new_scanner.cli check --mode llm_only --text "Your claim here"
..\venv\Scripts\python.exe -m new_scanner.cli evaluate --dataset ..\test_dataset.csv --mode rag
..\venv\Scripts\python.exe -m new_scanner.cli evaluate --dataset ..\test_dataset.csv --compare-baselines
..\venv\Scripts\python.exe -m new_scanner.cli train-classifier --dataset-dir C:\path\to\FakeNewsNet\politifact
```

## Current Scope

This is the first milestone:

- one local JSON corpus
- one retriever
- one verifier
- one CLI

It is intentionally simple so it can become the stable base for the rest of the thesis work.

## DistilBERT Baseline

The project now also includes a supervised DistilBERT baseline for FakeNewsNet PolitiFact.

Expected dataset shape:

```text
politifact/
  fake/
    politifact123/
      news content.json
  real/
    politifact456/
      news content.json
```

Training example:

```powershell
..\venv\Scripts\python.exe -m new_scanner.cli train-classifier --dataset-dir C:\path\to\FakeNewsNet\politifact
```

Evaluation example:

```powershell
..\venv\Scripts\python.exe -m new_scanner.cli evaluate-classifier --model-dir C:\path\to\saved\model --dataset-dir C:\path\to\FakeNewsNet\politifact
```
