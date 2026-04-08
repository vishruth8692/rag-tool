CONFIG    ?= configs/base.yaml
DATASET   ?= eval/datasets/wikipedia_eval_sample.jsonl
CONFIG_A  ?= configs/rag_a.yaml
CONFIG_B  ?= configs/rag_b.yaml
HOST      ?= 0.0.0.0
PORT      ?= 8000

.PHONY: ingest eval ab-test serve dashboard test all

## Build the vector index from data/
ingest:
	python3 scripts/ingest.py --config $(CONFIG)

## Run evaluation (default: wikipedia_eval_sample, 80 queries)
## Override: make eval DATASET=eval/datasets/wikipedia_eval_hard.jsonl
eval:
	python3 scripts/run_eval.py --config $(CONFIG) --dataset $(DATASET)

## Run A/B test comparing two configs
## Override: make ab-test CONFIG_A=configs/rag_a.yaml CONFIG_B=configs/rag_b.yaml
ab-test:
	python3 scripts/run_ab.py --config-a $(CONFIG_A) --config-b $(CONFIG_B) --dataset $(DATASET)

## Start the REST API server
serve:
	python3 scripts/serve_api.py --config $(CONFIG) --host $(HOST) --port $(PORT)

## Launch the Streamlit evaluation dashboard
dashboard:
	streamlit run src/dashboard/app.py

## Run all unit tests
test:
	python3 -m pytest -q

## Full pipeline: ingest → eval → ab-test
all: ingest eval ab-test
