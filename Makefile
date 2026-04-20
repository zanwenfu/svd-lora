# Repo-level entry points. Defers to project/scripts/*.sh for the heavy lifting.
#
# Usage:
#   make install         # pip install -r project/requirements.txt
#   make train-lora D=sst2
#   make train-hira D=sst2
#   make compress D=sst2 TAU=0.9 POST=2
#   make eval-lora D=sst2
#   make figures         # regenerate analysis plots from saved JSON + adapters

D ?= sst2
TAU ?= 0.9
POST ?= 2
PY ?= python

.PHONY: install train-lora train-hira compress eval-baseline eval-lora eval-hira \
        bench-lora bench-sparse-lora bench-hira figures clean-artifacts

install:
	$(PY) -m pip install -r project/requirements.txt

train-lora:
	./project/scripts/train_lora.sh $(D)

train-hira:
	./project/scripts/train_hira.sh $(D)

compress:
	./project/scripts/compress_svd.sh $(D) $(TAU) $(POST)

eval-baseline:
	./project/scripts/test_baseline_acc.sh $(D)

eval-lora:
	./project/scripts/test_lora_acc.sh $(D)

eval-hira:
	./project/scripts/test_hira_acc.sh $(D)

bench-lora:
	./project/scripts/test_lora_lat.sh $(D)

bench-sparse-lora:
	./project/scripts/test_sparse_lora_lat.sh $(D)

bench-hira:
	./project/scripts/test_hira_lat.sh $(D)

figures:
	cd project/src && $(PY) -m sparse_lora.analyze_sparse_lora_results

clean-artifacts:
	find project -type d -name '__pycache__' -prune -exec rm -rf {} +
	find project -type d -name 'checkpoint-*' -prune -exec rm -rf {} +
	find project -type d -name 'logs' -prune -exec rm -rf {} +
	find project -type d -name 'runs' -prune -exec rm -rf {} +
	find project -type f -name 'events.out.tfevents.*' -delete
