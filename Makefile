venv:
	conda activate alphagomoku

train:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
 		--epochs 100 \
 		--selfplay-games 64 \
 		--mcts-simulations 128 \
 		--batch-size 512 \
 		--lr 0.001 \
 		--min-lr 1e-4 \
 		--warmup-epochs 5 \
 		--lr-schedule cosine \
 		--map-size-gb 12 \
 		--buffer-max-size 500000 \
 		--batch-size-mcts 64 \
 		--parallel-workers 8 \
 		--adaptive-sims \
 		--difficulty medium \
 		--resume auto

test:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	MKL_THREADING_LAYER=SEQUENTIAL \
	KMP_DUPLICATE_LIB_OK=TRUE \
	KMP_AFFINITY=disabled \
	KMP_INIT_AT_FORK=FALSE \
	PYTORCH_ENABLE_MPS_FALLBACK=1 \
	pytest tests/unit -v
