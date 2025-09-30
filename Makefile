venv:
	conda activate alphagomoku

train:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
 		--epochs 100 \
 		--selfplay-games 128 \
 		--mcts-simulations 600 \
 		--batch-size 512 \
 		--lr 1e-3 \
 		--min-lr 5e-4 \
 		--warmup-epochs 8 \
 		--lr-schedule cosine \
 		--map-size-gb 12 \
 		--buffer-max-size 500000 \
 		--batch-size-mcts 64 \
 		--parallel-workers 8 \
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
