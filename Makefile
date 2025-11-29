venv:
	conda activate alphagomoku

train:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
 		--epochs 200 \
 		--selfplay-games 200 \
 		--mcts-simulations 150 \
 		--batch-size 512 \
 		--lr 1e-3 \
 		--min-lr 5e-4 \
 		--warmup-epochs 0 \
 		--lr-schedule cosine \
 		--map-size-gb 12 \
 		--buffer-max-size 500000 \
 		--batch-size-mcts 64 \
 		--parallel-workers 4 \
 		--difficulty medium \
		--debug-memory \
 		--resume auto

test-tactical-latest:
	@latest=$$(ls -t checkpoints/model_epoch_*.pt | head -1); \
	echo "Testing tactical awareness: $$latest"; \
	python scripts/test_tactical.py $$latest

test:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	MKL_THREADING_LAYER=SEQUENTIAL \
	KMP_DUPLICATE_LIB_OK=TRUE \
	KMP_AFFINITY=disabled \
	KMP_INIT_AT_FORK=FALSE \
	PYTORCH_ENABLE_MPS_FALLBACK=1 \
	pytest tests/unit -v
