venv:
	conda activate alphagomoku

train:
	export PYTORCH_ENABLE_MPS_FALLBACK=0
	export OMP_NUM_THREADS=1
	python scripts/train.py \
 		--epochs 50 \
 		--selfplay-games 200 \
 		--mcts-simulations 200 \
 		--batch-size 512 \
 		--lr 0.003 \
 		--map-size-gb 16 \
 		--buffer-max-size 300000 \
 		--batch-size-mcts 256 \
 		--parallel-workers 8 \
 		--adaptive-sims \
 		--difficulty medium \
 		--lr-schedule cosine \
 		--warmup-epochs 2 \
 		--min-lr 0.0001 \
 		--resume auto
