venv:
	conda activate alphagomoku
train:
	export PYTORCH_ENABLE_MPS_FALLBACK=0
	export OMP_NUM_THREADS=1
	python scripts/train.py \
 		--epochs 5 \
 		--selfplay-games 200 \
 		--batch-size 1024 \
 		--lr 0.01 \
 		--mcts-simulations 800 \
 		--map-size-gb 32 \
 		--buffer-max-size 400000 \
 		--batch-size-mcts 256 \
 		--parallel-workers 8 \
 		--adaptive-sims \
 		--resume auto

train-fast:
	export PYTORCH_ENABLE_MPS_FALLBACK=0
	export OMP_NUM_THREADS=1
	python scripts/train.py \
 		--epochs 5 \
 		--selfplay-games 100 \
 		--batch-size 1024 \
 		--lr 0.01 \
 		--mcts-simulations 400 \
 		--map-size-gb 8 \
 		--buffer-max-size 300000 \
 		--batch-size-mcts 256 \
 		--parallel-workers 8 \
 		--adaptive-sims \
 		--resume auto
