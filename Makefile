venv:
	conda activate alphagomoku
train:
	export PYTORCH_ENABLE_MPS_FALLBACK=0
	export OMP_NUM_THREADS=1
	python scripts/train.py \
 		--epochs 5 \
 		--selfplay-games 200 \
 		--mcts-simulations 800 \
 		--batch-size 1024 \
 		--lr 0.001 \
 		--map-size-gb 32 \
 		--buffer-max-size 400000 \
 		--batch-size-mcts 256 \
 		--parallel-workers 8 \
 		--adaptive-sims \
 		--difficulty medium \
 		--resume auto

train-fast:
	export PYTORCH_ENABLE_MPS_FALLBACK=0
	export OMP_NUM_THREADS=1
	python scripts/train.py \
 		--epochs 50 \
 		--selfplay-games 100 \
 		--mcts-simulations 400 \
 		--batch-size 1024 \
 		--lr 0.01 \
 		--map-size-gb 16 \
 		--buffer-max-size 300000 \
 		--batch-size-mcts 256 \
 		--parallel-workers 8 \
 		--adaptive-sims \
 		--difficulty medium \
 		--resume auto
