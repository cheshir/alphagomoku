# Cloud VM Recommendations for Training Strong Gomoku AI

**Goal**: Train the strongest possible model efficiently on rented cloud infrastructure.

---

## 🎯 Quick Recommendations

### **For Strong Player (Best Value)**
**Recommended: NVIDIA RTX 4090 or A6000**

```
GPU:     RTX 4090 (24GB VRAM) or A6000 (48GB VRAM)
CPU:     8-16 cores
RAM:     32 GB
Storage: 100 GB SSD
Cost:    ~$0.50-0.80/hour
Training: ~3-5 days for 200 epochs
```

### **For Maximum Strength (Premium)**
**Recommended: NVIDIA A100 40GB/80GB**

```
GPU:     A100 40GB or 80GB
CPU:     16-32 cores
RAM:     64 GB
Storage: 200 GB SSD
Cost:    ~$1.50-3.00/hour
Training: ~2-3 days for 200 epochs
```

### **Budget Option (Slower but Works)**
**Recommended: NVIDIA T4 or RTX 3080**

```
GPU:     T4 (16GB) or RTX 3080 (10GB)
CPU:     8 cores
RAM:     32 GB
Storage: 100 GB SSD
Cost:    ~$0.30-0.50/hour
Training: ~5-7 days for 200 epochs
```

---

## 📊 Detailed Cost-Benefit Analysis

### GPU Comparison

| GPU | VRAM | Rel. Speed | $/hour | 200 Epochs | Total Cost | Recommendation |
|-----|------|-----------|--------|------------|------------|----------------|
| **T4** | 16GB | 1.0x | $0.35 | 7 days | ~$60 | Budget |
| **RTX 3080** | 10GB | 1.5x | $0.50 | 5 days | ~$60 | Budget+ |
| **RTX 4090** | 24GB | 3.0x | $0.70 | 3 days | ~$50 | ⭐ **Best Value** |
| **A6000** | 48GB | 2.5x | $0.80 | 3.5 days | ~$67 | Good (more VRAM) |
| **A100 40GB** | 40GB | 4.0x | $1.50 | 2 days | ~$72 | Premium |
| **A100 80GB** | 80GB | 4.0x | $2.50 | 2 days | ~$120 | Overkill |

**Verdict: RTX 4090 is the sweet spot** - 3x faster than T4, best $/performance ratio.

---

## 🏆 Recommended Configuration for Strong Player

### **Configuration A: RTX 4090 (Best Choice)** ⭐

**Why:** Best price/performance, plenty of VRAM, widely available

**Specifications:**
```yaml
GPU:     NVIDIA RTX 4090 (24 GB VRAM)
CPU:     16 cores (AMD EPYC or Intel Xeon)
RAM:     32 GB
Storage: 100 GB SSD (NVMe preferred)
OS:      Ubuntu 22.04 LTS
```

**Training Configuration:**
```bash
python scripts/train.py \
    --model-preset medium \
    --parallel-workers 1 \
    --batch-size 1024 \
    --batch-size-mcts 128 \
    --selfplay-games 200 \
    --mcts-simulations 600 \
    --epochs 200 \
    --eval-frequency 10 \
    --device cuda
```

**Expected Performance:**
- Time per epoch: ~30-40 minutes
- Total training time: ~4-5 days
- Final Elo: 1800-1900+
- Total cost: ~$50-70

**Providers:**
- **Lambda Labs**: $0.70/hour (RTX 4090)
- **RunPod**: $0.69/hour (RTX 4090)
- **Vast.ai**: $0.60-0.80/hour (RTX 4090, varies)

---

### **Configuration B: A100 40GB (Premium)**

**Why:** Fastest training, most reliable, good for experimentation

**Specifications:**
```yaml
GPU:     NVIDIA A100 40GB
CPU:     16-32 cores
RAM:     64 GB
Storage: 200 GB SSD
OS:      Ubuntu 22.04 LTS
```

**Training Configuration:**
```bash
python scripts/train.py \
    --model-preset medium \
    --parallel-workers 1 \
    --batch-size 2048 \
    --batch-size-mcts 128 \
    --selfplay-games 250 \
    --mcts-simulations 800 \
    --epochs 200 \
    --eval-frequency 10 \
    --device cuda
```

**Expected Performance:**
- Time per epoch: ~20-30 minutes
- Total training time: ~2-3 days
- Final Elo: 1900-2000+
- Total cost: ~$70-110

**Providers:**
- **Google Cloud**: $2.48/hour (A100 40GB on-demand)
- **AWS**: $2.47/hour (p4d.24xlarge, has 8x A100)
- **Lambda Labs**: $1.29/hour (A100 40GB)
- **RunPod**: $1.49/hour (A100 40GB)

---

### **Configuration C: Budget (T4/3080)**

**Why:** Cheapest option, works well if not in a hurry

**Specifications:**
```yaml
GPU:     NVIDIA T4 (16GB) or RTX 3080 (10GB)
CPU:     8 cores
RAM:     32 GB
Storage: 100 GB SSD
OS:      Ubuntu 22.04 LTS
```

**Training Configuration:**
```bash
python scripts/train.py \
    --model-preset small \
    --parallel-workers 1 \
    --batch-size 512 \
    --batch-size-mcts 96 \
    --selfplay-games 150 \
    --mcts-simulations 400 \
    --epochs 200 \
    --eval-frequency 10 \
    --device cuda
```

**Expected Performance:**
- Time per epoch: ~60-90 minutes
- Total training time: ~5-7 days
- Final Elo: 1700-1800
- Total cost: ~$50-80

**Providers:**
- **Google Colab Pro**: $10/month (includes T4/V100)
- **Paperspace**: $0.45/hour (RTX 3080)
- **Vast.ai**: $0.30-0.50/hour (T4/3080)

---

## 🌐 Cloud Provider Comparison

### Lambda Labs (Recommended for ML)
**Pros:**
- ML-optimized instances
- Simple pricing, no hidden costs
- Fast setup
- Good for 24/7 training

**Cons:**
- Sometimes low availability
- Fewer regions

**Best for:** RTX 4090, A100 training

**Pricing:**
- RTX 4090: $0.70/hour
- A100 40GB: $1.29/hour

**Website:** https://lambdalabs.com/service/gpu-cloud

---

### RunPod (Best for Spot Instances)
**Pros:**
- Spot pricing (cheaper)
- Good availability
- Jupyter notebooks included
- SSH access

**Cons:**
- Spot instances can be interrupted
- UI less polished

**Best for:** RTX 4090, A100 (spot pricing)

**Pricing:**
- RTX 4090: $0.69/hour (on-demand), $0.39/hour (spot)
- A100 40GB: $1.49/hour (on-demand), $0.89/hour (spot)

**Website:** https://www.runpod.io/

---

### Vast.ai (Cheapest, Community Marketplace)
**Pros:**
- Cheapest prices (peer-to-peer)
- Many GPU options
- Good for experimentation

**Cons:**
- Variable reliability
- Need to check host quality
- Some hosts may disconnect

**Best for:** Budget training, experimentation

**Pricing:**
- RTX 4090: $0.60-0.80/hour
- A100 40GB: $1.00-1.50/hour

**Website:** https://vast.ai/

---

### Google Cloud / AWS / Azure (Enterprise)
**Pros:**
- Most reliable
- Global availability
- Good support
- Enterprise features

**Cons:**
- Most expensive
- Complex pricing
- Overkill for simple training

**Best for:** Production deployments, not personal training

**Pricing:**
- Google Cloud A100 40GB: $2.48/hour
- AWS p4d.24xlarge (A100): $32.77/hour (8 GPUs!)
- Azure NC A100 v4: $3.67/hour

---

## 💰 Cost Optimization Strategies

### Strategy 1: Use Spot/Preemptible Instances
**Savings:** 50-70%

```bash
# On RunPod, use spot instances
# Save checkpoints frequently (every epoch)
# Use --resume auto to continue if interrupted
```

### Strategy 2: Train Overnight/Weekends
**Savings:** Sometimes cheaper off-peak pricing

### Strategy 3: Use Smaller Model First
**Savings:** 30-50% on experimentation

```bash
# Train small model to validate pipeline
make train-fast --epochs 10

# Once validated, switch to medium
make train-production --epochs 200
```

### Strategy 4: Optimize Hyperparameters
**Savings:** Faster convergence = less time

```bash
# Use our optimized configs
python scripts/show_recommended_config.py --prefer-strength

# They're tuned for efficiency
```

---

## 🚀 Setup Guide for Cloud VM

### Step 1: Rent the VM

**For Lambda Labs:**
```bash
# 1. Go to https://lambdalabs.com/service/gpu-cloud
# 2. Sign up and add payment
# 3. Launch instance: RTX 4090, 32GB RAM, 100GB storage
# 4. SSH key setup
```

### Step 2: Initial Setup

```bash
# SSH into the instance
ssh ubuntu@<instance-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+
sudo apt install python3.11 python3.11-venv python3-pip git -y

# Clone repository
git clone <your-repo-url>
cd alphagomoku
```

### Step 3: Install Dependencies

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project
pip install -r requirements.txt
pip install -e .

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 4: Configure Training

```bash
# Check what's recommended for this VM
python scripts/show_recommended_config.py

# Copy the recommended command or use preset
make train-production
```

### Step 5: Monitor Training (Important!)

```bash
# Option 1: Use screen/tmux so you can disconnect
screen -S training
make train-production
# Ctrl+A, D to detach
# screen -r training to reattach

# Option 2: Use nohup
nohup make train-production > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Step 6: Download Results

```bash
# Compress checkpoints
tar -czf checkpoints.tar.gz checkpoints/

# Download to local machine (from local terminal)
scp ubuntu@<instance-ip>:~/alphagomoku/checkpoints.tar.gz .

# Or use rsync for resume capability
rsync -avz --progress ubuntu@<instance-ip>:~/alphagomoku/checkpoints/ ./checkpoints/
```

---

## 📈 Training Timeline Estimates

### With RTX 4090 (Recommended)

**Medium Model, 200 Epochs:**
```
Setup:              30 minutes
Epoch 1-50:         25 hours (30 min/epoch)
Epoch 51-100:       25 hours
Epoch 101-150:      25 hours
Epoch 151-200:      25 hours
Total:              ~4.2 days
Cost:               ~$70
```

**Breakdown:**
- Self-play: 80% of time
- Training: 15% of time
- Evaluation: 5% of time

### With A100 40GB (Premium)

**Medium Model, 200 Epochs:**
```
Setup:              30 minutes
Total training:     ~2.5 days (18 min/epoch)
Cost:               ~$90-110
```

### With T4 (Budget)

**Small Model, 200 Epochs:**
```
Total training:     ~6 days (45 min/epoch)
Cost:               ~$50-60
```

---

## ⚠️ Important Considerations

### 1. **Always Save Checkpoints**
```bash
# Training auto-saves every epoch to:
checkpoints/model_epoch_*.pt

# Download regularly!
```

### 2. **Monitor Costs**
```bash
# Set budget alerts on cloud platforms
# Expected costs for 200 epochs:
# - T4: $50-60
# - RTX 4090: $60-80
# - A100: $90-120
```

### 3. **Use Spot Instances Wisely**
```bash
# Only if you can handle interruptions
# Save frequently (auto-enabled)
# Use --resume auto
```

### 4. **Internet Costs**
```bash
# Uploading repo: ~500 MB
# Downloading checkpoints: ~100 MB per checkpoint
# Total egress: ~5 GB for full training
# Usually free or cheap (~$0.50)
```

---

## 🎯 Recommended Setup for You

Based on your goal of training a **strong player**:

### **Option 1: Best Value** ⭐
```
Provider:   Lambda Labs or RunPod
GPU:        RTX 4090 (24GB)
RAM:        32 GB
Time:       4-5 days
Cost:       ~$60-80
Model:      Medium (3M params)
Final Elo:  ~1850-1950
```

**Why:** Best price/performance ratio. Gets you 90% of maximum strength at 60% of the cost.

### **Option 2: Maximum Strength**
```
Provider:   Lambda Labs or RunPod
GPU:        A100 40GB
RAM:        64 GB
Time:       2-3 days
Cost:       ~$90-120
Model:      Medium (3M params) with extra sims
Final Elo:  ~1900-2000+
```

**Why:** Fastest, most reliable. Get results quickly, can experiment more.

---

## 📋 Pre-Launch Checklist

Before starting cloud training:

- [ ] Repository pushed to GitHub (private or public)
- [ ] Git credentials configured (or use HTTPS)
- [ ] SSH key for instance access
- [ ] Budget set aside (~$100 for safety)
- [ ] Alert set up for cost overruns
- [ ] Local backup of any existing work
- [ ] Understand how to SSH and use screen/tmux
- [ ] Know how to download results (scp/rsync)

---

## 🆘 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
--batch-size 512  # instead of 1024
--batch-size-mcts 96  # instead of 128
```

### Instance Disconnected
```bash
# Training should auto-resume
# If using screen:
screen -r training

# Check if process still running
ps aux | grep python
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Should see:
# - GPU util: 90-100%
# - Memory: 70-90% used
# If low, increase batch size
```

---

## 💡 Pro Tips

1. **Start Small**: Train for 10 epochs first to validate setup (~2-3 hours)
2. **Monitor Progress**: Download Elo history every 24 hours
3. **Budget Buffer**: Add 20% buffer to cost estimates
4. **Use Spot for Long Jobs**: 50% cheaper if you can handle interruptions
5. **Download Frequently**: Don't wait until end to download checkpoints

---

## 🎓 Summary

**For training a strong Gomoku player, I recommend:**

1. **Provider**: Lambda Labs or RunPod
2. **GPU**: RTX 4090 (24GB VRAM)
3. **RAM**: 32 GB
4. **Model**: Medium preset (3M params)
5. **Duration**: 4-5 days
6. **Cost**: ~$60-80

This gives you excellent price/performance and will produce a very strong player (Elo 1850-1950+).

**Next steps:**
1. Sign up at Lambda Labs or RunPod
2. Launch RTX 4090 instance
3. Follow setup guide above
4. Run: `make show-hardware-config` to verify
5. Start training: `make train-production`

Good luck! 🚀
