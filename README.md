# Lightweight LitGPT Finetuning Setup for Qwen2.5-7B (SFT + DPO)

## Overview
This setup uses LitGPT to finetune Qwen2.5-7B on your trader SFT (`trader_sft_data.jsonl`) and DPO (`trader_dpo_data.jsonl`) data. It's optimized for a single rented GPU (e.g., A100/H100) with QLoRA (4-bit + LoRA adapters) for ~12-16GB VRAM usage.

- **SFT**: Supervised finetuning on chat responses.
- **DPO**: Direct Preference Optimization for alignment (experimental in LitGPT for Qwen; falls back to SFT if issues).
- **Efficiency**: 1-3 hours per run on A100; total ~14GB model + data.

Data is already compatible:
- SFT: OpenAI chat format (system/user/assistant) → Uses `litgpt.data.JSON` with `chatml`.
- DPO: Standard (prompt/chosen/rejected) → Uses `litgpt.data.DPO`.

## Prerequisites
- Python 3.10+ with CUDA 12+ (for GPU).
- ~20GB disk space (model + outputs).
- Hugging Face account/token if model is gated (set `HF_TOKEN=your_token`).

## Installation
1. Clone LitGPT (if not done):
   ```
   git clone https://github.com/Lightning-AI/litgpt.git
   cd litgpt && pip install -e '.[all]' && cd ..
   ```

2. Fix any dep issues (e.g., dateutil):
   ```
   pip install --upgrade python-dateutil
   ```

3. Download base model (~14GB):
   ```
   mkdir -p checkpoints
   litgpt download Qwen/Qwen2.5-7B --out checkpoints/Qwen2.5-7B
   ```

## Data Preparation
No changes needed—files are in root:
- `trader_sft_data.jsonl`: Chat conversations (e.g., {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}).
- `trader_dpo_data.jsonl`: Preferences (e.g., {"prompt": "...", "chosen": "...", "rejected": "..."}).

LitGPT auto-splits 5% for validation, masks prompts, and handles Qwen's ChatML style.

## Finetuning

### 1. SFT (Supervised Finetuning)
Run from workspace root:
```
litgpt finetune --config finetune_qwen_sft.yaml
```
- **Params**: QLoRA (rank 16), batch 4, seq 2048, 1 epoch (~1000 steps).
- **Time/VRAM**: 1-2 hours / 12-16GB on A100.
- **Output**: `out/qwen-sft/` (checkpoints, logs.csv with loss/perplexity).
- **Monitor**: `tail -f out/qwen-sft/logs.csv` for training progress.

If OOM: Reduce `micro_batch_size` to 2 or `max_seq_length` to 1024 in YAML.

### 2. DPO (Preference Alignment)
Load SFT model and align with preferences:
```
litgpt finetune --config dpo_qwen.yaml
```
- **Params**: Similar to SFT, beta=0.1, 500 steps (shorter for alignment).
- **Time/VRAM**: 30-60 min / 12-16GB.
- **Output**: `out/qwen-dpo/`.
- **Note**: DPO in LitGPT is OLMo-focused; for Qwen, it uses adapted trainer. If errors (e.g., prompt mismatch), run SFT-only—it's a strong baseline. Custom script available if needed.

### Adjust Hyperparams
Edit YAMLs:
- More data/epochs: Increase `max_steps` or `epochs`.
- Lower VRAM: Set `quantize: bnb.fp4` or `lora_r: 8`.
- Multi-GPU: `--devices 2`.

## Testing the Model
Chat interactively:
```
litgpt chat out/qwen-sft/final --max_new_tokens 256  # Or out/qwen-dpo/final
```
- Input trader scenarios; it should respond in Chimera style.

Eval perplexity (on val set):
```
litgpt evaluate out/qwen-sft/final --tasks perplexity
```

## Deployment
Serve as API (e.g., on rented GPU):
```
litgpt serve out/qwen-sft/final --host 0.0.0.0 --port 8000
```
Query:
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"prompt": "Junior trader query..."}'
```

Export to HF:
```
litgpt export out/qwen-sft/final --out_dir hf-qwen-sft
```
Push: `git lfs install && git add . && git commit -m "Finetuned Qwen" && git push`.

## GPU Instructions (Rented Instance)
1. **Setup (e.g., RunPod/Lambda)**: Launch A100/H100 instance (40GB+ VRAM), attach volume, clone this repo.
2. **Env**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` (for CUDA 12.1).
3. **Run**: As above. Use `nvidia-smi` to monitor VRAM.
4. **Costs**: ~$0.50-1/hr on A100; full run <$5.
5. **Troubleshoot**:
   - OOM: Add `--gradient_checkpointing` to CLI or edit YAML (`gradient_checkpointing: true`).
   - DPO Errors: Skip to SFT; contact for custom `dpo_train.py`.
   - Logs: Check `out/*/lightning_logs/` for details.

## Next Steps
- Run SFT, monitor loss (<2.0 target).
- If DPO needed, test after SFT.
- For custom DPO trainer or more epochs, reply with results.

All files created: `finetune_qwen_sft.yaml`, `dpo_qwen.yaml`, this README.md. Ready to upload/train!
