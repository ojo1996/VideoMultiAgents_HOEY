# ReasoningVectors – Merge + Eval Pipeline

This repo lets you merge RL/SFT task vectors into a base model (e.g. Qwen2.5-3B) and evaluate the merged models on **AIME25** using `lm-evaluation-harness`.

---

```powershell
# setup a venv and activate
python -m venv .venv
.\.venv\Scripts\Activate

# install dependencies
pip install -r requirements.txt

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .

# download models 
mkdir models

huggingface-cli download Qwen/Qwen2.5-3B `
  --local-dir models\AFM-MHQA-Agent-3B-sft `
  --local-dir-use-symlinks False

huggingface-cli download PersonalAILab/AFM-MHQA-Agent-3B-rl `
  --local-dir models\AFM-MHQA-Agent-3B-rl `
  --local-dir-use-symlinks False
huggingface-cli download PersonalAILab/AFM-MHQA-Agent-3B-sft `
  --local-dir models\AFM-MHQA-Agent-3B-sft `
  --local-dir-use-symlinks False

huggingface-cli download PersonalAILab/AFM-MHQA-Agent-3B-rl `
  --local-dir models\AFM-MHQA-Agent-3B-rl `
  --local-dir-use-symlinks False

# Generate the resoning vector and apply it with various alphas
python generate_merged_models.py `
  --base_model Qwen/Qwen2.5-3B `
  --rl_model   models/AFM-MHQA-Agent-3B-rl `
  --sft_model  models/AFM-MHQA-Agent-3B-sft `
  --out_root   merges

# evaluate the models saving the results and samples.
python eval.py
