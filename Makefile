SWEEP:=python scripts/sweep_alpha.py
ALPHAS:=python tools/auto_make_alpha_settings.py
EVAL:=python eval.py --models "merges/alpha=*" --results_root results --config configs/eval.yaml --task ALL --device mps --alpha_json alpha_settings.json
AGG:=python tools/aggregate.py
PLOT:=python tools/plot_alpha.py
ANCHOR:=python tools/anchor_check.py --cfg configs/anchor_tol.yaml
PAPER:=python tools/paper_csv.py
TEST:=python tools/regression_test.py

all: sweep eval aggregate plot anchor paper
sweep: ; $(SWEEP)for a in 0.0 0.1 0.2 0.5 1.0; do
  python scripts/apply_vector_and_eval.py \
    --use_mergekit \
    --mk_base Qwen/Qwen2.5-3B \
    --mk_rl models/AFM-MHQA-Agent-3B-rl \
    --mk_sft models/AFM-MHQA-Agent-3B-sft \
    --alpha_task $a \
    --out_dir merges
done
eval:  ; $(ALPHAS) && $(EVAL) && $(ANCHOR)
aggregate: ; $(AGG)
plot: ; $(PLOT)
anchor: ; $(ANCHOR)
paper: ; $(PAPER)
test: ; $(TEST)

.PHONY: all sweep eval aggregate plot anchor paper test
