SWEEP:=python scripts/sweep_alpha.py
ALPHAS:=python tools/auto_make_alpha_settings.py
EVAL:=python eval.py --models "merges/alpha=*" --results_root results --config configs/eval.yaml --task ALL --device cuda:0 --alpha_json alpha_settings.json
AGG:=python tools/aggregate.py
PLOT:=python tools/plot_alpha.py
ANCHOR:=python tools/anchor_check.py --cfg configs/anchor_tol.yaml
PAPER:=python tools/paper_csv.py
TEST:=python tools/regression_test.py

all: sweep eval aggregate plot anchor paper
sweep: ; $(SWEEP)
eval:  ; $(ALPHAS) && $(EVAL) && $(ANCHOR)
aggregate: ; $(AGG)
plot: ; $(PLOT)
anchor: ; $(ANCHOR)
paper: ; $(PAPER)
test: ; $(TEST)

.PHONY: all sweep eval aggregate plot anchor paper test
