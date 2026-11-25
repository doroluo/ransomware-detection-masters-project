HYPERPARAMETER PERFORMANCE FOR YANPING
Prepared November 24, 2026. No model training or source-workbook edits performed.

Start with these five figures (PNG for slides; SVG for vector editing):
01 Overall held-out performance, original primary vs attention follow-up.
02 Original learning-rate, shared-dropout and weight-decay grids.
03 Original architecture sizes, layers, heads, instruction limits and batch sizes.
04 Attention heads/dropout effects and Model 3 head-count interactions.
05 Overall test performance across block counts and signal/callwalk sampling.

MAIN FINDINGS

1. Model 3 is sensitive to learning rate.
   In the original short optimizer screens, marginal validation ROC-AUC peaked
   near LR=0.0001 for M3 mean and LR=0.0003 for M3 CLS. LR=0.003 was much worse.
   M1/M2 marginal validation AUC peaked at LR=0.001. These are stage averages,
   not recommendations to overwrite the saved selected configurations.

2. Heavy shared dropout generally hurt performance.
   M2/M3 favored lower shared dropout on average. M1 had a small benefit at 0.1.
   This original setting controlled multiple dropout locations together. It is
   different from the independent attention dropout used in the later study.

3. Weight decay had a smaller effect in the original optimizer stage.
   The range of marginal validation AUC across weight-decay settings was less
   than 0.008 for every model/readout. That does not establish zero effect in
   every individual configuration or a universal best decay value.

4. Wider/deeper was not uniformly better.
   The original architecture screens generally favored moderate widths and
   shallow networks on average. M1 benefited from a larger instruction budget;
   M3's marginal validation AUC peaked at 64 instructions per block in this stage.
   Panel x-axis values are real tested settings. M1 and M3 used different layer
   and batch ranges, shown on a common union of tested levels.

5. Additional attention heads gave small, configuration-dependent effects.
   M1 favored four Transformer heads among the three fully completed attention
   grids. M2 and Model 3 readouts showed different preferences. Model 3's joint
   Transformer-head / GAT-head / dropout heatmaps reveal interactions that a
   single Pearson correlation would hide.

6. More blocks did not guarantee greater overall accuracy.
   Among complete standard local block-budget comparisons, M2 / signal750
   achieved 85.80% accuracy. Increasing that budget to 1000 did not improve it.
   M1 and M2 responded differently to signal versus callwalk sampling.

7. Held-out attention-study improvements were modest and mixed.
   Overall accuracy (original primary -> attention study):
     M1       81.93% -> 82.25%
     M2       85.28% -> 84.38%
     M3 mean  84.16% -> 84.59%
     M3 CLS   82.69% -> 83.48%
   Detection and macro-F1 improved slightly for M1/M3; some FPR/AUC measures
   worsened. No model reached 90% overall accuracy in these evaluations.

HOW TO INTERPRET THE CHARTS

Figures 2/3: validation screens only, seed 0, up to 12 epochs. Each parameter
level first averages the other settings in that same search stage within each
split, then averages the five split means with equal weight. No stage is mixed
with another. Stages inherit selections made earlier, so their baselines differ.
All levels of a given within-stage curve have balanced candidate counts.

Figure 4 top: marginal validation AUC change relative to four heads, or zero
attention dropout, within a split. M1 uses only the three complete 12-candidate
grids (splits 0,1,4), so every displayed level has the same comparison set. Six
M1 candidates failed with CUDA OOM in splits 2/3. M2/M3 use all five splits.
Figure 4 bottom: each cell averages M3 mean/CLS scores within a split, subtracts
the T=4 / G=4 / attention-dropout=0 reference in that split, then averages five
split differences. Cell numbers are validation AUC percentage-point changes.
They are not accuracy gains, independent replications, or new held-out scores.

Figures 1/5: test metrics pool predictions from all five held-out splits for each
seed, then average three seed scores. Error bars are sample SD across seeds,
not confidence intervals or uncertainty over possible ransomware families.
FPR is better when lower. ROC-AUC and PR-AUC are different from accuracy.

Original versus attention is not a pure head/dropout comparison: the follow-up
fixes Transformer width to 96 and changes the GAT output-head policy. Within
the attention study, feature dropout and the earlier fold-specific recipe stay
fixed while the displayed attention parameters vary.

Figure 5 includes only M1/M2 because their local 250/500/750/1000 comparison
curves are complete. Do not invent missing Model 3 or Radeon-return results.
The controlled block-budget series uses a common training-only API vocabulary;
the original-primary signal500 series in Figure 1 is separately labeled.

All curves are descriptive. They do not establish statistical significance,
causation, deployment accuracy, or a single optimal combination assembled by
taking the peak from each stage. Y-axis ranges vary across panels, so compare
the numeric effect sizes rather than the apparent steepness alone.

REPRODUCIBILITY

Source: Final_Paper_Results_With_Hyperparameters_2026-11-24.xlsx.
The source SHA-256 and analysis coverage are in manifest.json.
data/ holds numeric plot values and descriptive peak summaries. The maintained
builder is CT_GAT/pipeline/reporting/final_export/build_hyperparameter_figures.py.
The builder requires NumPy, Matplotlib, the original workbook, its layout.json,
and the committed final_export/verify.py XML reader. No raw samples, model
checkpoints, executable binaries or training datasets are included.
