"""
Analyze a completed grid search: rank architecture/PCA-component combinations
and produce comparison plots across the whole grid.

Reads results/grid/all_summaries.csv and all_folds.csv, written by
scripts/train.py when run with MODE='grid' — run that first.
"""
import sys
from pathlib import Path
import pandas as pd
from src import plots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GRID_DIR = ROOT / 'results' / 'grid'
METRIC   = 'auc'        # metric driving ranking / leaderboard / complexity plots
TOP_N    = 5             # combinations shown in the leaderboard and boxplot

summary_df = pd.read_csv(GRID_DIR / 'all_summaries.csv')
folds_df   = pd.read_csv(GRID_DIR / 'all_folds.csv')

print(f"Loaded {len(summary_df)} combinations, {len(folds_df)} fold-level rows.")
print(f"\nTop {TOP_N} by {METRIC}_mean:")
print(summary_df.nlargest(TOP_N, f'{METRIC}_mean')
      [['architecture', 'n_pc', f'{METRIC}_mean', f'{METRIC}_std']]
      .to_string(index=False))

plots.plot_grid_heatmap(summary_df, metric=f'{METRIC}_mean',
                        title=f'Mean {METRIC.upper()} across the grid',
                        save_path=str(GRID_DIR / f'heatmap_{METRIC}.png'))

plots.plot_grid_lines(summary_df, metric=METRIC,
                      title=f'{METRIC.upper()} vs PCA components',
                      save_path=str(GRID_DIR / f'lines_{METRIC}.png'))

plots.plot_grid_leaderboard(summary_df, metric=METRIC, top_n=TOP_N,
                            title=f'Top {TOP_N} combinations by {METRIC.upper()}',
                            save_path=str(GRID_DIR / f'leaderboard_{METRIC}.png'))

top_combos = [(row.architecture, row.n_pc)
             for row in summary_df.nlargest(TOP_N, f'{METRIC}_mean').itertuples()]
plots.plot_grid_boxplot(folds_df, top_combos, metric=METRIC,
                        title=f'Per-fold {METRIC.upper()} — top {TOP_N} combinations',
                        save_path=str(GRID_DIR / f'boxplot_{METRIC}.png'))

plots.plot_grid_complexity(summary_df, metric=METRIC,
                           title=f'{METRIC.upper()} vs model complexity',
                           save_path=str(GRID_DIR / f'complexity_{METRIC}.png'))

print(f"\nPlots saved to: {GRID_DIR}")
