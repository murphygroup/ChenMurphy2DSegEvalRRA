dir=$(pwd)

# Fig 1, Fig 3, Fig 4
bash $dir/scripts/run_metric_finalization.sh
bash $dir/scripts/run_metric_finalization_concat.sh
bash $dir/scripts/run_train_PCA_model.sh
# Fig 2
python $dir/scripts/hetero_metrics.py
# Fig 5
bash $dir/scripts/run_train_PCA_model_10_metrics.sh
python $dir/scripts/annotation_eval.py
# Fig 6
python $dir/scripts/benchmark_corr_plot.py
# Fig 7
python $dir/scripts/plot_underseg_eval.py
python $dir/scripts/plot_merge_metrics.py
# Fig 8
python $dir/scripts/pairwise_metrics_vis.py
python $dir/scripts/pairwise_metrics_vis_single.py
