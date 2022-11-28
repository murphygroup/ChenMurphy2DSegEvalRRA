# Fig 1, Fig 3, Fig 4, Sup Fig 3-13
bash run_metric_calculation.sh
bash run_metric_calculation_tissues_modalities.sh
bash run_metric_calculation_concat.sh
bash run_train_PCA_model.sh
# Fig 2
python hetero_metrics.py
# Fig 5
bash run_train_PCA_model_10_metrics.sh
python annotation_eval.py
# Fig 6
python benchmark_corr.py
# Fig 7
python run_evaluation_merged.py
python run_evaluation_repaired.py
python run_evaluation_shifted.py
python plot_merge_metrics.py
# Fig 8
python pairwise_metrics_vis.py
python pairwise_metrics_vis_single.py
