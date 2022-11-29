dir=$(pwd)
echo 'training 10-metric PCA model...'
python $dir/scripts/train_PCA_model_10_metrics.py merge cell_matched all all_tissues
