dir=$(pwd)
echo 'training PCA model...'
python $dir/scripts/train_PCA_model.py merge concatenated_compartments all_modalities all_tissues all
python $dir/scripts/train_PCA_model.py merge concatenated_compartments all_modalities all_tissues grey
python $dir/scripts/train_PCA_model.py merge concatenated_compartments all_modalities all_tissues less
python $dir/scripts/train_PCA_model.py gaussian concatenated_compartments CODEX all_tissues modality
python $dir/scripts/train_PCA_model.py gaussian concatenated_compartments CellDIVE all_tissues modality
python $dir/scripts/train_PCA_model.py gaussian concatenated_compartments MIBI all_tissues modality
python $dir/scripts/train_PCA_model.py gaussian concatenated_compartments IMC all_tissues modality
####
python $dir/scripts/train_PCA_model.py gaussian concatenated_compartments CODEX Large_Intestine tissue
python $dir/scripts/train_PCA_model.py gaussian concatenated_compartments CODEX Small_Intestine tissue
###
python $dir/scripts/train_PCA_model.py gaussian concatenated_compartments CODEX_IMC Spleen tissue
python $dir/scripts/train_PCA_model.py gaussian concatenated_compartments CODEX_IMC Thymus tissue
python $dir/scripts/train_PCA_model.py gaussian concatenated_compartments CODEX_IMC Lymph_Node tissue

#
#
