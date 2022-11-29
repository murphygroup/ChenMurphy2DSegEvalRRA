dir=$(pwd)

echo "finalizing metrics..."
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CellDIVE cell_matched all_tissues repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_matched all_tissues repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC cell_matched all_tissues repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/MIBI cell_matched all_tissues repaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CellDIVE nuclear_matched all_tissues repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX nuclear_matched all_tissues repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC nuclear_matched all_tissues repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/MIBI nuclear_matched all_tissues repaired
#
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CellDIVE cell_outside_nucleus_matched all_tissues repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_outside_nucleus_matched all_tissues repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC cell_outside_nucleus_matched all_tissues repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/MIBI cell_outside_nucleus_matched all_tissues repaired
#
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CellDIVE cell_matched all_tissues nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_matched all_tissues nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC cell_matched all_tissues nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/MIBI cell_matched all_tissues nonrepaired
#
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CellDIVE nuclear_matched all_tissues nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX nuclear_matched all_tissues nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC nuclear_matched all_tissues nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/MIBI nuclear_matched all_tissues nonrepaired
#
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CellDIVE cell_outside_nucleus_matched all_tissues nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_outside_nucleus_matched all_tissues nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC cell_outside_nucleus_matched all_tissues nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/MIBI cell_outside_nucleus_matched all_tissues nonrepaired





python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_matched Spleen repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_matched Large_Intestine repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_matched Small_Intestine repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_matched Lymph_Node repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_matched Thymus repaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_matched Spleen nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_matched Large_Intestine nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_matched Small_Intestine nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_matched Lymph_Node nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_matched Thymus nonrepaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_outside_nucleus_matched Spleen repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_outside_nucleus_matched Large_Intestine repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_outside_nucleus_matched Small_Intestine repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_outside_nucleus_matched Lymph_Node repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX cell_outside_nucleus_matched Thymus repaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_outside_nucleus_matched Spleen nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_outside_nucleus_matched Large_Intestine nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_outside_nucleus_matched Small_Intestine nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_outside_nucleus_matched Lymph_Node nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX cell_outside_nucleus_matched Thymus nonrepaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX nuclear_matched Spleen repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX nuclear_matched Large_Intestine repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX nuclear_matched Small_Intestine repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX nuclear_matched Lymph_Node repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX nuclear_matched Thymus repaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX nuclear_matched Spleen nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX nuclear_matched Large_Intestine nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX nuclear_matched Small_Intestine nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX nuclear_matched Lymph_Node nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX nuclear_matched Thymus nonrepaired




python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC cell_matched Spleen repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC cell_matched Lymph_Node repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC cell_matched Thymus repaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC cell_matched Spleen nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC cell_matched Lymph_Node nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC cell_matched Thymus nonrepaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC cell_outside_nucleus_matched Spleen repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC cell_outside_nucleus_matched Lymph_Node repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC cell_outside_nucleus_matched Thymus repaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC cell_outside_nucleus_matched Spleen nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC cell_outside_nucleus_matched Lymph_Node nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC cell_outside_nucleus_matched Thymus nonrepaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC nuclear_matched Spleen repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC nuclear_matched Lymph_Node repaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC nuclear_matched Thymus repaired

python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC nuclear_matched Spleen nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC nuclear_matched Lymph_Node nonrepaired
python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC nuclear_matched Thymus nonrepaired

