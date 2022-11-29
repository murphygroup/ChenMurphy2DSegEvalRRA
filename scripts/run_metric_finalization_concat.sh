dir=$(pwd)

 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CellDIVE concatenated_compartments all_tissues repaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX concatenated_compartments all_tissues repaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC concatenated_compartments all_tissues repaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/MIBI concatenated_compartments all_tissues repaired

 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CellDIVE concatenated_compartments all_tissues nonrepaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX concatenated_compartments all_tissues nonrepaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC concatenated_compartments all_tissues nonrepaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/MIBI concatenated_compartments all_tissues nonrepaired




 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX concatenated_compartments Spleen repaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX concatenated_compartments Large_Intestine repaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX concatenated_compartments Small_Intestine repaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX concatenated_compartments Lymph_Node repaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/CODEX concatenated_compartments Thymus repaired

 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX concatenated_compartments Spleen nonrepaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX concatenated_compartments Large_Intestine nonrepaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX concatenated_compartments Small_Intestine nonrepaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX concatenated_compartments Lymph_Node nonrepaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/CODEX concatenated_compartments Thymus nonrepaired



 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC concatenated_compartments Spleen repaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC concatenated_compartments Lymph_Node repaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_repaired/IMC concatenated_compartments Thymus repaired

 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC concatenated_compartments Spleen nonrepaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC concatenated_compartments Lymph_Node nonrepaired
 python $dir/scripts/metric_finalization.py $dir/data/intermediate/manuscript_v28_nonrepaired/IMC concatenated_compartments Thymus nonrepaired
