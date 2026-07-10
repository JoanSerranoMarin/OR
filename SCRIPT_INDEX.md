# Script index linked to the TFM

This index relates each script to the part of the TFM where the corresponding analysis is described or used.

## 01. BW annotation and distance descriptors

| Script | TFM link | Role in the analysis |
| --- | --- | --- |
| `scripts/01_bw_annotation_and_distances/gpcr_gpcrdb_tool.py` | Methods II.2-II.3 | GPCRdb-based structural annotation and extraction of conserved GPCR positions. |
| `scripts/01_bw_annotation_and_distances/1_gpcr_distances_only.py` | Methods II.3; Results III.1 | Computes inter-residue distance descriptors in GPCR structures. |
| `scripts/01_bw_annotation_and_distances/1_1_gpcr_distances_only.py` | Methods II.3; Results III.1 | Extended distance workflow including class annotation support through `familias.txt`. |
| `scripts/01_bw_annotation_and_distances/2_anotate_class.py` | Methods II.2-II.3 | Adds state and GPCR-class metadata to distance tables. |
| `scripts/01_bw_annotation_and_distances/gpcr_ionic_locks_from_excel.py` | Results III.1; Figures 1-2 | Extracts canonical and alternative ionic-lock metrics from receptor lists. |
| `scripts/01_bw_annotation_and_distances/gpcr_negativos_339_250.py` | Results III.1 | Computes metrics around the 3x39-2x50 acidic/polar region used in the structural descriptor comparison. |

## 02. Ionic-lock statistics and figures

| Script | TFM link | Role in the analysis |
| --- | --- | --- |
| `scripts/02_ionic_lock_statistics_and_figures/2_plot_dry_lock.py` | Results III.1; Figures 1-2 | Figure preparation for ionic-lock and DRY-related metrics. |
| `scripts/02_ionic_lock_statistics_and_figures/3_statistical_tests_distances.py` | Results III.1; Figures 1-2 | Statistical tests for distance descriptors across GPCR classes and states. |
| `scripts/02_ionic_lock_statistics_and_figures/make_gpcr_panels_from_table.py` | Results III.1; Figures 1-2 | Builds class-level panels from distance or log2FC tables. |
| `scripts/02_ionic_lock_statistics_and_figures/make_gpcr_lock_subsets.py` | Results III.1; Figures 1-2 | Separates receptor subsets according to canonical, alternative or mixed ionic-lock configurations. |
| `scripts/02_ionic_lock_statistics_and_figures/plot_339_250_like_original.py` | Results III.6; Figure 7 | Plots descriptors around the 3x39-2x50 region. |
| `scripts/02_ionic_lock_statistics_and_figures/alternative_statistical_tests_distances.py` | Results III.1 | Alternative statistical workflow used during distance-metric testing. |

## 03. Cavity-volume analysis

| Script | TFM link | Role in the analysis |
| --- | --- | --- |
| `scripts/03_cavity_volume_analysis/5_1_statistical_analysis_extracel_volumes.py` | Results III.4.1; Figure 3 | Statistical analysis and plotting of extracellular cavity volumes. |
| `scripts/03_cavity_volume_analysis/5_1_statistical_analysis_extracel_volumes_3x2_letters.py` | Results III.4.1; Figure 3 | Final multi-panel figure layout with post-hoc letters for extracellular volumes. |
| `scripts/03_cavity_volume_analysis/5_1_statistical_analysis_extracel_volumes_large_text.py` | Results III.4.1; Figure 3 | Figure-formatting variant for extracellular-volume panels. |
| `scripts/03_cavity_volume_analysis/5_1_statistical_analysis_extracel_volumes_narrow_heatmaps.py` | Results III.4.1; Figure 3 | Figure-formatting variant for extracellular-volume heatmaps. |
| `scripts/03_cavity_volume_analysis/5_1_statistical_analysis_extracel_volumes_readable.py` | Results III.4.1; Figure 3 | Readability-oriented figure-formatting variant for extracellular volumes. |
| `scripts/03_cavity_volume_analysis/5_2_statistical_analysis_transmembrane_volumes.py` | Results III.4.2; Figure 4 | Statistical analysis and plotting of transmembrane cavity volumes. |
| `scripts/03_cavity_volume_analysis/5_2_statistical_analysis_transmembrane_volumes_2x2_letters.py` | Results III.4.2; Figure 4 | Final multi-panel figure layout for transmembrane volume descriptors. |
| `scripts/03_cavity_volume_analysis/5_2_statistical_analysis_transmembrane_volumes_2x2_letters_autoROIs.py` | Results III.4.2; Figure 4 | Transmembrane-volume plotting with ROI labels inferred from inputs. |
| `scripts/03_cavity_volume_analysis/5_2_statistical_analysis_transmembrane_volumes_OPV_hydrophobic_slab_2x2_letters.py` | Results III.4.2; Figure 4 | Orthosteric pocket and hydrophobic-layer volume figure preparation. |
| `scripts/03_cavity_volume_analysis/5_3_statistical_analysis_intracellular_volumes.py` | Results III.4.3; Figure 5 | Statistical analysis and plotting of intracellular free volume. |
| `scripts/03_cavity_volume_analysis/5_3_statistical_analysis_intracellular_volumes_IC_1x2_letters.py` | Results III.4.3; Figure 5 | Final figure layout for intracellular-volume panels. |
| `scripts/03_cavity_volume_analysis/5_3_statistical_analysis_intracellular_volumes_IC_FIXED.py` | Results III.4.3; Figure 5 | Intracellular-volume figure workflow with fixed layout parameters. |
| `scripts/03_cavity_volume_analysis/5_3_statistical_analysis_intracellular_volumes_IC_FIXED_v2.py` | Results III.4.3; Figure 5 | Final adjusted intracellular-volume figure workflow. |
| `scripts/03_cavity_volume_analysis/make_gpcr_log2_excel_hydrophobic_slab.py` | Results III.4.2; Figure 4 | Creates active/inactive log2FC tables for the hydrophobic-layer cavity. |
| `scripts/03_cavity_volume_analysis/anova_log2_fc_por_clase_hydrophobic_slab.py` | Results III.4.2; Figure 4 | ANOVA and post-hoc analysis for hydrophobic-layer cavity log2FC values. |
| `scripts/03_cavity_volume_analysis/make_gpcr_log2_excel_ec_vestibule.py` | Results III.4.1; Figure 3 | Creates active/inactive log2FC tables for the EC vestibule. |
| `scripts/03_cavity_volume_analysis/anova_log2_fc_por_clase_ec_vestibule.py` | Results III.4.1; Figure 3 | ANOVA and post-hoc analysis for EC-vestibule log2FC values. |
| `scripts/03_cavity_volume_analysis/make_gpcr_log2_excel_cleft_volume.py` | Results III.4.1; Figure 3 | Creates active/inactive log2FC tables for the TM5-TM7 cleft. |
| `scripts/03_cavity_volume_analysis/anova_log2_fc_por_clase_cleft_volume.py` | Results III.4.1; Figure 3 | ANOVA and post-hoc analysis for TM5-TM7 cleft log2FC values. |
| `scripts/03_cavity_volume_analysis/make_gpcr_log2_excel_portal_throat.py` | Results III.4.1; Figure 3 | Creates active/inactive log2FC tables for the portal-throat descriptor. |
| `scripts/03_cavity_volume_analysis/anova_log2_fc_por_clase_portal_throat.py` | Results III.4.1; Figure 3 | ANOVA and post-hoc analysis for portal-throat log2FC values. |
| `scripts/03_cavity_volume_analysis/make_gpcr_log2_excel_portal_EC.py` | Results III.4.1; Figure 3 | Creates active/inactive log2FC tables for the extracellular portal descriptor. |
| `scripts/03_cavity_volume_analysis/anova_log2_fc_por_clase_portal_EC.py` | Results III.4.1; Figure 3 | ANOVA and post-hoc analysis for extracellular-portal log2FC values. |
| `scripts/03_cavity_volume_analysis/anova_log2_fc_por_clase_intracellular_cavity.py` | Results III.4.3; Figure 5 | ANOVA and post-hoc analysis for intracellular-cavity log2FC values. |
| `scripts/03_cavity_volume_analysis/make_gpcr_log2_excel_orthosteric_pocket.py` | Results III.4.2; Figure 4 | Creates active/inactive log2FC tables for orthosteric pocket volume. |
| `scripts/03_cavity_volume_analysis/anova_log2_fc_por_clase_orthosteric_pocket.py` | Results III.4.2; Figure 4 | ANOVA and post-hoc analysis for orthosteric-pocket log2FC values. |
| `scripts/03_cavity_volume_analysis/anova_log2_fc_por_clase_total_volume.py` | Results III.4 | ANOVA and post-hoc analysis for total cavity-volume descriptors. |
| `scripts/03_cavity_volume_analysis/family_mean_individual_points.py` | Results III.4 and III.6 | Family-level visualization of individual OR-family volume points. |
| `scripts/03_cavity_volume_analysis/family_paired_analysis.py` | Results III.4 and III.6 | Paired active/inactive comparisons at OR-family level. |
| `scripts/03_cavity_volume_analysis/family_state_pairwise_analysis.py` | Results III.4 and III.6 | Pairwise family comparisons by receptor state. |
| `scripts/03_cavity_volume_analysis/odorant_family_three_metric_analysis.py` | Results III.4 and III.6 | Joint analysis of three OR-family volume/access descriptors. |

## 04. ECL2 and disulfide-region analysis

| Script | TFM link | Role in the analysis |
| --- | --- | --- |
| `scripts/04_ecl2_disulfide_analysis/analyze_ecl2_z_depth.py` | Results III.5; Figure 6 | Measures ECL2 positional descriptors and extracellular-region architecture. |
| `scripts/04_ecl2_disulfide_analysis/analyze_ecl2_z_depth_v6_aligned_diagnostics.py` | Results III.5; Figure 6 | Diagnostic aligned-structure workflow for ECL2 depth analysis. |
| `scripts/04_ecl2_disulfide_analysis/analyze_ecl2_z_depth_v8_by_class_family.py` | Results III.5; Figure 6 | Final class/family-level ECL2 and disulfide-region analysis. |

## 05. Chemical voxel maps

| Script | TFM link | Role in the analysis |
| --- | --- | --- |
| `scripts/05_chemical_voxel_maps/tm_grid_density_byclass_mean_excel.py` | Methods II.3; Results III.6; Figure 7 | Builds class-level voxel chemical-density maps and pairwise voxel statistics. |
| `scripts/05_chemical_voxel_maps/pairwise_xlsx_voxels_to_pdb.py` | Results III.6; Figure 7 | Exports significant pairwise chemical-voxel differences to PDB/PML for structural inspection. |

## 06. Cavity geometry helpers

| Script | TFM link | Role in the analysis |
| --- | --- | --- |
| `scripts/06_cavity_geometry_helpers/4_volumes.py` | Methods II.3; Results III.4 | Membrane-aware cavity-volume utility used for receptor cavity descriptors. |
| `scripts/06_cavity_geometry_helpers/gpcr_cavity_volume.py` | Methods II.3; Results III.4 | Core geometric functions for GPCR cavity volume calculation. |
| `scripts/06_cavity_geometry_helpers/gpcr_cavity_pymol_vis.py` | Methods II.3; Results III.4 | PyMOL visualization helper for cavity regions. |
| `scripts/06_cavity_geometry_helpers/make_OPV_voxels_or8k3.py` | Methods II.3; Results III.4.2 | Helper for orthosteric-pocket voxel construction. |
| `scripts/06_cavity_geometry_helpers/plot_tm_mesh_3d.py` | Methods II.3; Results III.4 | 3D plotting helper for transmembrane mesh/cavity geometry. |

## 07. Experimental-structure controls

| Script | TFM link | Role in the analysis |
| --- | --- | --- |
| `scripts/07_experimental_structure_controls/experimentales.py` | Results III.1; Supplementary Figures 1-2 | Retrieves and prepares experimentally determined GPCR structures for control analyses. |
| `scripts/07_experimental_structure_controls/gpcr_ionic_locks_from_excel.py` | Results III.1; Supplementary Figures 1-2 | Computes ionic-lock metrics for experimentally determined GPCR structures. |
| `scripts/07_experimental_structure_controls/plot_locks_from_excels.py` | Results III.1; Supplementary Figures 1-2 | Plots experimental-structure control results for ionic-lock subsets. |
