# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
source("rev_code/Figures/figure_panel/SFig2_WLGS_wholecohort_setup.R")

# -----------------------------
# 5x5 WLGS grid (cachexia episodes only, weight-stable patients excluded):
# rows = baseline BMI bin
# cols = weight-loss bin (the same bins used by WLGS)
# -----------------------------

grid_counts <- eps_wlgs_episode[, .N, by = .(bmi_bin, wl_bin)]
setnames(grid_counts, "N", "Count")

grid_counts[, RowTotal := sum(Count), by = bmi_bin]
grid_counts[, RowProp  := Count / RowTotal]

grid_5x5_counts  <- dcast(grid_counts, bmi_bin ~ wl_bin, value.var = "Count",  fill = 0)
grid_5x5_rowprop <- dcast(grid_counts, bmi_bin ~ wl_bin, value.var = "RowProp", fill = 0)

print(grid_5x5_counts)
print(grid_5x5_rowprop)

row_sums <- grid_5x5_rowprop[, lapply(.SD, as.numeric), .SDcols = setdiff(names(grid_5x5_rowprop), "bmi_bin")]
row_sums[, row_sum := rowSums(.SD)]
print(data.table(bmi_bin = grid_5x5_rowprop$bmi_bin, row_sum = row_sums$row_sum))

fwrite(grid_5x5_rowprop, file.path(table_dir, paste0("SFig2_WLGS_5x5_rowprop_", DATE_STAMP, ".csv")))
fwrite(grid_5x5_counts,  file.path(table_dir, paste0("SFig2_WLGS_5x5_counts_", DATE_STAMP, ".csv")))

cat("\nWrote grid outputs to:", table_dir, "\n")
