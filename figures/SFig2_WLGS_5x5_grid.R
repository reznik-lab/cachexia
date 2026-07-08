source("rev_code/Figures/figure_panel/SFig2_WLGS_wholecohort_setup.R")


grid_counts <- eps_wlgs_episode[, .N, by = .(bmi_bin, wl_bin)]
setnames(grid_counts, "N", "Count")

grid_counts[, RowTotal := sum(Count), by = bmi_bin]
grid_counts[, RowProp  := Count / RowTotal]

bmi_order_desc <- rev(levels(eps_wlgs_episode$bmi_bin))
wl_order_asc   <- levels(eps_wlgs_episode$wl_bin)

grid_counts[, bmi_bin := factor(bmi_bin, levels = bmi_order_desc)]
grid_counts[, wl_bin  := factor(wl_bin,  levels = wl_order_asc)]

grid_5x5_counts  <- dcast(grid_counts, wl_bin ~ bmi_bin, value.var = "Count",  fill = 0)
grid_5x5_rowprop <- dcast(grid_counts, wl_bin ~ bmi_bin, value.var = "RowProp", fill = 0)

setcolorder(grid_5x5_counts,  c("wl_bin", bmi_order_desc))
setcolorder(grid_5x5_rowprop, c("wl_bin", bmi_order_desc))
setorder(grid_5x5_counts, wl_bin)
setorder(grid_5x5_rowprop, wl_bin)

print(grid_5x5_counts)
print(grid_5x5_rowprop)

col_sums <- grid_5x5_rowprop[, lapply(.SD, as.numeric), .SDcols = bmi_order_desc]
print(data.table(bmi_bin = bmi_order_desc, col_sum = sapply(col_sums, sum)))

fwrite(grid_5x5_rowprop, file.path(table_dir, paste0("SFig2_WLGS_5x5_rowprop_", DATE_STAMP, ".csv")))
fwrite(grid_5x5_counts,  file.path(table_dir, paste0("SFig2_WLGS_5x5_counts_", DATE_STAMP, ".csv")))

cat("\nWrote grid outputs to:", table_dir, "\n")
