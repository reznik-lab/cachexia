library(data.table)

# WLGS grade distribution + WL10/WL15 prevalence among the BMI>=28 ("obese")
# subgroup of cachexia episodes (manuscript Supplementary Table 3).
# Adapted from 0303_ccx_revisions/rev_code/rebuttal_figs_v1.R (WLG / >=28 sections).
# Note: the original script itself used BMI>=28 (the top bin of its 5-level
# BMI scheme) as the "obesity" cutoff, not a literal WHO BMI>=30 - so no
# separate >=30 tabulation is needed; this replicates the same >=28 logic.

BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
DATE_STAMP  <- "20260706"

eps_fp  <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
e10_fp  <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL10_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
e15_fp  <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL15_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))

eps <- fread(eps_fp)
e10 <- fread(e10_fp)
e15 <- fread(e15_fp)
eps[, MRN := as.integer(MRN)]
e10[, MRN := as.integer(MRN)]
e15[, MRN := as.integer(MRN)]

dt <- eps[
  has_cachexia_valid_edemaQC == 1 &
    is.finite(start_bmi) & is.finite(weight_loss) &
    is.finite(start_day) & is.finite(end_day),
  .(MRN, start_bmi = as.numeric(start_bmi), bmi_loss_pct = as.numeric(weight_loss))
]
cat("Episodes after filter:", nrow(dt), "\n")

dt[, wlgs := NA_integer_]
dt[bmi_loss_pct < 2.5 & start_bmi >= 28,                  wlgs := 0L]
dt[bmi_loss_pct < 2.5 & start_bmi >= 25 & start_bmi < 28, wlgs := 0L]
dt[bmi_loss_pct < 2.5 & start_bmi >= 22 & start_bmi < 25, wlgs := 1L]
dt[bmi_loss_pct < 2.5 & start_bmi >= 20 & start_bmi < 22, wlgs := 1L]
dt[bmi_loss_pct < 2.5 & start_bmi < 20,                   wlgs := 3L]

dt[bmi_loss_pct >= 2.5 & bmi_loss_pct < 6 & start_bmi >= 28,                  wlgs := 1L]
dt[bmi_loss_pct >= 2.5 & bmi_loss_pct < 6 & start_bmi >= 25 & start_bmi < 28, wlgs := 2L]
dt[bmi_loss_pct >= 2.5 & bmi_loss_pct < 6 & start_bmi >= 22 & start_bmi < 25, wlgs := 2L]
dt[bmi_loss_pct >= 2.5 & bmi_loss_pct < 6 & start_bmi >= 20 & start_bmi < 22, wlgs := 2L]
dt[bmi_loss_pct >= 2.5 & bmi_loss_pct < 6 & start_bmi < 20,                   wlgs := 3L]

dt[bmi_loss_pct >= 6 & bmi_loss_pct < 11 & start_bmi >= 28,                  wlgs := 2L]
dt[bmi_loss_pct >= 6 & bmi_loss_pct < 11 & start_bmi >= 25 & start_bmi < 28, wlgs := 3L]
dt[bmi_loss_pct >= 6 & bmi_loss_pct < 11 & start_bmi >= 22 & start_bmi < 25, wlgs := 3L]
dt[bmi_loss_pct >= 6 & bmi_loss_pct < 11 & start_bmi >= 20 & start_bmi < 22, wlgs := 3L]
dt[bmi_loss_pct >= 6 & bmi_loss_pct < 11 & start_bmi < 20,                   wlgs := 4L]

dt[bmi_loss_pct >= 11 & bmi_loss_pct < 15 & start_bmi >= 28,                  wlgs := 3L]
dt[bmi_loss_pct >= 11 & bmi_loss_pct < 15 & start_bmi >= 25 & start_bmi < 28, wlgs := 3L]
dt[bmi_loss_pct >= 11 & bmi_loss_pct < 15 & start_bmi >= 22 & start_bmi < 25, wlgs := 3L]
dt[bmi_loss_pct >= 11 & bmi_loss_pct < 15 & start_bmi >= 20 & start_bmi < 22, wlgs := 4L]
dt[bmi_loss_pct >= 11 & bmi_loss_pct < 15 & start_bmi < 20,                  wlgs := 4L]

dt[bmi_loss_pct >= 15 & start_bmi >= 28,                  wlgs := 3L]
dt[bmi_loss_pct >= 15 & start_bmi >= 25 & start_bmi < 28, wlgs := 4L]
dt[bmi_loss_pct >= 15 & start_bmi >= 22 & start_bmi < 25, wlgs := 4L]
dt[bmi_loss_pct >= 15 & start_bmi >= 20 & start_bmi < 22, wlgs := 4L]
dt[bmi_loss_pct >= 15 & start_bmi < 20,                   wlgs := 4L]

dt <- dt[!is.na(wlgs)]

# obesity subgroup: BMI >= 28 episodes
ob_dt <- dt[start_bmi >= 28]
cat("\nBMI >=28 episodes:", nrow(ob_dt), "\n")

ob_sum <- ob_dt[, .N, by = wlgs][order(wlgs)]
ob_sum[, pct := 100 * N / sum(N)]
cat("\nWLGS grade distribution among BMI>=28 episodes:\n")
print(ob_sum)

grade34_pct <- ob_sum[wlgs %in% c(3, 4), sum(pct)]
grade2_pct  <- ob_sum[wlgs == 2, pct]
cat(sprintf("\nGrade 3-4: %.1f%%  |  Grade 2: %.1f%%\n", grade34_pct, grade2_pct))

# among these same obese WL5+ patients, % also meeting WL10 / WL15
p10 <- unique(e10[has_cachexia_valid_edemaQC == 1 & !is.na(MRN), .(MRN)])[, has_WL10 := 1L]
p15 <- unique(e15[has_cachexia_valid_edemaQC == 1 & !is.na(MRN), .(MRN)])[, has_WL15 := 1L]

ob_pat <- unique(ob_dt[, .(MRN)])
ob_pat <- merge(ob_pat, p10, by = "MRN", all.x = TRUE)
ob_pat <- merge(ob_pat, p15, by = "MRN", all.x = TRUE)
ob_pat[is.na(has_WL10), has_WL10 := 0L]
ob_pat[is.na(has_WL15), has_WL15 := 0L]

pct_wl10 <- 100 * mean(ob_pat$has_WL10)
pct_wl15 <- 100 * mean(ob_pat$has_WL15)
cat(sprintf(">=10%%WL: %.1f%%  |  >=15%%WL: %.1f%%  (n=%d obese WL5+ patients)\n",
            pct_wl10, pct_wl15, nrow(ob_pat)))

out_dir <- file.path(BASE_REV, "rev_results", "fearon_definition", "SFig2")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out <- data.table(
  metric = c(paste0("WLGS_grade", 0:4), "pct_WL10", "pct_WL15"),
  value  = c(ob_sum$pct[match(0:4, ob_sum$wlgs)], pct_wl10, pct_wl15)
)
fwrite(out, file.path(out_dir, paste0("SFig2_WLGS_obesity_subgroup_", DATE_STAMP, ".csv")))
cat("\nWrote SFig2_WLGS_obesity_subgroup_", DATE_STAMP, ".csv to ", out_dir, "\n", sep = "")
