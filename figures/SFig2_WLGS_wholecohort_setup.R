library(data.table)

BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
DATE_STAMP  <- "20260706"

plot_dir  <- file.path(REV_PLOTS,   "fearon_definition", "SFig2")
table_dir <- file.path(REV_RESULTS, "fearon_definition", "SFig2")
dir.create(plot_dir,  recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

eps_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
eps <- fread(eps_fp)

bmi_fp <- file.path(REV_RESULTS, paste0("smoothed_bmi_all_patients_", DATE_STAMP, "_alpha0.2.csv"))
bmi_all <- fread(bmi_fp)
setorder(bmi_all, MRN, days_since_anchor)
baseline_bmi <- bmi_all[, .SD[1], by = MRN][, .(MRN, baseline_bmi = smoothed_BMI)]

assign_wlgs <- function(dt) {
  dt <- copy(dt)
  dt[, wlgs := fcase(
    weight_loss < 2.5 & start_bmi >= 28, 0L,
    weight_loss < 2.5 & start_bmi >= 25 & start_bmi < 28, 0L,
    weight_loss < 2.5 & start_bmi >= 22 & start_bmi < 25, 1L,
    weight_loss < 2.5 & start_bmi >= 20 & start_bmi < 22, 1L,
    weight_loss < 2.5 & start_bmi < 20, 3L,

    weight_loss >= 2.5 & weight_loss < 6 & start_bmi >= 28, 1L,
    weight_loss >= 2.5 & weight_loss < 6 & start_bmi >= 25 & start_bmi < 28, 2L,
    weight_loss >= 2.5 & weight_loss < 6 & start_bmi >= 22 & start_bmi < 25, 2L,
    weight_loss >= 2.5 & weight_loss < 6 & start_bmi >= 20 & start_bmi < 22, 2L,
    weight_loss >= 2.5 & weight_loss < 6 & start_bmi < 20, 3L,

    weight_loss >= 6 & weight_loss < 11 & start_bmi >= 28, 2L,
    weight_loss >= 6 & weight_loss < 11 & start_bmi >= 25 & start_bmi < 28, 3L,
    weight_loss >= 6 & weight_loss < 11 & start_bmi >= 22 & start_bmi < 25, 3L,
    weight_loss >= 6 & weight_loss < 11 & start_bmi >= 20 & start_bmi < 22, 3L,
    weight_loss >= 6 & weight_loss < 11 & start_bmi < 20, 4L,

    weight_loss >= 11 & weight_loss < 15 & start_bmi >= 28, 3L,
    weight_loss >= 11 & weight_loss < 15 & start_bmi >= 25 & start_bmi < 28, 3L,
    weight_loss >= 11 & weight_loss < 15 & start_bmi >= 22 & start_bmi < 25, 3L,
    weight_loss >= 11 & weight_loss < 15 & start_bmi >= 20 & start_bmi < 22, 4L,
    weight_loss >= 11 & weight_loss < 15 & start_bmi < 20, 4L,

    weight_loss >= 15 & start_bmi >= 28, 3L,
    weight_loss >= 15 & start_bmi >= 25 & start_bmi < 28, 4L,
    weight_loss >= 15 & start_bmi >= 22 & start_bmi < 25, 4L,
    weight_loss >= 15 & start_bmi >= 20 & start_bmi < 22, 4L,
    weight_loss >= 15 & start_bmi < 20, 4L
  )]

  dt[, bmi_bin := fifelse(start_bmi < 20, "<20",
                    fifelse(start_bmi < 22, "20–<22",
                     fifelse(start_bmi < 25, "22–<25",
                      fifelse(start_bmi < 28, "25–<28", "≥28"))))]

  dt[, wl_bin := fifelse(weight_loss < 2.5, "<2.5",
                   fifelse(weight_loss < 6,   "2.5–<6",
                    fifelse(weight_loss < 11,  "6–<11",
                     fifelse(weight_loss < 15,  "11–<15", "≥15"))))]

  dt[, bmi_bin := factor(bmi_bin, levels = c("<20","20–<22","22–<25","25–<28","≥28"))]
  dt[, wl_bin  := factor(wl_bin,  levels = c("<2.5","2.5–<6","6–<11","11–<15","≥15"))]

  dt[!is.na(wlgs)]
}

eps_wlgs_episode <- assign_wlgs(eps[has_cachexia_valid_edemaQC == 1, .(MRN, start_day, start_bmi, weight_loss)])

has_ep   <- eps[has_cachexia_valid_edemaQC == 1, .(MRN, start_bmi, weight_loss = weight_loss)]
no_ep    <- eps[has_cachexia_valid_edemaQC == 0, .(MRN)]
no_ep    <- merge(no_ep, baseline_bmi, by = "MRN", all.x = TRUE)
no_ep    <- no_ep[, .(MRN, start_bmi = baseline_bmi, weight_loss = 0)]

full_dt <- rbind(has_ep, no_ep)
full_dt <- full_dt[!is.na(start_bmi) & !is.na(weight_loss)]

stopifnot(uniqueN(full_dt$MRN) == uniqueN(eps$MRN))

full_dt <- assign_wlgs(full_dt)

setorder(full_dt, MRN, -wlgs, -weight_loss)
patient_worst <- full_dt[, .SD[1], by = MRN]

cat(sprintf("Whole-cohort patient-level table: %d patients (%d had >=1 valid episode, %d weight-stable)\n",
            uniqueN(patient_worst$MRN), uniqueN(has_ep$MRN), uniqueN(no_ep$MRN)))
