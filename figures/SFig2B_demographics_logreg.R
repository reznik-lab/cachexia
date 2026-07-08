suppressPackageStartupMessages({
  library(data.table)
  library(broom)
})

# SFig2B: Demographic associations with episode-defined weight loss.
# Univariable logistic regression by cancer type (n>=500) + pan-cancer, BH-FDR.
# Results-only (no heatmap plot - that panel is from a previous manuscript
# version and is no longer needed).
# Adapted from 0303_ccx_revisions/rev_code/github/regression_analysis.R

BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
DATE_STAMP  <- "20260706"

table_dir <- file.path(REV_RESULTS, "fearon_definition", "SFig2", "SFig2B_demographics_tables")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

msk_clin_fp <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")
bmi_fp      <- file.path(REV_INPUTS, "bmi_final_20260129.csv")
eps_fp      <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))

msk_clin <- fread(msk_clin_fp)
bmi      <- fread(bmi_fp)
eps      <- fread(eps_fp)

msk_clin[, MRN := as.integer(MRN)]
bmi[,     MRN := as.integer(MRN)]
eps[,     MRN := as.integer(MRN)]

has_ccx <- eps[, .(has_cachexia = as.integer(any(has_cachexia_valid_edemaQC == 1, na.rm = TRUE))), by = MRN]

dt <- merge(msk_clin, has_ccx, by = "MRN", all.x = TRUE)
dt[is.na(has_cachexia), has_cachexia := 0L]

bmi[, datetime := as.IDate(datetime)]
bmi[, days_since_anchor := as.integer(days_since_anchor)]
bmi0 <- bmi[days_since_anchor >= 0]
setorder(bmi0, MRN, days_since_anchor, datetime)

bmi0_day <- bmi0[, .(bmi = min(bmi, na.rm = TRUE)), by = .(MRN, days_since_anchor)]
bmi0_day <- bmi0_day[!is.na(bmi)]
baseline <- bmi0_day[bmi0_day[, .I[which.min(days_since_anchor)], by = MRN]$V1]

baseline[, bmi_cat := fifelse(
  is.na(bmi), NA_character_,
  fifelse(bmi < 18.5, "Underweight",
          fifelse(bmi < 25, "Normal",
                  fifelse(bmi < 30, "Overweight", "Obese")))
)]
baseline[, bmi_cat := factor(bmi_cat, levels = c("Normal", "Underweight", "Overweight", "Obese"))]

dt[baseline[, .(MRN, bmi_cat)], bmi_cat := i.bmi_cat, on = "MRN"]

dt[, AGE_AT_ANCHOR_YEARS := as.integer(AGE_AT_ANCHOR_YEARS)]
dt[AGE_AT_ANCHOR_YEARS < 0 | AGE_AT_ANCHOR_YEARS > 120, AGE_AT_ANCHOR_YEARS := NA_integer_]
dt[, age_group := fifelse(!is.na(AGE_AT_ANCHOR_YEARS) & AGE_AT_ANCHOR_YEARS >= 60, "≥60", "<60")]
dt[, age_group := factor(age_group, levels = c("<60", "≥60"))]
dt[, GENDER := factor(GENDER)]

uqctypes <- dt[!is.na(CANCER_TYPE_DETAILED),
               .(n_patients = uniqueN(MRN)),
               by = CANCER_TYPE_DETAILED][n_patients >= 500, CANCER_TYPE_DETAILED]
dt <- dt[CANCER_TYPE_DETAILED %in% uqctypes]

ref_gender <- "FEMALE"
ref_bmi    <- "Normal"
ref_age    <- "<60"
dt[, GENDER := factor(GENDER, levels = c(ref_gender, setdiff(unique(GENDER), ref_gender)))]
dt[, bmicat := factor(bmi_cat, levels = c(ref_bmi, setdiff(levels(bmi_cat), ref_bmi)))]
dt[, age    := factor(age_group, levels = c(ref_age, setdiff(levels(age_group), ref_age)))]

oncotree_map_dt <- dt[!is.na(ONCOTREE_CODE), .(ONCOTREE_CODE = ONCOTREE_CODE[1]), by = CANCER_TYPE_DETAILED]
oncotree_map <- setNames(oncotree_map_dt$ONCOTREE_CODE, oncotree_map_dt$CANCER_TYPE_DETAILED)

fit_one <- function(d, cancer_label) {
  out <- list()
  if (length(unique(na.omit(d$GENDER))) > 1) {
    m <- glm(has_cachexia ~ GENDER, data = d, family = binomial)
    r <- as.data.table(tidy(m, exponentiate = TRUE, conf.int = TRUE))
    r[, Variable := "Gender"]
    out <- c(out, list(r))
  }
  if (length(unique(na.omit(d$bmicat))) > 1) {
    m <- glm(has_cachexia ~ bmicat, data = d, family = binomial)
    r <- as.data.table(tidy(m, exponentiate = TRUE, conf.int = TRUE))
    r[, Variable := "BMI"]
    out <- c(out, list(r))
  }
  if (length(unique(na.omit(d$age))) > 1) {
    m <- glm(has_cachexia ~ age, data = d, family = binomial)
    r <- as.data.table(tidy(m, exponentiate = TRUE, conf.int = TRUE))
    r[, Variable := "Age"]
    out <- c(out, list(r))
  }
  res <- rbindlist(out, fill = TRUE)
  if (nrow(res) == 0) return(NULL)
  res[, Cancer_Type := cancer_label]
  res
}

res_list <- list()
for (ct in uqctypes) {
  d <- dt[CANCER_TYPE_DETAILED == ct]
  res_list[[ct]] <- fit_one(d, ct)
}
res_list[["All"]] <- fit_one(dt, "All")

or_results <- rbindlist(res_list, fill = TRUE)
or_results <- or_results[term != "(Intercept)"]

or_results[, term := gsub("^GENDER", "Gender_", term)]
or_results[, term := gsub("^bmicat", "BMI_", term)]
or_results[, term := gsub("^age", "Age_", term)]

or_results[, adjusted_p := p.adjust(p.value, method = "BH")]
or_results[, significance := fifelse(adjusted_p < 0.05, "*", "")]

sig <- copy(or_results)[adjusted_p < 0.05]
sig[, cancer := fifelse(Cancer_Type == "All", "All",
                        fifelse(!is.na(oncotree_map[Cancer_Type]),
                                oncotree_map[Cancer_Type], Cancer_Type))]

sig[, log2OR      := log2(estimate)]
sig[, log2CI_low  := log2(conf.low)]
sig[, log2CI_high := log2(conf.high)]
sig[, `:=`(
  log2OR      = round(log2OR, 3),
  log2CI_low  = round(log2CI_low, 3),
  log2CI_high = round(log2CI_high, 3),
  p.value     = signif(p.value, 3),
  adjusted_p  = signif(adjusted_p, 3)
)]
sig[, log2OR_CI := sprintf("%0.3f (%0.3f–%0.3f)", log2OR, log2CI_low, log2CI_high)]

clean_sig <- sig[, .(cancer, term, log2OR, log2OR_CI, p = p.value, FDR = adjusted_p)]
fwrite(clean_sig, file.path(table_dir, paste0("SFig2B_demographics_sig_table_", DATE_STAMP, ".csv")))

pancancer <- or_results[Cancer_Type == "All"]
cat("\n=== Pan-cancer (All) demographic associations with has_cachexia ===\n")
print(pancancer[, .(term, estimate, log2OR = round(log2(estimate), 3), p.value, adjusted_p)])

cat("\nWrote SFig2B significant table to:", table_dir, "\n")
