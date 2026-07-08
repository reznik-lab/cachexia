library(data.table)
library(dplyr)
library(lme4)
library(ggplot2)

# Derived-labs (composite index) GLMM, parameterized across the 5 sensitivity
# conditions used for Fig4F (main) and SFig4D (sensitivity):
#   Rscript Fig4F_derived_labs_glmm.R WL5_BMIlt20 ALL          (Fig4F main)
#   Rscript Fig4F_derived_labs_glmm.R WL5_BMIlt20 "Stage 1-3"
#   Rscript Fig4F_derived_labs_glmm.R WL5_BMIlt20 "Stage 4"
#   Rscript Fig4F_derived_labs_glmm.R WL10 ALL
#   Rscript Fig4F_derived_labs_glmm.R WL15 ALL

args <- commandArgs(trailingOnly = TRUE)
WL_LABEL     <- args[1]
STAGE_FILTER <- args[2]
stopifnot(WL_LABEL %in% c("WL5_BMIlt20", "WL10", "WL15"))
stopifnot(STAGE_FILTER %in% c("ALL", "Stage 1-3", "Stage 4"))

OUT_TAG <- if (STAGE_FILTER == "ALL") WL_LABEL else paste0(WL_LABEL, "_", gsub(" ", "", STAGE_FILTER))

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_TABLES  <- file.path(BASE_REV, "rev_tables")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots", "fearon_definition")
DATE_STAMP  <- "20260706"

spans_lab_fp <- file.path(REV_RESULTS, "serology_spans_labs",
                           paste0("spans_labtests_spans_fearon_", WL_LABEL, "_", DATE_STAMP, "_", DATE_STAMP, ".csv"))
msk_fp <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")

OUT_TAB_DIR  <- file.path(REV_TABLES, "derived_labs_glmm", paste0(OUT_TAG, "_", DATE_STAMP))
OUT_PLOT_DIR <- file.path(REV_PLOTS, if (STAGE_FILTER == "ALL" && WL_LABEL == "WL5_BMIlt20") "Fig4" else "SFig4")
dir.create(OUT_TAB_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

# candidate cancer-type list is fixed across main + all sensitivity conditions,
# computed from the full clinical cohort (matches original: msk_clin, threshold 500)
msk_clin <- fread(msk_fp)

sp <- fread(spans_lab_fp)
sp[CANCER_TYPE_DETAILED %in% c("Colon Adenocarcinoma", "Rectal Adenocarcinoma"),
   CANCER_TYPE_DETAILED := "Colorectal Adenocarcinoma"]
setnames(sp, "Days Since Diagnosis", "days_since_diagnosis", skip_absent = TRUE)

if (STAGE_FILTER != "ALL") {
  sp <- sp[STAGE_CDM_DERIVED == STAGE_FILTER]
}

sp2 <- copy(sp)
required_cols <- c("BUN", "Creatinine", "AST", "ALT", "Calcium", "Albumin", "Lymph", "Neut")
for (cc in required_cols) if (!cc %in% names(sp2)) sp2[, (cc) := NA_real_]
sp2[, (required_cols) := lapply(.SD, as.numeric), .SDcols = required_cols]

sp2[, BUN_Creatinine_Ratio := fifelse(!is.na(BUN) & !is.na(Creatinine) & Creatinine != 0,
                                      BUN / Creatinine, NA_real_)]
sp2[, AST_ALT_Ratio := fifelse(!is.na(AST) & !is.na(ALT) & ALT != 0,
                               AST / ALT, NA_real_)]
sp2[, Corrected_Calcium := fifelse(!is.na(Calcium) & !is.na(Albumin),
                                   Calcium + 0.8 * (4 - Albumin), NA_real_)]
sp2[, PNI := fifelse(!is.na(Albumin) & !is.na(Lymph),
                     Albumin + 5 * Lymph, NA_real_)]
sp2[, NLR := fifelse(!is.na(Neut) & !is.na(Lymph) & Lymph != 0,
                     Neut / Lymph, NA_real_)]

derived_labs <- c("BUN_Creatinine_Ratio", "AST_ALT_Ratio", "Corrected_Calcium", "PNI", "NLR")

keep_cols <- c("MRN", "span", "days_since_diagnosis", "start_day", "end_day",
               "CANCER_TYPE_DETAILED", "GENDER", "AGE_AT_ANCHOR_YEARS", "AGE_AT_ANCHOR", derived_labs)
for (cc in keep_cols) if (!cc %in% names(sp2)) sp2[, (cc) := NA]
derived_tests <- sp2[, ..keep_cols]

derived_long <- melt(
  derived_tests,
  id.vars = c("MRN", "span", "days_since_diagnosis", "start_day", "end_day",
              "CANCER_TYPE_DETAILED", "GENDER", "AGE_AT_ANCHOR_YEARS", "AGE_AT_ANCHOR"),
  measure.vars = derived_labs,
  variable.name = "lab_name",
  value.name   = "lab_value"
)

skewed <- c("BUN_Creatinine_Ratio", "AST_ALT_Ratio", "NLR")
derived_long[, value_for_scale := fifelse(lab_name %in% skewed, log1p(lab_value), lab_value)]

derived_long[, c("mu", "sigma") := .(
  mean(value_for_scale, na.rm = TRUE),
  sd(value_for_scale,   na.rm = TRUE)
), by = .(lab_name, GENDER)]

derived_long[is.na(sigma) | sigma <= 0, sigma := 1]
derived_long[, lab_value_z := (value_for_scale - mu) / sigma]
derived_long[, lab_value_z := pmax(pmin(lab_value_z, 6), -6)]

derived_long[, MRN := as.factor(MRN)]
derived_long[, span01 := suppressWarnings(as.integer(as.character(span)))]
if (any(is.na(derived_long$span01))) derived_long[, span01 := as.integer(span)]

derived_long[, AGE_AT_ANCHOR_YEARS := as.numeric(AGE_AT_ANCHOR_YEARS)]
derived_long[, AGE_AT_ANCHOR       := as.numeric(AGE_AT_ANCHOR)]

ctypes <- msk_clin[, .(n_patients = uniqueN(MRN)), by = CANCER_TYPE_DETAILED]
ctypes <- ctypes[n_patients >= 500][order(-n_patients)]

derived_results_list <- list()

for (cancer_type in ctypes$CANCER_TYPE_DETAILED) {
  for (lab in derived_labs) {
    subset_data <- derived_long[
      CANCER_TYPE_DETAILED == cancer_type &
        lab_name == lab &
        is.finite(lab_value_z) &
        !is.na(span01)
    ]

    if (nrow(subset_data) <= 10L) next
    if (subset_data[, uniqueN(span01)] < 2L) next
    if (subset_data[, uniqueN(MRN)]   < 2L) next

    include_gender <- subset_data[, uniqueN(GENDER)] > 1L
    frm <- if (include_gender) {
      span01 ~ lab_value_z + AGE_AT_ANCHOR_YEARS + GENDER + (1 | MRN)
    } else {
      span01 ~ lab_value_z + AGE_AT_ANCHOR_YEARS + (1 | MRN)
    }

    fit <- tryCatch(
      glmer(frm, data = subset_data, family = binomial("logit"),
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)),
            nAGQ = 0),
      error = function(e) NULL
    )
    if (is.null(fit) || isSingular(fit, tol = 1e-4)) next

    co <- summary(fit)$coefficients
    if (!"lab_value_z" %in% rownames(co)) next

    beta <- co["lab_value_z", "Estimate"]
    se   <- co["lab_value_z", "Std. Error"]
    pval <- co["lab_value_z", "Pr(>|z|)"]

    derived_results_list[[paste(cancer_type, lab, sep = "___")]] <- data.table(
      cancer_type = cancer_type, test = lab,
      estimate = exp(beta), logor = beta,
      lower_ci = exp(beta - 1.96 * se), upper_ci = exp(beta + 1.96 * se),
      p = pval, n_rows = nrow(subset_data), n_mrn = subset_data[, uniqueN(MRN)]
    )
  }
}

derived_results_final <- rbindlist(derived_results_list, use.names = TRUE, fill = TRUE)
derived_results_final[, adjusted_p := p.adjust(p, method = "BH")]

out_csv <- file.path(OUT_TAB_DIR, paste0("glmm_derived_labs_results_z_", OUT_TAG, "_", DATE_STAMP, ".csv"))
fwrite(derived_results_final, out_csv)

test_label_map <- c(
  "BUN_Creatinine_Ratio" = "BUN:Creatinine Ratio",
  "AST_ALT_Ratio"        = "AST:ALT Ratio",
  "Corrected_Calcium"    = "Corrected Calcium",
  "PNI"                  = "PNI",
  "NLR"                  = "NLR"
)

# same convention as Fig4C tissue-agnostic signature: mean log2(OR) + 95% CI error bars
mean_or_df <- derived_results_final %>%
  as.data.frame() %>%
  group_by(test) %>%
  summarise(
    mean_log_or = mean(logor, na.rm = TRUE),
    n_eff       = sum(!is.na(logor)),
    se          = sd(logor, na.rm = TRUE) / sqrt(pmax(n_eff, 1)),
    lower       = mean_log_or - 1.96 * se,
    upper       = mean_log_or + 1.96 * se,
    .groups     = "drop"
  ) %>%
  arrange(mean_log_or) %>%
  mutate(
    direction  = ifelse(mean_log_or > 0, "Enriched in Cachexia", "Down in Cachexia"),
    test_label = unname(test_label_map[test])
  )

mean_or_df <- mean_or_df %>%
  mutate(
    mean_log2_or = mean_log_or / log(2),
    lower2       = lower       / log(2),
    upper2       = upper       / log(2)
  )

max_abs <- with(mean_or_df, max(abs(c(lower2, upper2)), na.rm = TRUE))

p_bar <- ggplot(
  mean_or_df,
  aes(x = mean_log2_or, y = reorder(test_label, mean_log2_or), fill = direction)
) +
  geom_col(width = 0.8) +
  geom_segment(
    aes(x = lower2, xend = upper2,
        y = reorder(test_label, mean_log2_or), yend = reorder(test_label, mean_log2_or)),
    linewidth = 0.5
  ) +
  geom_vline(xintercept = 0, color = "gray40", linetype = "dashed") +
  scale_fill_manual(values = c(
    "Enriched in Cachexia" = "#EF6F6AFF",
    "Down in Cachexia"     = "#6388B4FF"
  )) +
  scale_x_continuous(
    limits = c(-max_abs, max_abs),
    breaks = seq(floor(-max_abs * 2) / 2, ceiling(max_abs * 2) / 2, by = 0.5),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  labs(title = "Derived labs", x = "Mean log2(OR)", y = NULL, fill = "Direction") +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title        = element_text(hjust = 0.5, size = 9),
    legend.position   = "top",
    legend.text       = element_text(size = 9),
    legend.title      = element_text(size = 9),
    axis.title.x      = element_text(size = 9),
    axis.title.y      = element_text(size = 9),
    axis.text.x       = element_text(size = 9),
    axis.text.y       = element_text(size = 9),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.line.y       = element_line(color = "black", linewidth = 0.25),
    axis.line.x       = element_line(color = "black", linewidth = 0.25),
    axis.ticks        = element_line(color = "black", linewidth = 0.3)
  )

ggsave(file.path(OUT_PLOT_DIR, paste0("derived_labs_summary_", OUT_TAG, "_", DATE_STAMP, ".pdf")),
       p_bar, width = 4.3, height = 3, device = "pdf")

fwrite(mean_or_df, file.path(OUT_PLOT_DIR, paste0("derived_labs_summary_data_", OUT_TAG, "_", DATE_STAMP, ".csv")))

cat("\n[", OUT_TAG, "] Wrote derived-labs GLMM outputs to:\n  ", OUT_TAB_DIR, "\n  ", OUT_PLOT_DIR, "\n")
