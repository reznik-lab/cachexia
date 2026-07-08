library(data.table)
library(lme4)
library(dplyr)
library(ggplot2)

# Additional-labs GLMM, parameterized across the 5 sensitivity conditions used
# for Fig4G (main) and SFig4D (sensitivity). Requires additional_labs_merge.py
# to have been run first for the given WL_LABEL.
#   Rscript Fig4G_additional_labs_glmm.R WL5_BMIlt20 ALL          (Fig4G main)
#   Rscript Fig4G_additional_labs_glmm.R WL5_BMIlt20 "Stage 1-3"
#   Rscript Fig4G_additional_labs_glmm.R WL5_BMIlt20 "Stage 4"
#   Rscript Fig4G_additional_labs_glmm.R WL10 ALL
#   Rscript Fig4G_additional_labs_glmm.R WL15 ALL

args <- commandArgs(trailingOnly = TRUE)
WL_LABEL     <- args[1]
STAGE_FILTER <- args[2]
stopifnot(WL_LABEL %in% c("WL5_BMIlt20", "WL10", "WL15"))
stopifnot(STAGE_FILTER %in% c("ALL", "Stage 1-3", "Stage 4"))

OUT_TAG <- if (STAGE_FILTER == "ALL") WL_LABEL else paste0(WL_LABEL, "_", gsub(" ", "", STAGE_FILTER))
IS_MAIN <- (STAGE_FILTER == "ALL" && WL_LABEL == "WL5_BMIlt20")
# same fixed 18-cancer-type candidate list (>=500 patients) used for main AND
# all sensitivity conditions, consistent with routine/derived labs
min_patients <- 500L

BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_TABLES  <- file.path(BASE_REV, "rev_tables")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots", "fearon_definition")
DATE_STAMP  <- "20260706"

spans_fp <- file.path(REV_RESULTS, "serology_spans_labs", paste0("spans_additionallabs_", WL_LABEL, "_", DATE_STAMP, ".csv"))
msk_fp   <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")

OUT_TAB_DIR  <- file.path(REV_TABLES, "serology_additional_labs", paste0(OUT_TAG, "_", DATE_STAMP))
OUT_PLOT_DIR <- file.path(REV_PLOTS, if (IS_MAIN) "Fig4" else "SFig4")
dir.create(OUT_TAB_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

labs2use <- c("Ferritin", "Triglyceride", "Cholesterol", "HDL Cholesterol", "LDL Cholesterol", "C-Reactive Protein")
skewed   <- c("Ferritin", "Triglyceride", "C-Reactive Protein")

# candidate cancer-type list is fixed across main + all sensitivity conditions,
# computed from the full clinical cohort (matches original: msk_clin). Threshold
# is 500 for the main panel, 300 for sensitivity (matches original SFig4.R,
# which used a lower bar since additional labs are drawn far less frequently).
# Stage filtering IS applied to the modeling data below (the 0303 original left
# this filter commented out for additional labs - a bug we're fixing here).
msk_clin <- fread(msk_fp)

labs_span <- fread(spans_fp)

if (STAGE_FILTER != "ALL") {
  labs_span <- labs_span[STAGE_CDM_DERIVED == STAGE_FILTER]
}

labs_span[, span01 := suppressWarnings(as.integer(as.character(span)))]
labs_span[is.na(span01), span01 := suppressWarnings(as.integer(span))]
labs_span <- labs_span[span01 %in% c(0L, 1L)]

labs_span[, value_for_scale := fifelse(CLEANED_TEST_NAME %in% skewed, log1p(LR_RESULT_VALUE), LR_RESULT_VALUE)]
labs_span[, c("mu", "sigma") := .(
  mean(value_for_scale, na.rm = TRUE),
  sd(value_for_scale,   na.rm = TRUE)
), by = .(CLEANED_TEST_NAME, GENDER)]
labs_span[is.na(sigma) | sigma <= 0, sigma := 1]
labs_span[, LR_RESULT_Z := (value_for_scale - mu) / sigma]
labs_span[, LR_RESULT_Z := pmax(pmin(LR_RESULT_Z, 6), -6)]

labs_span[, MRN := as.factor(MRN)]
labs_span[, AGE_AT_ANCHOR_YEARS := as.numeric(AGE_AT_ANCHOR_YEARS)]

ctypes <- msk_clin[, .(n_patients = uniqueN(MRN)), by = CANCER_TYPE_DETAILED]
ctypes <- ctypes[n_patients >= min_patients][order(-n_patients)]
cancers <- ctypes$CANCER_TYPE_DETAILED

results_list <- list()

for (ct in cancers) {
  for (lab in labs2use) {
    sub <- labs_span[
      CANCER_TYPE_DETAILED == ct &
        CLEANED_TEST_NAME == lab &
        is.finite(LR_RESULT_Z) &
        !is.na(span01)
    ]
    if (nrow(sub) < 20L) next
    if (sub[, uniqueN(span01)] < 2L) next
    if (sub[, uniqueN(MRN)] < 2L) next

    include_gender <- sub[, uniqueN(GENDER)] > 1L
    frm <- if (include_gender) {
      span01 ~ LR_RESULT_Z + AGE_AT_ANCHOR_YEARS + GENDER + (1 | MRN)
    } else {
      span01 ~ LR_RESULT_Z + AGE_AT_ANCHOR_YEARS + (1 | MRN)
    }
    fit <- tryCatch(
      glmer(frm, data = sub, family = binomial("logit"),
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)),
            nAGQ = 0),
      error = function(e) NULL
    )
    if (is.null(fit) || isSingular(fit, tol = 1e-4)) next

    co <- summary(fit)$coefficients
    if (!"LR_RESULT_Z" %in% rownames(co)) next

    beta <- co["LR_RESULT_Z", "Estimate"]
    se   <- co["LR_RESULT_Z", "Std. Error"]
    pval <- co["LR_RESULT_Z", "Pr(>|z|)"]

    results_list[[paste(ct, lab, sep = "___")]] <- data.table(
      cancer_type = ct, test = lab,
      estimate = exp(beta), logor = beta,
      lower_ci = exp(beta - 1.96 * se), upper_ci = exp(beta + 1.96 * se),
      p = pval, n_rows = nrow(sub), n_mrn = sub[, uniqueN(MRN)]
    )
  }
}

results_final <- rbindlist(results_list, use.names = TRUE, fill = TRUE)
results_final[, adjusted_p := p.adjust(p, method = "BH")]

out_csv <- file.path(OUT_TAB_DIR, sprintf("table_additionallabs_glmm_z%d_%s_%s.csv", min_patients, OUT_TAG, DATE_STAMP))
fwrite(results_final, out_csv)

# same convention as Fig4C tissue-agnostic signature: mean log2(OR) + 95% CI error bars
mean_or_df <- results_final %>%
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
  mutate(direction = ifelse(mean_log_or > 0, "Enriched in Cachexia", "Down in Cachexia"))

mean_or_df <- mean_or_df %>%
  mutate(
    mean_log2_or = mean_log_or / log(2),
    lower2       = lower       / log(2),
    upper2       = upper       / log(2)
  )

max_abs <- with(mean_or_df, max(abs(c(lower2, upper2)), na.rm = TRUE))

p_bar <- ggplot(
  mean_or_df,
  aes(x = mean_log2_or, y = reorder(test, mean_log2_or), fill = direction)
) +
  geom_col(width = 0.8) +
  geom_segment(
    aes(x = lower2, xend = upper2,
        y = reorder(test, mean_log2_or), yend = reorder(test, mean_log2_or)),
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
  labs(title = "Additional labs", x = "Mean log2(OR)", y = NULL, fill = "Direction") +
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

ggsave(file.path(OUT_PLOT_DIR, paste0("additional_labs_summary_", OUT_TAG, "_", DATE_STAMP, ".pdf")),
       p_bar, width = 4.3, height = 3, device = "pdf")

fwrite(mean_or_df, file.path(OUT_PLOT_DIR, paste0("additional_labs_summary_data_", OUT_TAG, "_", DATE_STAMP, ".csv")))

cat("\n[", OUT_TAG, "] Wrote additional-labs GLMM outputs to:\n  ", OUT_TAB_DIR, "\n  ", OUT_PLOT_DIR, "\n")
