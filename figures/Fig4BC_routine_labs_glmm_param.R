library(future.apply)
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)
library(patchwork)
library(lme4)
library(stringr)
library(grid)
library(progressr)

options(future.rng.onMisuse = "ignore")


args <- commandArgs(trailingOnly = TRUE)
WL_LABEL     <- args[1]
STAGE_FILTER <- args[2]
stopifnot(WL_LABEL %in% c("WL5_BMIlt20", "WL10", "WL15"))
stopifnot(STAGE_FILTER %in% c("ALL", "Stage 1-3", "Stage 4"))

OUT_TAG <- if (STAGE_FILTER == "ALL") WL_LABEL else paste0(WL_LABEL, "_", gsub(" ", "", STAGE_FILTER))

BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_TABLES  <- file.path(BASE_REV, "rev_tables")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots", "fearon_definition")
DATE_STAMP  <- "20260706"

spans_lab_fp <- file.path(REV_RESULTS, "serology_spans_labs",
                           paste0("spans_labtests_spans_fearon_", WL_LABEL, "_", DATE_STAMP, "_", DATE_STAMP, ".csv"))
msk_fp <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")

OUT_TAB_DIR  <- file.path(REV_TABLES, "serology_glmm", paste0(OUT_TAG, "_", DATE_STAMP))
OUT_PLOT_DIR <- file.path(REV_PLOTS, "SFig4")
dir.create(OUT_TAB_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

msk_clin <- fread(msk_fp)

spans_labtests <- fread(spans_lab_fp)

if (STAGE_FILTER != "ALL") {
  spans_labtests <- spans_labtests[STAGE_CDM_DERIVED == STAGE_FILTER]
}

spans_labtests <- spans_labtests %>%
  mutate(
    MRN = as.factor(MRN),
    span = as.factor(span),
    GENDER = as.factor(GENDER),
    CANCER_TYPE_DETAILED = as.factor(CANCER_TYPE_DETAILED),
    AGE_AT_ANCHOR_YEARS = as.numeric(AGE_AT_ANCHOR_YEARS),
    AGE_AT_ANCHOR       = as.numeric(AGE_AT_ANCHOR)
  )

spans_labtests <- spans_labtests %>%
  arrange(MRN, start_day) %>%
  group_by(MRN) %>%
  mutate(SpanID = cumsum(!duplicated(start_day))) %>%
  ungroup()

labs2use <- c(
  'HGB', 'HCT', 'RBC', 'MCV', 'MCH', 'WBC', 'Platelets', 'MCHC', 'RDW', 'Neut',
  'Creatinine', 'CO2', 'Glucose', 'Sodium', 'Chloride', 'BUN', 'Calcium', 'Potassium',
  'Lymph', 'Mono', 'Eos', 'Baso', 'Albumin', 'ALK', 'ALT', 'Bilirubin, Total',
  'Protein, Total', 'AST', 'Nucleated RBC', 'Immature Granulocyte'
)

ccounts <- msk_clin %>%
  select(MRN, CANCER_TYPE_DETAILED) %>%
  distinct() %>%
  group_by(CANCER_TYPE_DETAILED) %>%
  summarise(count = n(), .groups = 'drop') %>%
  arrange(desc(count))

c2use <- ccounts %>% filter(count >= 500)
spans_labtests <- spans_labtests %>%
  filter(CANCER_TYPE_DETAILED %in% c2use$CANCER_TYPE_DETAILED)

n_workers <- max(1, parallel::detectCores() - 1)
future::plan(future::multisession, workers = n_workers)

cts <- as.character(c2use$CANCER_TYPE_DETAILED)

fit_one_cancer <- function(cancer_type, spans_labtests, labs2use) {
  dt <- spans_labtests %>% dplyr::filter(CANCER_TYPE_DETAILED == cancer_type)
  if (nrow(dt) < 2) return(NULL)

  results <- data.frame(
    test = character(), cancer_type = character(),
    estimate = numeric(), logor = numeric(),
    lower_ci = numeric(), upper_ci = numeric(),
    p = numeric(), stringsAsFactors = FALSE
  )

  for (test_name in labs2use) {
    current_data <- dt %>%
      dplyr::filter(!is.na(.data[[test_name]])) %>%
      dplyr::mutate(result = as.numeric(.data[[test_name]]))

    if (nrow(current_data) <= 1) next

    unique_sex <- length(unique(current_data$GENDER))
    formula <- if (unique_sex > 1) {
      as.formula("span ~ result + AGE_AT_ANCHOR_YEARS + GENDER + (1 | MRN)")
    } else {
      as.formula("span ~ result + AGE_AT_ANCHOR_YEARS + (1 | MRN)")
    }

    model <- tryCatch(
      glmer(
        formula, data = current_data, family = binomial(link = "logit"),
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
      ),
      error = function(e) NULL
    )

    if (is.null(model) || isSingular(model, tol = 1e-5)) next

    cs <- summary(model)$coefficients
    if (!("result" %in% rownames(cs))) next

    log_or <- cs["result", "Estimate"]
    se     <- cs["result", "Std. Error"]
    pval   <- cs["result", "Pr(>|z|)"]

    results <- rbind(
      results,
      data.frame(
        test = test_name, cancer_type = cancer_type,
        estimate = exp(log_or), logor = log_or,
        lower_ci = exp(log_or - 1.96 * se),
        upper_ci = exp(log_or + 1.96 * se),
        p = pval, stringsAsFactors = FALSE
      )
    )
  }
  results
}

CKPT_DIR <- file.path(OUT_TAB_DIR, "checkpoints")
dir.create(CKPT_DIR, recursive = TRUE, showWarnings = FALSE)
safe_ct_name <- function(x) gsub("[^A-Za-z0-9_]", "_", x)

cts_todo <- cts[!file.exists(file.path(CKPT_DIR, paste0(safe_ct_name(cts), ".csv")))]
cat(sprintf("[%s] Cancer types already checkpointed: %d / %d\n", OUT_TAG, length(cts) - length(cts_todo), length(cts)))

handlers("txtprogressbar")

if (length(cts_todo) > 0) {
  with_progress({
    p <- progressor(along = cts_todo)
    invisible(future_lapply(
      cts_todo,
      function(ct) {
        out <- fit_one_cancer(ct, spans_labtests, labs2use)
        if (is.null(out)) out <- data.frame()
        fwrite(out, file.path(CKPT_DIR, paste0(safe_ct_name(ct), ".csv")))
        p(sprintf("%s (%s rows)", ct, nrow(out)))
        NULL
      },
      future.seed = TRUE
    ))
  })
}

safe_read_csv <- function(f, tries = 5) {
  for (i in seq_len(tries)) {
    result <- tryCatch(fread(f), error = function(e) NULL)
    if (!is.null(result)) return(result)
    result <- tryCatch(as.data.table(read.csv(f, stringsAsFactors = FALSE)), error = function(e) NULL)
    if (!is.null(result)) return(result)
    Sys.sleep(0.5 * i)
  }
  stop("Could not read checkpoint file after ", tries, " attempts: ", f)
}

res_list <- lapply(cts, function(ct) {
  f <- file.path(CKPT_DIR, paste0(safe_ct_name(ct), ".csv"))
  if (file.exists(f)) safe_read_csv(f) else NULL
})

results <- dplyr::bind_rows(res_list) %>% dplyr::mutate(index = dplyr::row_number())
results$log_or <- log(results$estimate)
results$adjusted_p <- p.adjust(results$p, method = "BH")

out_results_csv <- file.path(OUT_TAB_DIR, paste0("glmm_results_labs_", OUT_TAG, "_", DATE_STAMP, ".csv"))
write.csv(results, out_results_csv, row.names = FALSE)

summary_results <- results %>%
  group_by(test) %>%
  summarise(mean_log_or = mean(log_or, na.rm = TRUE), .groups = "drop") %>%
  arrange(mean_log_or)

p_bar <- ggplot(summary_results, aes(x = reorder(test, mean_log_or), y = mean_log_or, fill = mean_log_or)) +
  geom_bar(stat = "identity", width = 0.6) +
  scale_fill_gradient2(low = "#2c7bb6", mid = "white", high = "#d7191c", midpoint = 0) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
  coord_flip() +
  labs(x = NULL, y = "Mean Log Odds Ratio", title = paste0("Routine labs (", OUT_TAG, ")")) +
  theme_minimal(base_family = "ArialMT") +
  theme(legend.position = "none", plot.title = element_text(size = 9, hjust = 0.5))

ggsave(file.path(OUT_PLOT_DIR, paste0("routine_labs_summary_", OUT_TAG, "_", DATE_STAMP, ".pdf")),
       p_bar, width = 5, height = 6, dpi = 300)

cat("\n[", OUT_TAG, "] Wrote routine-labs GLMM outputs to:\n  ", OUT_TAB_DIR, "\n  ", OUT_PLOT_DIR, "\n")
