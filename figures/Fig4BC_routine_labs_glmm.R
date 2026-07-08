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

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_TABLES  <- file.path(BASE_REV, "rev_tables")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots", "fearon_definition")
DATE_STAMP  <- "20260706"

spans_lab_fp <- file.path(REV_RESULTS, "serology_spans_labs",
                           paste0("spans_labtests_spans_fearon_WL5_BMIlt20_", DATE_STAMP, "_", DATE_STAMP, ".csv"))
msk_fp <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")

msk_clin <- fread(msk_fp)

OUT_TAB_DIR  <- file.path(REV_TABLES, "serology_glmm", paste0("WL5_", DATE_STAMP))
OUT_PLOT_DIR <- file.path(REV_PLOTS, "Fig4")
dir.create(OUT_TAB_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

spans_labtests <- fread(spans_lab_fp)

spans_labtests <- spans_labtests %>%
  mutate(
    MRN = as.factor(MRN),
    span = as.factor(span),
    GENDER = as.factor(GENDER),
    CANCER_TYPE_DETAILED = as.factor(CANCER_TYPE_DETAILED)
  )

spans_labtests <- spans_labtests %>%
  arrange(MRN, start_day) %>%
  group_by(MRN) %>%
  mutate(SpanID = cumsum(!duplicated(start_day))) %>%
  ungroup()

# AGE_AT_ANCHOR_YEARS / AGE_AT_ANCHOR already came in via the spans_labs.py merge
# (unlike the 0303 original, which needed this separate join)
spans_labtests <- spans_labtests %>%
  mutate(
    AGE_AT_ANCHOR_YEARS = as.numeric(AGE_AT_ANCHOR_YEARS),
    AGE_AT_ANCHOR       = as.numeric(AGE_AT_ANCHOR)
  )

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

  dt <- spans_labtests %>%
    dplyr::filter(CANCER_TYPE_DETAILED == cancer_type)

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
        p = pval,
        stringsAsFactors = FALSE
      )
    )
  }

  results
}

# per-cancer-type checkpointing: a kill mid-run only loses whatever cancer
# type(s) were still in flight, not the whole job. Rerunning resumes by
# skipping cancer types whose checkpoint already exists.
CKPT_DIR <- file.path(OUT_TAB_DIR, "checkpoints")
dir.create(CKPT_DIR, recursive = TRUE, showWarnings = FALSE)
safe_ct_name <- function(x) gsub("[^A-Za-z0-9_]", "_", x)

cts_todo <- cts[!file.exists(file.path(CKPT_DIR, paste0(safe_ct_name(cts), ".csv")))]
cat(sprintf("Cancer types already checkpointed: %d / %d\n", length(cts) - length(cts_todo), length(cts)))

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

# reads on this OneDrive-synced path intermittently fail (mmap errors, or
# transient "file not fully hydrated" hiccups) even though the file is valid -
# retry a few times with short pauses, alternating fread/read.csv, before giving up
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

out_results_csv <- file.path(OUT_TAB_DIR, paste0("glmm_results_labs_5_", DATE_STAMP, ".csv"))
write.csv(results, out_results_csv, row.names = FALSE)

results <- fread(out_results_csv)

# ------------------ Fig4B: per-cancer x per-lab heatmap ------------------ #

log_or_limits <- c(-1, 0.7)
results$log_or <- log(results$estimate)
results$log_or_capped <- pmax(pmin(results$log_or, log_or_limits[2]), log_or_limits[1])
results$adjusted_p <- p.adjust(results$p, method = "BH")
adjusted_p_value_threshold <- 0.05

results <- results %>%
  mutate(test = fct_relevel(test, sort(unique(test))))

p1 <- ggplot(results, aes(x = test, y = fct_rev(cancer_type))) +
  geom_point(aes(size = abs(log_or_capped), color = log_or_capped)) +
  geom_point(
    data = results[results$adjusted_p < adjusted_p_value_threshold, ],
    aes(x = test, y = fct_rev(cancer_type), size = abs(log_or_capped)),
    shape = 21, color = "black", stroke = 0.5
  ) +
  scale_size_continuous(range = c(1, 10)) +
  scale_color_gradient2(low = "#2c7bb6", mid = "white", high = "#d7191c",
                        midpoint = 0, limits = log_or_limits) +
  labs(x = "Serological Test", y = "Cancer Type", color = "Log Odds Ratio") +
  guides(size = "none") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y = element_text(size = 16, face = "bold"),
    axis.text.x  = element_text(angle = 90, hjust = 1, size = 12),
    axis.text.y  = element_text(size = 12),
    axis.line.x  = element_line(color = "black"),
    axis.line.y  = element_line(color = "black"),
    panel.grid   = element_blank(),
    legend.position = "right"
  )

ggsave(
  file = file.path(OUT_PLOT_DIR, paste0("Fig4B_labvalues_heatmap_", DATE_STAMP, ".pdf")),
  p1, width = 14, height = 12
)

# ------------------ Fig4C: tissue-agnostic mean log2OR summary ------------------ #

log_or_limits2 <- c(-1.2, 0.7)
results <- results %>% mutate(test = factor(test, levels = unique(results$test)))
results$log_or_capped <- pmax(pmin(results$log_or, log_or_limits2[2]), log_or_limits2[1])

summary_results <- results %>%
  group_by(test) %>%
  summarise(mean_log_or = mean(log_or, na.rm = TRUE), .groups = "drop") %>%
  mutate(test = factor(test, levels = levels(results$test)))
summary_results$mean_log_or <- pmax(pmin(summary_results$mean_log_or, log_or_limits2[2]), log_or_limits2[1])

heatmap_plot <- ggplot(results, aes(x = test, y = fct_rev(cancer_type))) +
  geom_point(aes(size = abs(log_or_capped), color = log_or_capped)) +
  geom_point(
    data = results[results$adjusted_p < adjusted_p_value_threshold, ],
    aes(x = test, y = fct_rev(cancer_type), size = abs(log_or_capped)),
    shape = 21, color = "black", stroke = 0.5
  ) +
  scale_size_continuous(
    name = "abs(log OR)",
    breaks = c(0.25, 0.5, 0.75, 0.9),
    labels = c("0.25", "0.50", "0.75", "> 0.9"),
    range = c(1, 10),
    guide = guide_legend(override.aes = list(color = "gray"))
  ) +
  scale_color_gradient2(
    name = "Log OR",
    low = "#2c7bb6", mid = "white", high = "#d7191c",
    midpoint = 0, limits = log_or_limits2
  ) +
  labs(x = NULL, y = "Cancer Type") +
  theme_minimal() +
  theme(
    axis.title.y = element_text(size = 16, face = "bold"),
    axis.text.x  = element_blank(),
    axis.text.y  = element_text(size = 12),
    panel.grid   = element_blank(),
    axis.line.x  = element_blank(),
    axis.line.y  = element_blank(),
    legend.position = "right"
  )

bar_plot <- ggplot(summary_results, aes(x = test, y = mean_log_or, fill = mean_log_or)) +
  geom_bar(stat = "identity", width = 0.6) +
  scale_fill_gradient2(low = "#2c7bb6", mid = "white", high = "#d7191c",
                       midpoint = 0, limits = log_or_limits2) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed", linewidth = 1) +
  labs(x = "Serological Test", y = "Mean Log Odds Ratio") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y = element_blank(),
    axis.text.x  = element_text(angle = 90, hjust = 1, size = 12),
    panel.grid   = element_blank(),
    axis.line.x  = element_blank(),
    axis.line.y  = element_blank(),
    legend.position = "none"
  )

final_plot <- heatmap_plot / bar_plot + plot_layout(heights = c(5, 1))
ggsave(
  filename = file.path(OUT_PLOT_DIR, paste0("Fig4C_labvalues_summary_", DATE_STAMP, ".pdf")),
  plot = final_plot, width = 14, height = 10, dpi = 300
)

# ------------------ lab test coverage bar chart ------------------ #

lab_categories <- data.frame(
  Lab_Test = c(
    'HGB','HCT','RBC','WBC','Platelets','MCH','MCHC','MCV','RDW',
    'Neut','Abs Neut','Abs Lymph','Eos','Lymph','Mono','Baso',
    'Abs Baso','Abs Eos','Nucleated RBC','Immature Granulocyte',
    'Albumin','ALK','ALT','AST','Bilirubin, Total','Protein, Total',
    'Creatinine','CO2','Glucose','Sodium','Chloride','BUN','Calcium','Potassium'
  ),
  Category = c(rep("Complete Blood Count", 20), rep("Comprehensive Metabolic Panel", 14))
)

lab_counts <- spans_labtests %>%
  select(MRN, all_of(labs2use)) %>%
  pivot_longer(cols = -MRN, names_to = "Lab_Test", values_to = "Value") %>%
  filter(!is.na(Value)) %>%
  group_by(Lab_Test) %>%
  summarise(Patient_Count = n_distinct(MRN), .groups = "drop") %>%
  arrange(desc(Patient_Count)) %>%
  left_join(lab_categories, by = "Lab_Test")

cbc_color <- "gray70"
cmp_color <- "gray30"

p_counts <- ggplot(lab_counts, aes(x = reorder(Lab_Test, -Patient_Count), y = Patient_Count, fill = Category)) +
  geom_bar(stat = "identity") +
  facet_wrap(~Category, scales = "free_x", nrow = 1) +
  scale_fill_manual(values = c("Complete Blood Count" = cbc_color,
                               "Comprehensive Metabolic Panel" = cmp_color)) +
  labs(
    x = "Routinely Done Serological Test",
    y = "# of Patients",
    title = "Serological Tests"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 90, hjust = 1, size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    strip.text = element_text(face = "bold", size = 12),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(color = "black", linewidth = 1),
    axis.line.y = element_line(color = "black", linewidth = 1),
    legend.position = "none"
  ) +
  scale_y_continuous(breaks = seq(0, max(lab_counts$Patient_Count, na.rm = TRUE), by = 5000))

ggsave(
  filename = file.path(OUT_PLOT_DIR, paste0("MSK_lab_tests_", DATE_STAMP, ".png")),
  plot = p_counts, width = 10, height = 6
)

# ------------------ rankings ------------------ #

cancer_ranking <- results %>%
  group_by(cancer_type) %>%
  summarise(
    mean_log_or = mean(abs(log_or), na.rm = TRUE),
    sig_count = sum(adjusted_p < 0.05, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(sig_count), desc(mean_log_or))

test_ranking <- results %>%
  group_by(test) %>%
  summarise(
    mean_log_or = mean(abs(log_or), na.rm = TRUE),
    sig_count = sum(adjusted_p < 0.05, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(sig_count), desc(mean_log_or))

fwrite(cancer_ranking, file.path(OUT_TAB_DIR, paste0("cancer_ranking_", DATE_STAMP, ".csv")))
fwrite(test_ranking,   file.path(OUT_TAB_DIR, paste0("test_ranking_", DATE_STAMP, ".csv")))
fwrite(ccounts,         file.path(OUT_TAB_DIR, paste0("cancer_counts_", DATE_STAMP, ".csv")))

cat("\nWrote routine-labs GLMM outputs to:\n  ", OUT_TAB_DIR, "\n  ", OUT_PLOT_DIR, "\n")
