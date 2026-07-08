library(data.table)
library(ggplot2)
library(future)
library(future.apply)
library(progressr)

code_map <- c(
  "Esophageal Adenocarcinoma" = "ESCA", "Acute Myeloid Leukemia" = "AML",
  "Stomach Adenocarcinoma" = "STAD", "Pancreatic Adenocarcinoma" = "PAAD",
  "Small Cell Lung Cancer" = "SCLC", "Intrahepatic Cholangiocarcinoma" = "IHCH",
  "Uterine Serous Carcinoma/Uterine Papillary Serous Carcinoma" = "USC",
  "High-Grade Serous Ovarian Cancer" = "HGSOC", "Cancer of Unknown Primary" = "CUP",
  "Colon Adenocarcinoma" = "COAD", "Colorectal Adenocarcinoma" = "COADREAD",
  "Rectal Adenocarcinoma" = "READ", "Diffuse Large B-Cell Lymphoma, NOS" = "DLBCLNOS",
  "Lung Squamous Cell Carcinoma" = "LUSC", "Non-Small Cell Lung Cancer" = "NSCLC",
  "Bladder Urothelial Carcinoma" = "BLCA", "Glioblastoma Multiforme" = "GBM",
  "Lung Adenocarcinoma" = "LUAD", "Renal Clear Cell Carcinoma" = "CCRCC",
  "Plasma Cell Myeloma" = "PCM", "Invasive Breast Carcinoma" = "BRCA",
  "Myelodysplastic Workup" = "MDSWP", "Gastrointestinal Stromal Tumor" = "GIST",
  "Breast Invasive Ductal Carcinoma" = "IDC", "Cutaneous Melanoma" = "SKCM",
  "Breast Invasive Lobular Carcinoma" = "ILC", "Uterine Endometrioid Carcinoma" = "UEC",
  "Prostate Adenocarcinoma" = "PRAD", "Follicular Lymphoma" = "FL",
  "Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma" = "CLLSLL"
)

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")

DATE_STAMP <- "20260706"
WL_LABEL   <- "WL5_BMIlt20"  # main Fig3 panel = consensus/WL5+modifier definition

spans_fp <- file.path(REV_RESULTS, paste0("spans_fearon_", WL_LABEL, "_", DATE_STAMP, ".csv"))
prog_fp  <- file.path(REV_INPUTS,  "table_timeline_radiology_cancer_progression_predictions.csv")
msk_fp   <- file.path(REV_INPUTS,  "dx_cohort_metadata_20260126_v2.csv")

out_dir <- file.path(REV_RESULTS, "Fig3", paste0("perm_", WL_LABEL, "_", DATE_STAMP))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

spans    <- fread(spans_fp)
prog     <- fread(prog_fp)
msk_clin <- fread(msk_fp)

spans[, duration := end_day - start_day + 1]

prog[, START_DATE := as.IDate(START_DATE)]
msk_clin[, anchor_final := as.IDate(anchor_final)]

msk_clin[CANCER_TYPE_DETAILED %in% c("Colon Adenocarcinoma", "Rectal Adenocarcinoma"),
      CANCER_TYPE_DETAILED := "Colorectal Adenocarcinoma"]

spans <- merge(spans, msk_clin[, .(MRN, CANCER_TYPE_DETAILED)], by = "MRN", all.x = TRUE)

prog <- merge(prog, msk_clin[, .(MRN, anchor_final)], by = "MRN", all.x = TRUE)
prog[, days_since_anchor := as.numeric(START_DATE - anchor_final)]
prog <- prog[!is.na(days_since_anchor)]

perm_once <- function(spans_df, prog_days) {
  shuffled <- spans_df[sample(.N)]
  shuffled[, start_day := shift(cumsum(duration), fill = 0)]
  shuffled[, end_day   := start_day + duration - 1]
  sum(vapply(
    prog_days,
    function(d) any(d >= shuffled$start_day & d <= shuffled$end_day & shuffled$span == 1),
    logical(1)
  ))
}

safe_name <- function(x) gsub("[^A-Za-z0-9_]", "_", x)

cancer_counts <- spans[, .(n_patients = uniqueN(MRN)), by = CANCER_TYPE_DETAILED]
eligible_cancers <- cancer_counts[n_patients >= 500]$CANCER_TYPE_DETAILED

if ("showtext" %in% loadedNamespaces()) showtext::showtext_auto(FALSE)

plan(multisession, workers = max(1, parallel::detectCores() - 1))
data.table::setDTthreads(1)
RNGkind("L'Ecuyer-CMRG"); set.seed(123)

n_perm <- 1000
results <- list()

handlers(global = TRUE)

for (cancer_type in eligible_cancers) {

  cat("\n", cancer_type, "\n")
  safe_cancer <- safe_name(cancer_type)

  f_Mraw <- file.path(out_dir, paste0("Mraw_", WL_LABEL, "_", safe_cancer, "_", DATE_STAMP, ".csv"))
  f_Praw <- file.path(out_dir, paste0("Praw_", WL_LABEL, "_", safe_cancer, "_", DATE_STAMP, ".csv"))

  spans_sub <- spans[CANCER_TYPE_DETAILED == cancer_type,
                     .(MRN, duration, span, start_day, end_day)]

  prog_sub <- prog[
    MRN %in% spans_sub$MRN & PROGRESSION == "Yes",
    .(MRN, days_since_anchor)
  ]

  mrns <- unique(prog_sub$MRN)
  if (length(mrns) == 0) {
    cat("No progression events for:", cancer_type, "\n")
    next
  }

  spans_by_mrn <- split(spans_sub[, .(duration, span, start_day, end_day)], spans_sub$MRN)
  prog_by_mrn  <- split(prog_sub$days_since_anchor, prog_sub$MRN)

  per_mrn <- with_progress({
    p <- progressor(along = mrns)
    future_lapply(
      mrns,
      function(x) {
        p(sprintf("MRN %s", x))

        spans_mrn <- spans_by_mrn[[as.character(x)]]
        prog_mrn  <- prog_by_mrn[[as.character(x)]]

        ccx <- spans_mrn$span == 1L
        P_i <- sum(vapply(
          prog_mrn,
          function(d) any(d >= spans_mrn$start_day[ccx] & d <= spans_mrn$end_day[ccx]),
          logical(1)
        ))

        M_i <- replicate(n_perm, perm_once(spans_mrn[, .(duration, span)], prog_mrn))
        list(P = P_i, M = M_i)
      },
      future.seed = TRUE
    )
  })

  P_vec <- vapply(per_mrn, `[[`, numeric(1), "P")
  M_mat <- do.call(rbind, lapply(per_mrn, `[[`, "M"))

  T_total <- nrow(prog_sub)
  P_norm  <- sum(P_vec) / T_total
  S_perm  <- colSums(M_mat) / T_total
  null_mean <- mean(S_perm)

  p_val <- mean(abs(S_perm - null_mean) >= abs(P_norm - null_mean))

  cat("Observed normalized overlap =", round(P_norm, 3), "\n")
  cat("Permutation p-value =", p_val, "\n")

  fwrite(as.data.table(M_mat), f_Mraw)
  fwrite(data.table(P_vec = P_vec), f_Praw)

  df_hist <- data.frame(S_perm = S_perm)
  ph <- ggplot(df_hist, aes(x = S_perm)) +
    geom_histogram(aes(y = after_stat(density)),
                   bins = 50, fill = "#A9B5AEFF", color = "#A9B5AEFF") +
    geom_density(color = "#8CC2CAFF", linewidth = 1, alpha = 0.6) +
    geom_vline(xintercept = P_norm, color = "#FFAE34FF", linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = null_mean, color = "black", linetype = "dotted") +
    labs(
      title = paste0("Permutation Test - ", cancer_type),
      subtitle = paste0("p = ", signif(p_val, 3), ", observed = ", round(P_norm, 3)),
      x = "Normalized progression overlap",
      y = "Density"
    ) +
    scale_y_continuous(expand = c(0, 0)) +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title   = element_text(hjust = 0.5, size = 9),
      plot.subtitle = element_text(hjust = 0.5, size = 8),
      axis.title.x = element_text(size = 9),
      axis.title.y = element_text(size = 9),
      axis.text.x  = element_text(size = 9),
      axis.line    = element_line(color = "black", linewidth = 0.2),
      axis.ticks.x = element_line(color = "black", linewidth = 0.3),
      axis.ticks.y = element_line(color = "black", linewidth = 0.3),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )

  ggsave(
    file.path(out_dir, paste0(safe_cancer, "_permutation_histogram_", WL_LABEL, "_", DATE_STAMP, ".pdf")),
    ph, width = 2.5, height = 2.5, units = "in"
  )

  results[[cancer_type]] <- list(P_norm = P_norm, p_val = p_val, null_mean = null_mean, S_perm = S_perm)
}

summary_df <- data.table(
  CANCER_TYPE = names(results),
  P_norm      = sapply(results, function(x) x$P_norm),
  Null_Mean   = sapply(results, function(x) x$null_mean),
  p_val       = sapply(results, function(x) x$p_val)
)

fwrite(summary_df, file.path(out_dir, paste0("perm_summary_", WL_LABEL, "_", DATE_STAMP, ".csv")))

cat("\nWrote permutation test outputs to:", out_dir, "\n")
