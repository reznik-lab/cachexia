library(data.table)
library(ggplot2)

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

BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots", "fearon_definition")
DATE_STAMP  <- "20260706"
WL_LABEL    <- "WL5_BMIlt20"

FIG3_DIR <- file.path(REV_PLOTS, "Fig3")
dir.create(FIG3_DIR, recursive = TRUE, showWarnings = FALSE)

perm_path <- file.path(REV_RESULTS, "Fig3", paste0("perm_", WL_LABEL, "_", DATE_STAMP))

prog_fp <- file.path(REV_INPUTS, "table_timeline_radiology_cancer_progression_predictions.csv")
msk_fp  <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")

prog     <- fread(prog_fp)
msk_clin <- fread(msk_fp)

prog[, START_DATE := as.IDate(START_DATE)]
msk_clin[, anchor_final := as.IDate(anchor_final)]

prog <- merge(prog, msk_clin[, .(MRN, anchor_final, CANCER_TYPE_DETAILED)], by = "MRN", all.x = TRUE)
prog[, days_since_anchor := as.numeric(START_DATE - anchor_final)]
prog <- prog[!is.na(days_since_anchor)]

safe_name <- function(x) gsub("[^A-Za-z0-9_]", "_", x)
msk_clin[, safe_ct := safe_name(CANCER_TYPE_DETAILED)]

mraw_files <- list.files(
  perm_path,
  pattern = paste0("^Mraw_", WL_LABEL, "_.*_", DATE_STAMP, "\\.csv$"),
  full.names = TRUE
)

cancer_types_safe <- gsub(
  paste0("^Mraw_", WL_LABEL, "_|_", DATE_STAMP, "\\.csv$"),
  "",
  basename(mraw_files)
)

summary_list <- list()

for (i in seq_along(cancer_types_safe)) {
  safe_cancer <- cancer_types_safe[i]

  mfile <- file.path(perm_path, paste0("Mraw_", WL_LABEL, "_", safe_cancer, "_", DATE_STAMP, ".csv"))
  pfile <- file.path(perm_path, paste0("Praw_", WL_LABEL, "_", safe_cancer, "_", DATE_STAMP, ".csv"))
  if (!file.exists(mfile) || !file.exists(pfile)) next

  M_mat <- fread(mfile)
  P_vec <- fread(pfile)$P_vec

  cancer_type <- msk_clin[safe_ct == safe_cancer, unique(CANCER_TYPE_DETAILED)][1]
  mrns_used   <- unique(msk_clin[safe_ct == safe_cancer, MRN])

  prog_sub <- prog[MRN %in% mrns_used & PROGRESSION == "Yes"]
  T_total  <- nrow(prog_sub)
  if (T_total == 0) next

  P_norm    <- sum(P_vec) / T_total
  S_perm    <- colSums(M_mat) / T_total
  null_mean <- mean(S_perm)

  p_val <- mean(abs(S_perm - null_mean) >= abs(P_norm - null_mean))

  summary_list[[cancer_type]] <- data.table(
    Cancer_Type = cancer_type,
    Observed    = P_norm,
    Null_Mean   = null_mean,
    P_Value     = p_val,
    S_perm      = list(S_perm)
  )
}

summary_df <- rbindlist(summary_list, fill = TRUE)
fwrite(summary_df[, .(Cancer_Type, Observed, Null_Mean, P_Value)],
       file.path(REV_RESULTS, "Fig3", paste0("perm_summary_final_", WL_LABEL, "_", DATE_STAMP, ".csv")))

df_obs    <- summary_df[, .(Cancer_Type, Observed, Null_Mean, P_Value)]
df_violin <- summary_df[, .(S_perm = unlist(S_perm)), by = Cancer_Type]

df_obs[,    Code := code_map[as.character(Cancer_Type)]]
df_violin[, Code := code_map[as.character(Cancer_Type)]]
df_obs[is.na(Code),    Code := Cancer_Type]
df_violin[is.na(Code), Code := Cancer_Type]

ord <- df_obs[order(-Observed), Code]
df_obs[,    Code := factor(Code, levels = ord)]
df_violin[, Code := factor(Code, levels = ord)]

global_mean_obs <- df_obs[, mean(Observed, na.rm = TRUE)]

p <- ggplot(df_violin, aes(x = Code, y = S_perm)) +
  geom_violin(aes(fill = "Permutation null"),
              color = "#A9B5AEFF", width = 0.8, scale = "width") +
  geom_point(data = df_obs, aes(x = Code, y = Null_Mean, color = "Null mean"), size = 2.5) +
  geom_point(data = df_obs, aes(x = Code, y = Observed,  color = "Observed"),  size = 2.5) +
  geom_hline(aes(yintercept = global_mean_obs, linetype = "Global mean (observed)"),
             color = "black", linewidth = 0.5, show.legend = TRUE) +
  labs(y = "Normalized progression overlap", x = NULL) +
  scale_fill_manual(name = NULL, values = c("Permutation null" = "#A9B5AEFF")) +
  scale_color_manual(name = NULL, values = c("Observed" = "#FFAE34FF", "Null mean" = "black")) +
  scale_linetype_manual(name = NULL, values = c("Global mean (observed)" = "dashed")) +
  guides(fill = guide_legend(order = 1, override.aes = list(color = NA)),
         color = guide_legend(order = 2),
         linetype = guide_legend(order = 3)) +
  theme_minimal(base_family = "ArialMT") +
  theme(
    legend.position   = "bottom",
    legend.title      = element_blank(),
    legend.text       = element_text(size = 9),
    plot.title        = element_blank(),
    axis.title.y      = element_text(size = 9),
    axis.text.y       = element_text(size = 9),
    axis.text.x       = element_text(size = 9, angle = 90, vjust = 0.5, hjust = 1),
    axis.line         = element_line(color = "black", linewidth = 0.3),
    axis.ticks        = element_line(color = "black", linewidth = 0.3),
    axis.ticks.length = unit(0.15, "cm"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )

plot_width <- max(3.5, 0.27 * length(levels(df_violin$Code)))
ggsave(
  file.path(FIG3_DIR, paste0("Fig3H_progression_summary_violin_", WL_LABEL, "_", DATE_STAMP, ".pdf")),
  p, width = plot_width, height = 3.2, units = "in"
)

cat("\nWrote Fig3H summary violin to:", FIG3_DIR, "\n")
cat("Cancer types summarized:", nrow(df_obs), "\n")
