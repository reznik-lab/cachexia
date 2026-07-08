library(data.table)
library(ggplot2)
library(grid)
library(patchwork)

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
REV_PLOTS   <- file.path(BASE_REV, "rev_plots", "fearon_definition")
DATE_STAMP  <- "20260706"

SFIG3_DIR <- file.path(REV_PLOTS, "SFig3")
dir.create(SFIG3_DIR, recursive = TRUE, showWarnings = FALSE)

WL_LIST <- c("WL10", "WL15")

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

summarize_perm <- function(WL_LABEL) {
  perm_path <- file.path(REV_RESULTS, "Fig3", paste0("perm_", WL_LABEL, "_", DATE_STAMP))
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
      WL = WL_LABEL, Cancer_Type = cancer_type,
      Observed = P_norm, Null_Mean = null_mean, P_Value = p_val,
      S_perm = list(S_perm)
    )
  }
  rbindlist(summary_list, fill = TRUE)
}

summary_all <- rbindlist(lapply(WL_LIST, summarize_perm), fill = TRUE)

fwrite(summary_all[, .(WL, Cancer_Type, Observed, Null_Mean, P_Value)],
       file.path(REV_RESULTS, "Fig3", paste0("perm_summary_WL10_WL15_combined_", DATE_STAMP, ".csv")))

df_obs    <- summary_all[, .(WL, Cancer_Type, Observed, Null_Mean, P_Value)]
df_violin <- summary_all[, .(S_perm = unlist(S_perm)), by = .(WL, Cancer_Type)]

df_obs[,    Code := code_map[as.character(Cancer_Type)]]
df_violin[, Code := code_map[as.character(Cancer_Type)]]
df_obs[is.na(Code),    Code := Cancer_Type]
df_violin[is.na(Code), Code := Cancer_Type]

ord10 <- df_obs[WL == "WL10"][order(-Observed), unique(Code)]

df_obs    <- df_obs[Code %in% ord10]
df_violin <- df_violin[Code %in% ord10]

df_obs[,    Code := factor(Code, levels = ord10)]
df_violin[, Code := factor(Code, levels = ord10)]

global_df <- df_obs[, .(global_mean_obs = mean(Observed, na.rm = TRUE)), by = WL]

base_theme <- theme_minimal(base_family = "ArialMT") +
  theme(
    legend.position    = "bottom",
    legend.title       = element_blank(),
    legend.text        = element_text(size = 8),
    strip.text         = element_text(size = 9),
    axis.text.y        = element_text(size = 9),
    axis.text.x        = element_text(size = 7.5),
    axis.title.x       = element_text(size = 9),
    axis.line          = element_line(color = "black", linewidth = 0.3),
    axis.ticks         = element_line(color = "black", linewidth = 0.3),
    axis.ticks.length  = unit(0.15, "cm"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank()
  )

col_scales <- list(
  scale_fill_manual(name = NULL, values = c("Permutation null" = "#A9B5AEFF")),
  scale_color_manual(name = NULL, values = c("Observed" = "#FFAE34FF", "Null mean" = "black")),
  scale_linetype_manual(name = NULL, values = c("Global mean" = "dashed")),
  guides(fill = guide_legend(order = 1, override.aes = list(color = NA), nrow = 2),
         color = guide_legend(order = 2, nrow = 2), linetype = guide_legend(order = 3, nrow = 2))
)

make_panel <- function(wl_label, n_breaks, show_y_text) {
  dv <- df_violin[WL == wl_label]
  do <- df_obs[WL == wl_label]
  gd <- global_df[WL == wl_label]

  p <- ggplot(dv, aes(x = S_perm, y = Code)) +
    geom_violin(aes(fill = "Permutation null"),
                color = "#A9B5AEFF", width = 0.8, scale = "width") +
    geom_point(data = do, aes(x = Null_Mean, y = Code, color = "Null mean"), size = 2.2) +
    geom_point(data = do, aes(x = Observed,  y = Code, color = "Observed"),  size = 2.2) +
    geom_vline(
      data = gd,
      aes(xintercept = global_mean_obs, linetype = "Global mean"),
      color = "black", linewidth = 0.5
    ) +
    facet_grid(. ~ WL) +
    scale_x_continuous(breaks = scales::breaks_pretty(n = n_breaks)) +
    labs(x = NULL, y = NULL) +
    col_scales +
    base_theme

  if (!show_y_text) p <- p + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
  p
}

p_wl10 <- make_panel("WL10", n_breaks = 5, show_y_text = TRUE)
p_wl15 <- make_panel("WL15", n_breaks = 3, show_y_text = FALSE)

p_combined <- (p_wl10 + p_wl15 + plot_layout(widths = c(1, 0.7), guides = "collect")) &
  theme(legend.position = "bottom")

p_combined <- p_combined + plot_annotation(
  caption = "Normalized progression overlap",
  theme = theme(plot.caption = element_text(size = 9, hjust = 0.5, family = "ArialMT"))
)

out_pdf <- file.path(SFIG3_DIR, paste0("sfig3C_progression_summary_violin_WL10_WL15_", DATE_STAMP, ".pdf"))
ggsave(out_pdf, p_combined, width = 4.8, height = 4.3, units = "in")

cat("\nWrote SFig3C combined WL10/WL15 summary violin to:", out_pdf, "\n")
