library(ggplot2)
library(dplyr)
library(tidyr)
library(data.table)
library(grid)
library(ggrepel)
library(stringr)


BASE_REV   <- "."
REV_TABLES <- file.path(BASE_REV, "rev_tables")
REV_PLOTS  <- file.path(BASE_REV, "rev_plots")
DATE_STAMP <- "20260706"

in_fp <- file.path(REV_TABLES, "serology_glmm", paste0("WL5_", DATE_STAMP),
                   paste0("glmm_results_labs_5_", DATE_STAMP, ".csv"))

out_fig_dir <- file.path(REV_PLOTS, "fearon_definition", "Fig4")
out_tab_dir <- file.path(REV_TABLES, "serology_glmm", paste0("WL5_", DATE_STAMP))
dir.create(out_fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_tab_dir, recursive = TRUE, showWarnings = FALSE)

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


glmm_labs <- read.csv(in_fp, stringsAsFactors = FALSE)
glmm_labs$cancer_type <- trimws(glmm_labs$cancer_type)

glmm_labs <- glmm_labs %>%
  mutate(
    OR = estimate,
    log_or_use = dplyr::coalesce(log_or, logor),
    p_adj = dplyr::coalesce(adjusted_p, p.adjust(p, method = "fdr")),
    log2_or = log_or_use / log(2),
    log_p = -log10(p_adj),
    cancer_code = unname(code_map[cancer_type]),
    label = paste0(cancer_code, ": ", test)
  )

max_logp <- max(glmm_labs$log_p[is.finite(glmm_labs$log_p)], na.rm = TRUE)
glmm_labs$log_p[!is.finite(glmm_labs$log_p)] <- max_logp + 5

glmm_labs <- glmm_labs %>%
  mutate(
    group = case_when(
      p_adj < 0.05 & log2_or > 0 ~ "Enriched in Cachexia",
      p_adj < 0.05 & log2_or < 0 ~ "Down in Cachexia",
      TRUE ~ "NS"
    ),
    group = factor(group, levels = c("Down in Cachexia", "Enriched in Cachexia", "NS"))
  )

sig_all <- glmm_labs %>%
  filter(p_adj < 0.05, !is.na(log2_or), !is.na(log_p), !is.na(label), !is.na(cancer_code))

sig_up   <- sig_all %>% filter(group == "Enriched in Cachexia")
sig_down <- sig_all %>% filter(group == "Down in Cachexia")

N_PER_SIDE <- 5

pick_spread_y <- function(df, n = 7, ycol = "log_p") {
  if (nrow(df) == 0) return(df)

  y <- df[[ycol]]
  probs <- seq(0, 1, length.out = n + 1)
  qs <- unique(stats::quantile(y, probs = probs, na.rm = TRUE, names = FALSE, type = 7))

  if (length(qs) >= 3) {
    df$ybin <- cut(y, breaks = qs, include.lowest = TRUE, labels = FALSE)
  } else {
    df$ybin <- cut(y, breaks = n, include.lowest = TRUE, labels = FALSE)
  }

  picked <- df[0, ]
  used_cancers <- character()
  used_tests <- character()

  for (b in sort(unique(df$ybin))) {
    pool <- df[df$ybin == b, , drop = FALSE]
    if (nrow(pool) == 0) next

    ymed <- stats::median(pool[[ycol]], na.rm = TRUE)
    pool <- pool %>%
      mutate(dist_to_bin = abs(.data[[ycol]] - ymed)) %>%
      arrange(dist_to_bin, p_adj, desc(abs(log2_or))) %>%
      distinct(cancer_code, test, .keep_all = TRUE)

    chosen <- NULL
    for (i in seq_len(nrow(pool))) {
      r <- pool[i, ]
      new_cancer <- !(r$cancer_code %in% used_cancers)
      new_test   <- !(r$test %in% used_tests)
      if (new_cancer || new_test) { chosen <- r; break }
    }
    if (is.null(chosen)) chosen <- pool[1, ]

    picked <- bind_rows(picked, chosen)
    used_cancers <- c(used_cancers, chosen$cancer_code)
    used_tests   <- c(used_tests, chosen$test)

    if (nrow(picked) >= n) break
  }

  if (nrow(picked) < n) {
    fill <- df %>%
      anti_join(picked, by = c("cancer_code", "test")) %>%
      arrange(p_adj, desc(abs(log2_or))) %>%
      slice_head(n = n - nrow(picked))
    picked <- bind_rows(picked, fill)
  }

  picked
}

sig_labels <- bind_rows(
  pick_spread_y(sig_up,   n = N_PER_SIDE),
  pick_spread_y(sig_down, n = N_PER_SIDE)
)

p_volcano <- ggplot(
  glmm_labs %>% filter(!is.na(log2_or), !is.na(log_p)),
  aes(x = log2_or, y = log_p, color = group)
) +
  geom_point(alpha = 0.7, size = 2.0) +
  geom_hline(yintercept = -log10(0.05), linetype = "dotted", color = "gray50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  ggrepel::geom_text_repel(
    data = sig_labels,
    aes(label = label),
    size = 2.6,
    max.overlaps = Inf,
    force = 20,
    force_pull = 0.3,
    max.time = 3,
    max.iter = 200000,
    box.padding = 0.6,
    point.padding = 0.4,
    min.segment.length = 0,
    segment.color = "gray30",
    segment.size  = 0.4,
    seed = 42
  ) +
  scale_color_manual(values = c(
    "Enriched in Cachexia" = "#EF6F6AFF",
    "Down in Cachexia"     = "#6388B4FF",
    "NS"                   = "gray75"
  )) +
  scale_x_continuous(limits = c(-1.5, 1.5), breaks = seq(-1.5, 1.5, 0.5)) +
  labs(
    title = "Lab Profiles in Cachexia",
    x = expression(-log[2]("OR")),
    y = expression(-log[10]("FDR")),
    color = "Direction"
  ) +
  coord_cartesian(ylim = c(0, max(glmm_labs$log_p, na.rm = TRUE) + 2)) +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_text(hjust = 0.5, size = 9),
    axis.title.x = element_text(size = 9),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    legend.text  = element_text(size = 9),
    legend.title = element_text(size = 9),
    axis.line    = element_line(color = "black", linewidth = 0.2),
    panel.grid   = element_blank(),
    axis.ticks   = element_line(color = "black", linewidth = 0.3),
    legend.position = "top"
  )

ggsave(file.path(out_fig_dir, paste0("Fig4B_volcano_", DATE_STAMP, ".pdf")), p_volcano, width = 3.7, height = 3.6)

si_table <- glmm_labs %>%
  transmute(
    cancer_code, test, OR = OR,
    CI_low  = lower_ci, CI_high = upper_ci,
    log2_or = log2_or, p = p, FDR = p_adj
  ) %>%
  arrange(cancer_code, test) %>%
  mutate(
    OR = round(OR, 3), CI_low = round(CI_low, 3), CI_high = round(CI_high, 3),
    log2_or = round(log2_or, 2), p = signif(p, 3), FDR = signif(FDR, 3),
    `OR (95% CI)` = sprintf("%.3f (%.3f-%.3f)", OR, CI_low, CI_high)
  ) %>%
  select(cancer_code, test, `OR (95% CI)`, log2_or, p, FDR)

fwrite(si_table, file.path(out_tab_dir, paste0("STable_glmm_labs_5WL_", DATE_STAMP, ".csv")))


glmm_labs2 <- read.csv(in_fp, stringsAsFactors = FALSE) %>%
  mutate(OR = estimate, log_or_use = dplyr::coalesce(log_or, logor))

mean_or_df <- glmm_labs2 %>%
  group_by(test) %>%
  summarise(
    mean_log_or = mean(log_or_use, na.rm = TRUE),
    n_eff       = sum(!is.na(log_or_use)),
    se          = sd(log_or_use, na.rm = TRUE) / sqrt(pmax(n_eff, 1)),
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

p_signature <- ggplot(
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
  labs(title = "Tissue-agnostic signature", x = "Mean log2(OR)", y = NULL, fill = "Direction") +
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

ggsave(file.path(out_fig_dir, paste0("Fig4C_tissue_agnostic_signature_", DATE_STAMP, ".pdf")),
       p_signature, width = 4.3, height = 5, device = "pdf")

fwrite(mean_or_df, file.path(out_fig_dir, paste0("Fig4C_tissue_agnostic_signature_data_", DATE_STAMP, ".csv")))

cat("\nWrote Fig4B (volcano) and Fig4C (tissue-agnostic signature) to:", out_fig_dir, "\n")
