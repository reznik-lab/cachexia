library(data.table)
library(ggplot2)

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV   <- "."
REV_INPUTS <- file.path(BASE_REV, "rev_inputs")
REV_PLOTS  <- file.path(BASE_REV, "rev_plots", "fearon_definition")
DATE_STAMP <- "20260706"

PLOT_ROOT <- file.path(REV_PLOTS, "SI_Fig1")
dir.create(PLOT_ROOT, recursive = TRUE, showWarnings = FALSE)

msk_clin <- fread(file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv"))
msk_clin[, MRN := as.integer(MRN)]

msk_clin[
  CANCER_TYPE_DETAILED %chin% c("Colon Adenocarcinoma", "Rectal Adenocarcinoma"),
  CANCER_TYPE_DETAILED := "Colorectal Adenocarcinoma"
]

if ("showtext" %in% loadedNamespaces()) showtext::showtext_auto(FALSE)

# ------------------ SFig1B: cancer-type distribution (N>=500) in full cohort ------------------ #
# purely cohort demographics (CANCER_TYPE_DETAILED counts) - independent of cachexia
# episode definition, so unaffected by the smoothed-BMI correction

cancer_data <- msk_clin[!is.na(CANCER_TYPE_DETAILED) & nzchar(trimws(CANCER_TYPE_DETAILED))]
valid_counts <- unique(cancer_data[, .(MRN, CANCER_TYPE_DETAILED)])[
  , .(N = uniqueN(MRN)), by = CANCER_TYPE_DETAILED
][N >= 500][order(N)]

valid_counts[, CANCER_TYPE_DETAILED := factor(CANCER_TYPE_DETAILED, levels = CANCER_TYPE_DETAILED)]

long_lbl <- "Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma"

p1b <- ggplot(valid_counts, aes(x = CANCER_TYPE_DETAILED, y = N)) +
  geom_bar(stat = "identity", width = 0.83, color = "black", fill = "black") +
  coord_flip() +
  scale_x_discrete(
    expand = expansion(mult = c(0.02, 0.02)),
    labels = function(x) ifelse(x == long_lbl,
                                "Chronic Lymphocytic Leukemia/\nSmall Lymphocytic Lymphoma", x)
  ) +
  scale_y_continuous(expand = c(0, 0)) +
  labs(title = NULL, x = NULL, y = "Number of patients") +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_text(hjust = 0.5, size = 9),
    axis.title.x = element_text(size = 9),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 6.5),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", linewidth = 0.2),
    axis.ticks.x = element_line(color = "black", linewidth = 0.3),
    axis.ticks.y = element_line(color = "black", linewidth = 0.3),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position  = "none"
  )

ggsave(file.path(PLOT_ROOT, paste0("si_fig1_B_cancer_distribution_", DATE_STAMP, ".pdf")),
       plot = p1b, width = 3.7, height = 4.2, dpi = 300)

fwrite(valid_counts, file.path(PLOT_ROOT, paste0("si_fig1_B_cancer_distribution_counts_", DATE_STAMP, ".csv")))

cat("\nWrote SFig1B cancer distribution to:", PLOT_ROOT, "\n")
cat("Cancer types (n>=500):", nrow(valid_counts), "| total patients:", sum(valid_counts$N), "\n")
