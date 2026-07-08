library(data.table)
library(dplyr)
library(ggplot2)
library(mgcv)
library(scales)
library(patchwork)


BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots", "fearon_definition")
DATE_STAMP  <- "20260706"

labs_fp  <- file.path(REV_INPUTS, "labs_flattened_20260202.csv")
spans_fp <- file.path(REV_RESULTS, "spans_fearon_WL5_BMIlt20_20260706.csv")
msk_fp   <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")

out_dir <- file.path(REV_PLOTS, "Fig4")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

labs  <- fread(labs_fp)
spans <- fread(spans_fp)
msk   <- fread(msk_fp)

setorder(spans, MRN, start_day, end_day)

labs[, Date := as.IDate(Date)]
msk[, anchor_final := as.IDate(anchor_final)]

labs <- merge(
  labs,
  msk[, .(MRN, anchor_final, CANCER_TYPE_DETAILED, GENDER)],
  by = "MRN", all.x = TRUE
)
labs <- labs[!is.na(anchor_final) & !is.na(Date)]
labs[, Days_Since_Dx := as.integer(Date - anchor_final)]

spans_keep <- unique(spans[, .(MRN, start_day, end_day, span)])
setkey(spans_keep, MRN)
setkey(labs, MRN)

spans_labs <- merge(spans_keep, labs, by = "MRN", allow.cartesian = TRUE)

spans_labtests <- spans_labs[
  !is.na(Days_Since_Dx) & Days_Since_Dx >= start_day & Days_Since_Dx <= end_day
]

spans_labtests[
  CANCER_TYPE_DETAILED %in% c("Colon Adenocarcinoma", "Rectal Adenocarcinoma"),
  CANCER_TYPE_DETAILED := "Colorectal Adenocarcinoma"
]

spans_labtests[, span := factor(span, levels = c(0, 1), labels = c("Outside", "During"))]
spans_labtests[, MRN := as.factor(MRN)]
spans_labtests[, GENDER := as.factor(GENDER)]
spans_labtests[, CANCER_TYPE_DETAILED := as.factor(CANCER_TYPE_DETAILED)]

span_cols <- c("Outside" = "#6388B4FF", "During" = "#EF6F6AFF")

ref_low_alb  <- 3.4; ref_high_alb <- 5.4
ref_low_alk  <- 44;  ref_high_alk <- 147


coadread <- spans_labtests[
  CANCER_TYPE_DETAILED == "Colorectal Adenocarcinoma" &
    !is.na(Albumin) & !is.na(ALK) & !is.na(Days_Since_Dx)
]

ref_times <- coadread[, .(
  ref_start_day = min(start_day, na.rm = TRUE),
  ref_end_day   = max(end_day,   na.rm = TRUE)
), by = MRN]

coadread <- merge(coadread, ref_times, by = "MRN", all.x = TRUE)
coadread[, standardized_time := (Days_Since_Dx - ref_start_day) /
           pmax((ref_end_day - ref_start_day), 1e-5)]
coadread <- coadread[!is.na(standardized_time)]

slope_tab_alb <- coadread[, .(slope = coef(lm(Albumin ~ standardized_time))[2]), by = span]
cat("Albumin slopes:\n"); print(slope_tab_alb)

p_albumin <- ggplot(coadread, aes(x = standardized_time, y = Albumin, color = span)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = ref_low_alb, ymax = ref_high_alb,
           fill = "gray90", alpha = 0.4) +
  geom_hline(yintercept = ref_low_alb,  linetype = "dotted", color = "black", linewidth = 0.4) +
  geom_hline(yintercept = ref_high_alb, linetype = "dotted", color = "black", linewidth = 0.4) +
  geom_line(aes(group = MRN), alpha = 0.1, linewidth = 0.3) +
  geom_smooth(data = coadread[span == "Outside"], aes(y = Albumin),
              method = "gam", formula = y ~ s(x, bs = "cs"),
              se = TRUE, color = span_cols["Outside"], fill = span_cols["Outside"], linewidth = 1.0) +
  geom_smooth(data = coadread[span == "During"], aes(y = Albumin),
              method = "gam", formula = y ~ s(x, bs = "cs"),
              se = TRUE, color = span_cols["During"], fill = span_cols["During"], linewidth = 1.0) +
  scale_color_manual(values = span_cols, name = "Status") +
  scale_y_continuous(limits = c(2.2, 5.5)) +
  labs(title = NULL, x = NULL, y = "Albumin (g/dL)") +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_text(hjust = 0.5, size = 9, face = "bold"),
    legend.position = "top",
    legend.title = element_text(size = 9),
    legend.text  = element_text(size = 9),
    axis.title.x = element_text(size = 9),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", linewidth = 0.2),
    panel.grid   = element_blank(),
    axis.ticks   = element_line(color = "black", linewidth = 0.3)
  )

slope_tab_alk <- coadread[, .(slope = coef(lm(ALK ~ standardized_time))[2]), by = span]
cat("ALK slopes:\n"); print(slope_tab_alk)

p_alk <- ggplot(coadread, aes(x = standardized_time, y = ALK, color = span)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = ref_low_alk, ymax = ref_high_alk,
           fill = "gray90", alpha = 0.4) +
  geom_hline(yintercept = ref_low_alk,  linetype = "dotted", color = "black", linewidth = 0.4) +
  geom_hline(yintercept = ref_high_alk, linetype = "dotted", color = "black", linewidth = 0.4) +
  geom_line(aes(group = MRN), alpha = 0.1, linewidth = 0.3) +
  geom_smooth(data = coadread[span == "Outside"], aes(y = ALK),
              method = "gam", formula = y ~ s(x, bs = "cs"),
              se = TRUE, color = span_cols["Outside"], fill = span_cols["Outside"], linewidth = 1.0) +
  geom_smooth(data = coadread[span == "During"], aes(y = ALK),
              method = "gam", formula = y ~ s(x, bs = "cs"),
              se = TRUE, color = span_cols["During"], fill = span_cols["During"], linewidth = 1.0) +
  scale_color_manual(values = span_cols, name = "Status") +
  scale_y_continuous(limits = c(30, 180)) +
  labs(title = NULL, x = "Standardized time", y = "ALK (U/L)") +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_text(hjust = 0.5, size = 9, face = "bold"),
    legend.position = "top",
    legend.title = element_text(size = 9),
    legend.text  = element_text(size = 9),
    axis.title.x = element_text(size = 9),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", linewidth = 0.2),
    panel.grid   = element_blank(),
    axis.ticks   = element_line(color = "black", linewidth = 0.3)
  )

combined_plot <- p_albumin / p_alk + plot_layout(ncol = 1, guides = "collect") &
  theme(legend.position = "none")

out_path <- file.path(out_dir, paste0("fig4e_coadread_albumin_alk_trajectory_", DATE_STAMP, ".pdf"))
ggsave(out_path, combined_plot, width = 3.7, height = 3.2, device = "pdf")

cat("\nWrote Fig4E (COADREAD Albumin/ALK trajectory) to:", out_path, "\n")
