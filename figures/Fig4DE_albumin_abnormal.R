library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(forcats)
library(patchwork)

# Fig4D (pan-cancer subnormal Albumin frequency, in/out of episode) and
# Fig4E (same, restricted to COADREAD). Needs RAW (non-z-scored) Albumin, since
# the abnormal threshold (3.4 g/dL) is a clinical unit - the z-scored routine-labs
# spans file used for Fig4B/C is unsuitable here. Merges the raw flattened labs
# file directly against our corrected spans, same as the 0303 original.
# Adapted from 0303_ccx_revisions/rev_code/Figures/Fig4.R (Fig4D/E section).

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
spans_labtests[, GENDER := as.factor(GENDER)]
spans_labtests[, CANCER_TYPE_DETAILED := as.factor(CANCER_TYPE_DETAILED)]

# -------------------------
# Albumin abnormal flag (LOW only; raw units)
# -------------------------
lower_bound_albumin <- 3.4

spans_labtests[, Albumin_abn := fifelse(
  is.na(Albumin), NA_real_,
  fifelse(Albumin < lower_bound_albumin, 1, 0)
)]

fill_colors <- c("Below threshold" = "#EF6F6AFF", "In-range" = "#6388B4FF")

# -------------------------
# Fig4D: Pan-cancer Albumin
# -------------------------
df_4d <- spans_labtests %>%
  filter(!is.na(Albumin_abn)) %>%
  group_by(span, Albumin_abn) %>%
  summarise(n = n(), .groups = "drop") %>%
  mutate(label = ifelse(Albumin_abn == 1, "Below threshold", "In-range")) %>%
  group_by(span) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

p_4d <- ggplot(df_4d, aes(x = span, y = prop, fill = fct_rev(label))) +
  geom_bar(stat = "identity", position = "stack", width = 0.9) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                      labels = number_format(accuracy = 0.01), expand = c(0, 0)) +
  scale_x_discrete(expand = c(0.5, 0)) +
  scale_fill_manual(values = fill_colors) +
  labs(title = "Pan-cancer: Albumin", x = "", y = "Proportion", fill = "") +
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

ggsave(file.path(out_dir, paste0("fig4d_albumin_pan_cancer_", DATE_STAMP, ".pdf")), p_4d, width = 2, height = 2.2, device = "pdf")

# -------------------------
# Fig4E: COADREAD Albumin
# -------------------------
df_4e <- spans_labtests %>%
  filter(CANCER_TYPE_DETAILED == "Colorectal Adenocarcinoma", !is.na(Albumin_abn)) %>%
  group_by(span, Albumin_abn) %>%
  summarise(n = n(), .groups = "drop") %>%
  mutate(label = ifelse(Albumin_abn == 1, "Below threshold", "In-range")) %>%
  ungroup() %>%
  complete(span, label, fill = list(n = 0)) %>%
  group_by(span) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

p_4e <- ggplot(df_4e, aes(x = span, y = prop, fill = fct_rev(label))) +
  geom_bar(stat = "identity", position = "stack", width = 0.9) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                      labels = number_format(accuracy = 0.01), expand = c(0, 0)) +
  scale_x_discrete(expand = c(0.5, 0)) +
  scale_fill_manual(values = fill_colors) +
  labs(title = "COADREAD: Albumin", x = "", y = "Proportion", fill = "") +
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

ggsave(file.path(out_dir, paste0("fig4e_albumin_coadread_", DATE_STAMP, ".pdf")), p_4e, width = 2, height = 2.2, device = "pdf")

# -------------------------
# Fig4D+E combined (faceted)
# -------------------------
df_4d2 <- df_4d %>% mutate(cohort = "Pan-cancer")
df_4e2 <- df_4e %>% mutate(cohort = "COADREAD")
df_4d2$cohort <- factor(df_4d2$cohort, levels = c("Pan-cancer", "COADREAD"))
df_4e2$cohort <- factor(df_4e2$cohort, levels = c("Pan-cancer", "COADREAD"))

df_combo <- bind_rows(df_4d2, df_4e2) %>%
  group_by(cohort, span) %>%
  mutate(prop = prop / sum(prop)) %>%
  ungroup()

p_combo <- ggplot(df_combo, aes(x = span, y = prop, fill = fct_rev(label))) +
  geom_bar(stat = "identity", position = "stack", width = 0.9) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                      labels = number_format(accuracy = 0.01), expand = c(0, 0)) +
  scale_x_discrete(expand = c(0.5, 0)) +
  scale_fill_manual(values = c("Below threshold" = "#EF6F6AFF", "In-range" = "#A9B5AEFF")) +
  facet_wrap(~ cohort, nrow = 1, strip.position = "top") +
  labs(title = "Albumin", x = "", y = "Proportion", fill = "") +
  theme_minimal(base_family = "ArialMT") +
  theme(
    strip.background = element_blank(),
    strip.text   = element_text(size = 9),
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
    legend.position = "none"
  )

ggsave(file.path(out_dir, paste0("fig4de_albumin_combined_", DATE_STAMP, ".pdf")), p_combo, width = 3.5, height = 1.9, device = "pdf")

# -------------------------
# Fisher's exact test (COADREAD, cachexia vs non-cachexia, below vs in-range)
# -------------------------
pan_pct  <- df_4d %>% filter(label == "Below threshold") %>% select(span, pct = prop) %>% mutate(pct = 100 * pct)
coad_pct <- df_4e %>% filter(label == "Below threshold") %>% select(span, pct = prop) %>% mutate(pct = 100 * pct)

coad_counts <- df_4e %>%
  select(span, label, n) %>%
  pivot_wider(names_from = label, values_from = n, values_fill = 0)

a <- coad_counts %>% filter(span == "During") %>% pull(`Below threshold`)
b <- coad_counts %>% filter(span == "During") %>% pull(`In-range`)
c <- coad_counts %>% filter(span == "Outside") %>% pull(`Below threshold`)
d <- coad_counts %>% filter(span == "Outside") %>% pull(`In-range`)

tab <- matrix(c(a, b, c, d), nrow = 2, byrow = TRUE)
ft  <- fisher.test(tab)

summary_stats <- data.frame(
  pan_cachexia_pct  = pan_pct$pct[pan_pct$span == "During"],
  pan_non_pct       = pan_pct$pct[pan_pct$span == "Outside"],
  coad_cachexia_pct = coad_pct$pct[coad_pct$span == "During"],
  coad_non_pct      = coad_pct$pct[coad_pct$span == "Outside"],
  coad_or           = unname(ft$estimate),
  coad_p            = ft$p.value
)

fwrite(summary_stats, file.path(out_dir, paste0("fig4de_albumin_stats_", DATE_STAMP, ".csv")))
print(summary_stats)

cat("\nWrote Fig4D/E (Albumin abnormal proportions) to:", out_dir, "\n")
