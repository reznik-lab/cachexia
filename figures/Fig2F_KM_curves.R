source("rev_code/Figures/figure_panel/Fig2F_setup.R")
library(survminer)
library(patchwork)

my_theme <- theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_text(hjust = 0.5, size = 9),
    axis.title.x = element_text(size = 9),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", size = 0.2),
    axis.ticks.x = element_line(color = "black", size = 0.3),
    axis.ticks.y = element_line(color = "black", size = 0.3),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

fmt_p <- function(p, thresh = 1e-6) {
  if (is.na(p)) return("p=NA")
  if (p < thresh) return(paste0("p<", formatC(thresh, format = "e", digits = 0)))
  sprintf("p=%.3g", p)
}

hr_label_fixed <- function(df) {
  fit <- coxph(Surv(time_mt, event) ~ cachexia_f, data = df)
  cname <- grep("^cachexia_f", rownames(summary(fit)$coefficients), value = TRUE)
  hr  <- exp(coef(fit)[[cname]])
  p   <- summary(fit)$coefficients[cname, "Pr(>|z|)"]
  sprintf("HR=%.2f, %s", hr, fmt_p(p))
}

km_df <- cac_fixed %>%
  mutate(
    cachexia_f = factor(cachexia, levels = c(0, 1), labels = c("No cachexia", "Cachexia"))
  ) %>%
  filter(!is.na(time_mt) & time_mt > 0)

fit_pan   <- survfit(Surv(time_mt, event) ~ cachexia_f, data = km_df)
label_pan <- hr_label_fixed(km_df)

p_pan <- ggsurvplot(
  fit_pan, data = km_df,
  conf.int = TRUE, conf.int.style = "ribbon", conf.int.alpha = 0.18,
  pval = FALSE, risk.table = FALSE,
  legend.title = "", legend.labs = levels(km_df$cachexia_f),
  palette = c("#000000", "#8CC2CAFF"),
  xlab = "Months", ylab = NULL,
  ggtheme = my_theme, title = "Pan-cancer",
  surv.median.line = "hv"
)
p_pan$plot <- p_pan$plot +
  coord_cartesian(ylim = c(0, 1), clip = "off") +
  annotate("text", x = Inf, y = Inf, label = label_pan, hjust = 1.02, vjust = 1.5, size = 2.6)

ggsave(file.path(PLOT_ROOT, "km_OS_vanilla_pan_cancer.pdf"),
       plot = p_pan$plot, width = 3.2, height = 3.4, dpi = 300)

luad_fixed <- km_df %>% filter(CANCER_TYPE_DETAILED == "Lung Adenocarcinoma")

fit_luad   <- survfit(Surv(time_mt, event) ~ cachexia_f, data = luad_fixed)
label_luad <- hr_label_fixed(luad_fixed)

p_luad <- ggsurvplot(
  fit_luad, data = luad_fixed,
  conf.int = TRUE, conf.int.style = "ribbon", conf.int.alpha = 0.18,
  pval = FALSE, risk.table = FALSE,
  legend.title = "", legend.labs = levels(luad_fixed$cachexia_f),
  palette = c("#000000", "#8CC2CAFF"),
  xlab = "Months", ylab = "Overall survival probability",
  ggtheme = my_theme, title = "LUAD",
  surv.median.line = "hv"
)
p_luad$plot <- p_luad$plot +
  coord_cartesian(ylim = c(0, 1), clip = "off") +
  annotate("text", x = Inf, y = Inf, label = label_luad, hjust = 1.02, vjust = 1.5, size = 2.6)

ggsave(file.path(PLOT_ROOT, "km_OS_vanilla_LUAD.pdf"),
       plot = p_luad$plot, width = 3.2, height = 3.4, dpi = 300)

combined_vanilla <- (p_luad$plot + p_pan$plot) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.box.margin = margin(t = -6, r = 0, b = 0, l = 0),
    legend.margin = margin(t = -4, r = 0, b = 0, l = 0),
    plot.margin = margin(t = 5, r = 5, b = 0, l = 5),
    plot.spacing = unit(2, "pt")
  )

ggsave(file.path(PLOT_ROOT, "km_OS_vanilla_pan_LUAD_combined.pdf"),
       plot = combined_vanilla, width = 4.5, height = 2.7, dpi = 300)
