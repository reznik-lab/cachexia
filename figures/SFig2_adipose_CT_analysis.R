library(data.table)
library(readxl)
library(ggplot2)
library(patchwork)
library(scales)

BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
DATE_STAMP  <- "20260706"

table_dir <- file.path(REV_RESULTS, "fearon_definition", "SFig2")
plot_dir  <- file.path(REV_PLOTS,   "fearon_definition", "SFig2")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir,  recursive = TRUE, showWarnings = FALSE)

MAX_GAP <- 90L


fp <- file.path(REV_INPUTS, "eps_recovery_rad_0820_upd.xlsx")
sh1 <- excel_sheets(fp)[1]
rad <- as.data.table(read_excel(fp, sheet = sh1))
mod_cols <- c("baseline_mod", "onset_mod", "end_mod", "rec_mod")
stopifnot(all(mod_cols %in% names(rad)))

is_blank <- function(x) is.na(x) | !nzchar(trimws(as.character(x)))
rad <- rad[!(is_blank(baseline_mod) & is_blank(onset_mod) & is_blank(end_mod) & is_blank(rec_mod))]

drop_cols <- intersect(c("...28", "ANALYSIS_DATE"), names(rad))
rad <- rad[, setdiff(names(rad), drop_cols), with = FALSE]

rad[, MRN := as.integer(MRN)]
rad[, `:=`(start_date = as.Date(start_date), end_date = as.Date(end_date))]
rad <- rad[!is.na(MRN) & !is.na(start_date) & !is.na(end_date) & start_date < end_date]
rad[, rad_row_id := .I]

has_val <- function(vf, sq) !is.na(vf) | !is.na(sq)
rad[, has_baseline := has_val(baseline_VF, baseline_SQ)]
rad[, has_onset    := has_val(onset_VF,    onset_SQ)]
rad[, has_end      := has_val(end_VF,      end_SQ)]
rad[, has_rec      := has_val(rec_VF,      rec_SQ)]


eps_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
eps_new <- fread(eps_fp)
eps_new[, MRN := as.integer(MRN)]
eps_new <- eps_new[has_cachexia_valid_edemaQC == 1]

regain_fp <- file.path(REV_RESULTS, paste0("episode_regain_WL5_BMIlt20rule_", DATE_STAMP, "_recover5pct_6mo.csv"))
regain <- fread(regain_fp)
regain[, MRN := as.integer(MRN)]

eps_new <- merge(eps_new, regain[, .(MRN, start_day, end_day, recovery_day = regain_day)],
                 by = c("MRN", "start_day", "end_day"), all.x = TRUE)

eps_new <- eps_new[MRN %chin% unique(rad$MRN)]
eps_new[, start_day := as.numeric(start_day)]
eligible_mrns <- eps_new[is.finite(start_day) & start_day > 0, unique(MRN)]
rad     <- rad[MRN %chin% eligible_mrns]
eps_new <- eps_new[MRN %chin% eligible_mrns]

setorder(eps_new, MRN, start_day)
eps_new[, eps_ep_id := seq_len(.N), by = MRN]


meta_fp <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")
msk_clin <- fread(meta_fp)
msk_clin[, MRN := as.integer(MRN)]
msk_clin[, anchor_final := as.Date(anchor_final)]
anc <- unique(msk_clin[!is.na(anchor_final), .(MRN, anchor_final)])

rad2 <- anc[copy(rad), on = .(MRN)]
eps2 <- anc[copy(eps_new), on = .(MRN)]
rad2 <- rad2[!is.na(anchor_final)]
eps2 <- eps2[!is.na(anchor_final)]

rad2[, `:=`(
  start_date = as.Date(start_date),
  end_date   = as.Date(end_date),
  rec_date   = as.Date(rec_rad_date)
)]

rad2[, `:=`(
  rad_start_day = as.integer(start_date - anchor_final),
  rad_end_day   = as.integer(end_date   - anchor_final),
  rad_rec_day   = as.integer(rec_date   - anchor_final)
)]
rad2 <- rad2[is.finite(rad_start_day) & is.finite(rad_end_day) & rad_start_day < rad_end_day]
rad2[, rad_row_id := .I]

setorder(eps2, MRN, start_day)
eps2[, eps_ep_id := seq_len(.N), by = MRN]

safe_pct <- function(new, old) fifelse(is.finite(new) & is.finite(old) & old != 0, 100 * (new - old) / old, NA_real_)

rad2[, `:=`(rad_start_day = as.integer(rad_start_day), rad_end_day = as.integer(rad_end_day), rad_rec_day = as.integer(rad_rec_day))]
eps2[, `:=`(start_day = as.integer(start_day), end_day = as.integer(end_day))]

rad2[, old_rec_flag := is.finite(rad_rec_day)]
eps2[, new_rec_flag := is.finite(recovery_day)]

rad2 <- rad2[is.finite(rad_start_day) & is.finite(rad_end_day) & rad_start_day < rad_end_day]
eps2 <- eps2[is.finite(start_day) & is.finite(end_day) & start_day < end_day]


rad_match <- rad2[, .(
  MRN, rad_row_id, r_start = rad_start_day, r_end = rad_end_day,
  r_lo_start = rad_start_day - MAX_GAP, r_hi_start = rad_start_day + MAX_GAP,
  r_lo_end   = rad_end_day   - MAX_GAP, r_hi_end   = rad_end_day   + MAX_GAP,
  old_rec_flag
)]

eps_match <- eps2[, .(MRN, eps_ep_id, e_start = start_day, e_end = end_day, recovery_day, new_rec_flag)]

setkey(rad_match, MRN)
setkey(eps_match, MRN)

cand <- eps_match[
  rad_match,
  on = .(MRN, e_start >= r_lo_start, e_start <= r_hi_start, e_end >= r_lo_end, e_end <= r_hi_end),
  nomatch = 0L, allow.cartesian = TRUE
]

if (nrow(cand) == 0) {
  stop("No matches found under current MAX_GAP = ", MAX_GAP, " days.")
}

cand[, `:=`(
  d_start = abs(e_start - r_start),
  d_end   = abs(e_end   - r_end),
  score   = abs(e_start - r_start) + abs(e_end - r_end),
  d_max   = pmax(abs(e_start - r_start), abs(e_end - r_end))
)]

setorder(cand, rad_row_id, score, d_max)
best <- cand[, .SD[1], by = rad_row_id]

best_pairs <- merge(
  rad2,
  best[, .(rad_row_id, eps_ep_id, e_start, e_end, recovery_day, new_rec_flag, d_start, d_end, score, d_max)],
  by = "rad_row_id", all.x = FALSE
)

best_pairs[, rec_cat := fifelse(old_rec_flag & new_rec_flag, "Both rec",
                          fifelse(old_rec_flag & !new_rec_flag, "Old rec only",
                            fifelse(!old_rec_flag & new_rec_flag, "New rec only", "Neither rec")))]

loss_cohort   <- best_pairs[has_baseline == TRUE & has_end == TRUE]
follow_cohort <- best_pairs[has_end == TRUE & has_rec == TRUE]
follow_cohort <- follow_cohort[is.finite(rad_rec_day) & is.finite(rad_end_day) & rad_rec_day > rad_end_day]
bmi_rec_subset <- best_pairs[old_rec_flag == TRUE & new_rec_flag == TRUE & has_end == TRUE & has_rec == TRUE]
bmi_rec_subset <- bmi_rec_subset[is.finite(rad_rec_day) & is.finite(rad_end_day) & rad_rec_day > rad_end_day]

loss_cohort[, `:=`(pVF_end = safe_pct(end_VF, baseline_VF), pSQ_end = safe_pct(end_SQ, baseline_SQ))]
follow_cohort[, `:=`(pVF_end_to_fu = safe_pct(rec_VF, end_VF), pSQ_end_to_fu = safe_pct(rec_SQ, end_SQ))]
bmi_rec_subset[, `:=`(pVF_end_to_fu = safe_pct(rec_VF, end_VF), pSQ_end_to_fu = safe_pct(rec_SQ, end_SQ))]

fwrite(best_pairs,     file.path(table_dir, paste0("SFig2J_rad_eps_matched_pairs_MAXGAP", MAX_GAP, ".csv")))
fwrite(loss_cohort,    file.path(table_dir, paste0("SFig2J_loss_cohort_MAXGAP", MAX_GAP, ".csv")))
fwrite(follow_cohort,  file.path(table_dir, paste0("SFig2J_follow_cohort_MAXGAP", MAX_GAP, ".csv")))
fwrite(bmi_rec_subset, file.path(table_dir, paste0("SFig2J_bmi_rec_subset_MAXGAP", MAX_GAP, ".csv")))

cat(sprintf("Matched pairs: %d | loss cohort: %d | follow-up cohort: %d | bmi-rec subset: %d\n",
            nrow(best_pairs), nrow(loss_cohort), nrow(follow_cohort), nrow(bmi_rec_subset)))


plot_pair_violin_hist <- function(dt_in, col_a, col_b, title, ylab, hist_xlab,
                                   ref_lines = NULL, vcol = "#A9B5AEFF", mcol = "#EF6F6AFF",
                                   hcol = "#6388B4FF", w = 0.18, bins = 40,
                                   trim_q = c(0.01, 0.99), n_ybreaks = 7, clamp_pc = 200,
                                   xlabs = c("Baseline", "End")) {
  dt <- as.data.table(copy(dt_in))
  dt <- dt[is.finite(get(col_a)) & is.finite(get(col_b)) & get(col_a) > 0 & get(col_b) > 0]
  dt[, episode_id := .I]

  ylo <- quantile(c(dt[[col_a]], dt[[col_b]]), trim_q[1], na.rm = TRUE)
  yhi <- quantile(c(dt[[col_a]], dt[[col_b]]), trim_q[2], na.rm = TRUE)

  a_t <- paste0(col_a, "_t"); b_t <- paste0(col_b, "_t")
  dt[, (a_t) := pmax(ylo, pmin(yhi, get(col_a)))]
  dt[, (b_t) := pmax(ylo, pmin(yhi, get(col_b)))]

  pts <- rbind(
    dt[, .(episode_id, MRN, Timepoint = "A", Y = get(a_t))],
    dt[, .(episode_id, MRN, Timepoint = "B", Y = get(b_t))]
  )
  pts[, Timepoint := factor(Timepoint, levels = c("A", "B"), labels = xlabs)]

  wide <- dt[, .(episode_id, MRN, y_a = get(a_t), y_b = get(b_t))]
  set.seed(1)
  wide[, `:=`(x_a = 1 + runif(.N, -w, w), x_b = 2 + runif(.N, -w, w))]

  pts2 <- rbind(
    wide[, .(episode_id, MRN, x = x_a, Timepoint = xlabs[1], Y = y_a)],
    wide[, .(episode_id, MRN, x = x_b, Timepoint = xlabs[2], Y = y_b)]
  )
  pts2[, Timepoint := factor(Timepoint, levels = xlabs)]

  y_breaks <- pretty(c(ylo, yhi), n = n_ybreaks)

  p_violin <- ggplot() +
    { if (!is.null(ref_lines)) geom_hline(yintercept = ref_lines, linetype = "dotted", linewidth = 0.35, color = "grey40") } +
    geom_violin(data = pts, aes(x = Timepoint, y = Y), fill = vcol, color = "black", linewidth = 0.25, width = 0.85, trim = TRUE) +
    geom_segment(data = wide, aes(x = x_a, xend = x_b, y = y_a, yend = y_b), linewidth = 0.15, alpha = 0.25, color = "black") +
    geom_point(data = pts2, aes(x = x, y = Y), size = 0.5, alpha = 0.45) +
    stat_summary(data = pts, aes(x = Timepoint, y = Y), fun = mean, geom = "point", size = 2.0, color = mcol) +
    stat_summary(data = pts, aes(x = Timepoint, y = Y, group = 1), fun = mean, geom = "line", linewidth = 0.8, color = mcol) +
    labs(title = title, x = NULL, y = ylab) +
    scale_y_continuous(breaks = y_breaks) +
    coord_cartesian(ylim = c(ylo, yhi)) +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title   = element_text(size = 9, hjust = 0.5),
      axis.title.x = element_text(size = 9),
      axis.title.y = element_text(size = 9),
      axis.text.x  = element_text(size = 8),
      axis.text.y  = element_text(size = 8),
      axis.line    = element_line(color = "black", linewidth = 0.2),
      axis.ticks   = element_line(color = "black", linewidth = 0.3),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin  = margin(t = 5, r = 5, b = 2, l = 5)
    )

  dt[, pct_change := 100 * (get(col_b) - get(col_a)) / get(col_a)]
  hist_dt <- dt[is.finite(pct_change)]
  hist_dt[, pct_t := pmax(-clamp_pc, pmin(clamp_pc, pct_change))]
  mu_pc <- mean(hist_dt$pct_change, na.rm = TRUE)

  p_hist <- ggplot(hist_dt, aes(x = pct_t)) +
    geom_histogram(bins = bins, fill = hcol, color = "black", linewidth = 0.2) +
    geom_vline(xintercept = pmax(-clamp_pc, pmin(clamp_pc, mu_pc)), color = mcol, linetype = "dotted", linewidth = 0.8) +
    scale_y_continuous(expand = c(0, 0)) +
    scale_x_continuous(limits = c(-clamp_pc, clamp_pc)) +
    labs(x = hist_xlab, y = "Episodes") +
    theme_minimal(base_family = "ArialMT") +
    theme(
      axis.title.x = element_text(size = 9),
      axis.title.y = element_text(size = 9),
      axis.text.x  = element_text(size = 8),
      axis.text.y  = element_text(size = 8),
      axis.line    = element_line(color = "black", linewidth = 0.2),
      axis.ticks   = element_line(color = "black", linewidth = 0.3),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin  = margin(t = 1, r = 1, b = 1, l = 1)
    )

  p_violin / p_hist + plot_layout(heights = c(1.9, 0.85))
}

p_SQ_loss <- plot_pair_violin_hist(loss_cohort, "baseline_SQ", "end_SQ",
                                    "SAT", "SAT Area (cm²)", "% SAT Area change",
                                    ref_lines = c(130, 300), clamp_pc = 200,
                                    xlabs = c("Baseline", "End"))

p_VF_loss <- plot_pair_violin_hist(loss_cohort, "baseline_VF", "end_VF",
                                    "VAT", "VAT Area (cm²)", "% VAT Area change",
                                    ref_lines = c(100), clamp_pc = 200,
                                    xlabs = c("Baseline", "End"))

p_SQ_follow <- plot_pair_violin_hist(follow_cohort, "end_SQ", "rec_SQ",
                                      "SAT", "SAT Area (cm²)", "% SAT Area change",
                                      ref_lines = c(130, 300), clamp_pc = 200,
                                      xlabs = c("End", "Follow-up"))

p_VF_follow <- plot_pair_violin_hist(follow_cohort, "end_VF", "rec_VF",
                                      "VAT", "VAT Area (cm²)", "% VAT Area change",
                                      ref_lines = c(100), clamp_pc = 200,
                                      xlabs = c("End", "Follow-up"))

save_pw <- function(pw, filename, w = 2.1, h = 2.8, dpi = 300) {
  ggsave(filename = file.path(plot_dir, filename), plot = pw, width = w, height = h, dpi = dpi, useDingbats = FALSE)
}

save_pw(p_SQ_loss,   paste0("SFig2J_SAT_start_to_end_MAXGAP",    MAX_GAP, ".pdf"))
save_pw(p_VF_loss,   paste0("SFig2L_VAT_start_to_end_MAXGAP",    MAX_GAP, ".pdf"))
save_pw(p_SQ_follow, paste0("SFig2K_SAT_end_to_followup_MAXGAP", MAX_GAP, ".pdf"))
save_pw(p_VF_follow, paste0("SFig2M_VAT_end_to_followup_MAXGAP", MAX_GAP, ".pdf"))

cat("\nWrote adipose (SAT/VAT) CT outputs to:\n  ", table_dir, "\n  ", plot_dir, "\n")
