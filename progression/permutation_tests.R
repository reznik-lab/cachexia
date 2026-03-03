# ============================================================
# Progression overlap permutation test 
#
# INPUTS (data.tables):
#   spans:
#     patient_id, CANCER_TYPE_DETAILED, span (0/1), start_day, end_day
#   prog:
#     patient_id, days_since_anchor, PROGRESSION (e.g., "Yes"/"No")
#
# OUTPUT:
#   data.table with (per cancer type):
#     CANCER_TYPE, n_patients, n_prog, Observed, Null_Mean, P_Value
# ============================================================

library(data.table)

perm_once <- function(spans_df, prog_days) {
  shuffled <- spans_df[sample(.N)]
  shuffled[, start_day := shift(cumsum(duration), fill = 0)]
  shuffled[, end_day   := start_day + duration - 1]
  
  sum(vapply(
    prog_days,
    function(d) any(d >= shuffled$start_day & d <= shuffled$end_day & shuffled$span == 1L),
    logical(1)
  ))
}

# observed overlap within a single patient_id (no permutation)
obs_once <- function(spans_df, prog_days) {
  sum(vapply(
    prog_days,
    function(d) any(d >= spans_df$start_day & d <= spans_df$end_day & spans_df$span == 1L),
    logical(1)
  ))
}

run_perm_progression_overlap <- function(spans,
                                         prog,
                                         min_patients = 500,
                                         n_perm = 1000,
                                         seed = 123) {
  
  setDT(spans); setDT(prog)
  set.seed(seed)
  
  spans <- spans[!is.na(patient_id) & !is.na(CANCER_TYPE_DETAILED) &
                   !is.na(start_day) & !is.na(end_day) & !is.na(span)]
  spans[, span := as.integer(span)]
  spans[, duration := as.integer(end_day - start_day + 1L)]
  spans <- spans[duration >= 1L]
  
  prog <- prog[PROGRESSION == "Yes" & !is.na(patient_id) & !is.na(days_since_anchor)]
  prog[, days_since_anchor := as.numeric(days_since_anchor)]
  
  cancer_counts <- spans[, .(n_patients = uniqueN(patient_id)), by = CANCER_TYPE_DETAILED]
  eligible <- cancer_counts[n_patients >= min_patients, CANCER_TYPE_DETAILED]
  
  results <- list()
  
  for (ct in eligible) {
    
    spans_ct <- spans[CANCER_TYPE_DETAILED == ct,
                      .(patient_id, duration, span, start_day, end_day)]
    prog_ct  <- prog[patient_id %in% spans_ct$patient_id,
                     .(patient_id, days_since_anchor)]
    
    ids <- intersect(unique(spans_ct$patient_id), unique(prog_ct$patient_id))
    if (length(ids) == 0L) next
    
    spans_by_id <- split(spans_ct[, .(duration, span, start_day, end_day)], spans_ct$patient_id)
    prog_by_id  <- split(prog_ct$days_since_anchor, prog_ct$patient_id)
    
    P_vec <- integer(length(ids))
    M_mat <- matrix(0L, nrow = length(ids), ncol = n_perm)
    
    for (i in seq_along(ids)) {
      pid <- as.character(ids[i])
      spans_id <- spans_by_id[[pid]]
      prog_id  <- prog_by_id[[pid]]
      
      P_vec[i]    <- obs_once(spans_id, prog_id)
      M_mat[i, ]  <- replicate(n_perm, perm_once(spans_id[, .(duration, span)], prog_id))
    }
    
    T_total <- nrow(prog_ct[patient_id %in% ids])
    if (T_total == 0L) next
    
    Observed  <- sum(P_vec) / T_total
    S_perm    <- colSums(M_mat) / T_total
    Null_Mean <- mean(S_perm)
    
    P_Value <- mean(abs(S_perm - Null_Mean) >= abs(Observed - Null_Mean))
    
    results[[ct]] <- data.table(
      CANCER_TYPE = ct,
      n_patients  = uniqueN(spans_ct$patient_id),
      n_prog      = T_total,
      Observed    = Observed,
      Null_Mean   = Null_Mean,
      P_Value     = P_Value
    )
  }
  
  rbindlist(results, fill = TRUE)
}

# summary_df <- run_perm_progression_overlap(spans, prog, min_patients = 500, n_perm = 1000, seed = 123)
# fwrite(summary_df, "perm_summary.csv")