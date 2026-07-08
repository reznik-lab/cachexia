library(dplyr)
library(data.table)
library(survival)

BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_TABLES  <- file.path(BASE_REV, "rev_tables")
DATE_STAMP  <- "20260706"

meta_fp  <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")
eps_fp   <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
spans_fp <- file.path(REV_RESULTS, paste0("spans_fearon_WL5_BMIlt20_", DATE_STAMP, ".csv"))

PLOT_ROOT  <- file.path(REV_PLOTS, "fearon_definition", "Fig2", "time_dep_cox")
TABLE_ROOT <- file.path(REV_TABLES, "fearon_definition")
dir.create(PLOT_ROOT, recursive = TRUE, showWarnings = FALSE)
dir.create(TABLE_ROOT, recursive = TRUE, showWarnings = FALSE)

meta  <- fread(meta_fp)
eps   <- fread(eps_fp)
spans <- fread(spans_fp)

meta[,  MRN := as.integer(MRN)]
eps[,   MRN := as.integer(MRN)]
spans[, MRN := as.integer(MRN)]

meta_followup <- unique(meta, by = "MRN")
meta_followup[, followup_end := fifelse(!is.na(PT_DEATH_DTE), PT_DEATH_DTE, PLA_LAST_CONTACT_DTE)]
meta_followup[, event := as.integer(!is.na(PT_DEATH_DTE))]
meta_followup[, time  := as.numeric(followup_end - anchor_final)]
meta_followup[is.na(time) | time < 0, time := NA_real_]

DAY_PER_MONTH <- 30.4375
meta_followup[, time_months := time / DAY_PER_MONTH]

cac_spans <- copy(spans)
setnames(cac_spans, old = c("start_day","end_day","span"), new = c("start_time","end_time","cachexia_status"))

cac_spans[meta_followup[, .(MRN, CANCER_TYPE_DETAILED, GENDER, AGE_AT_ANCHOR_YEARS, Sidedness,
                            event, time, time_months, STAGE_CDM_DERIVED, STAGE_CDM_DERIVED_GRANULAR)],
          `:=`(
            CANCER_TYPE_DETAILED = i.CANCER_TYPE_DETAILED,
            GENDER               = i.GENDER,
            age_at_diagnosis     = i.AGE_AT_ANCHOR_YEARS,
            Sidedness            = i.Sidedness,
            event                = i.event,
            time                 = i.time,
            time_months          = i.time_months,
            stage_cdm            = i.STAGE_CDM_DERIVED,
            stage_granular       = i.STAGE_CDM_DERIVED_GRANULAR
          ),
          on = .(MRN)]

cac_spans[, age_at_diagnosis := as.integer(age_at_diagnosis)]
cac_spans[, Sidedness := fifelse(is.na(Sidedness) | trimws(Sidedness) == "", NA_character_, Sidedness)]
cac_spans[CANCER_TYPE_DETAILED != "Colorectal Adenocarcinoma", Sidedness := NA_character_]

cac_spans <- cac_spans[!is.na(time) & time > 0]
cac_spans[end_time > time, end_time := time]
cac_spans <- cac_spans[start_time < end_time]

cac_spans[, span_event := 0L]
cac_spans[event == 1L & start_time < time & end_time >= time,
          `:=`(end_time = time, span_event = 1L)]

cac_spans[, GENDER_NUM := fifelse(GENDER == "MALE", 1L, 0L)]
cac_spans[, traj_start := 0L]
cac_spans[, traj_end   := time]

cac_spans[, `:=`(
  start_mt      = start_time / DAY_PER_MONTH,
  end_mt        = end_time   / DAY_PER_MONTH,
  time_mt       = time       / DAY_PER_MONTH,
  traj_start_mt = 0,
  traj_end_mt   = time       / DAY_PER_MONTH
)]

cac_spans[, stage_gran := as.character(stage_granular)]
cac_spans[is.na(stage_gran) | trimws(stage_gran) == "" | stage_gran %in% c("NA", "N/A"),
          stage_gran := "Missing"]
cac_spans[!stage_gran %in% c("1","2","3","4","Missing"), stage_gran := "Missing"]
cac_spans[, stage_gran := factor(stage_gran, levels = c("1","2","3","4","Missing"), ordered = TRUE)]

cac_spans <- as.data.table(cac_spans)
setorder(cac_spans, MRN, start_mt, end_mt)

cac_fixed <- cac_spans %>%
  group_by(MRN) %>%
  summarize(
    cachexia             = ifelse(any(cachexia_status == 1), 1, 0),
    GENDER_NUM           = first(GENDER_NUM),
    CANCER_TYPE_DETAILED = first(CANCER_TYPE_DETAILED),
    time                 = first(time),
    time_mt              = first(time_mt),
    event                = first(event),
    age_at_diagnosis     = first(age_at_diagnosis),
    stage_gran           = first(stage_gran),
    .groups = "drop"
  )

cac_time_dep <- cac_spans %>%
  arrange(MRN, start_mt, end_mt) %>%
  group_by(MRN) %>%
  mutate(
    first_episode = ifelse(any(cachexia_status == 1),
                           min(which(cachexia_status == 1)),
                           n() + 1),
    cachexia = ifelse(row_number() >= first_episode, 1, 0)
  ) %>%
  ungroup()

large_n_cancer_types <- cac_spans %>%
  group_by(CANCER_TYPE_DETAILED) %>%
  summarize(n = n_distinct(MRN), .groups = "drop") %>%
  filter(n >= 500)

get_all_coefficients <- function(data, cancer_types_list, formula, method) {
  cox_coef_df <- data.frame()

  for (cancer_type in cancer_types_list) {
    cancer_type_data <- data %>% filter(CANCER_TYPE_DETAILED == cancer_type)
    fit <- coxph(formula, data = cancer_type_data, cluster = MRN)
    sm  <- summary(fit)$coefficients

    logHR <- as.numeric(sm[, "coef"])
    pval  <- as.numeric(sm[, "Pr(>|z|)"])
    ci_log <- confint(fit)

    tmp <- data.frame(
      Cancer.Type      = cancer_type,
      Coefficient.Name = rownames(sm),
      Coefficient      = logHR,
      Lower.CI         = as.numeric(ci_log[, 1]),
      Upper.CI         = as.numeric(ci_log[, 2]),
      P.Value          = pval,
      stringsAsFactors = FALSE
    )
    cox_coef_df <- rbind(cox_coef_df, tmp)
  }

  cox_coef_df <- cox_coef_df %>% group_by(Coefficient.Name) %>%
    mutate(P.Adjust = p.adjust(P.Value, method = "BH")) %>%
    ungroup() %>%
    mutate(Significant = (P.Adjust < 0.05))

  write.csv(cox_coef_df, file.path(TABLE_ROOT, paste0(method, "_cox_coef_stage.csv")), row.names = FALSE)
  return(cox_coef_df)
}

code_map <- c(
  "Esophageal Adenocarcinoma" = "ESCA",
  "Acute Myeloid Leukemia" = "AML",
  "Stomach Adenocarcinoma" = "STAD",
  "Pancreatic Adenocarcinoma" = "PAAD",
  "Small Cell Lung Cancer" = "SCLC",
  "Intrahepatic Cholangiocarcinoma" = "IHCH",
  "Uterine Serous Carcinoma/Uterine Papillary Serous Carcinoma" = "USC",
  "High-Grade Serous Ovarian Cancer" = "HGSOC",
  "Cancer of Unknown Primary" = "CUP",
  "Colon Adenocarcinoma" = "COAD",
  "Colorectal Adenocarcinoma" = "COADREAD",
  "Rectal Adenocarcinoma" = "READ",
  "Diffuse Large B-Cell Lymphoma, NOS" = "DLBCLNOS",
  "Lung Squamous Cell Carcinoma" = "LUSC",
  "Non-Small Cell Lung Cancer" = "NSCLC",
  "Bladder Urothelial Carcinoma" = "BLCA",
  "Glioblastoma Multiforme" = "GBM",
  "Lung Adenocarcinoma" = "LUAD",
  "Renal Clear Cell Carcinoma" = "CCRCC",
  "Plasma Cell Myeloma" = "PCM",
  "Invasive Breast Carcinoma" = "BRCA",
  "Myelodysplastic Workup" = "MDSWP",
  "Gastrointestinal Stromal Tumor" = "GIST",
  "Breast Invasive Ductal Carcinoma" = "IDC",
  "Cutaneous Melanoma" = "SKCM",
  "Breast Invasive Lobular Carcinoma" = "ILC",
  "Uterine Endometrioid Carcinoma" = "UEC",
  "Prostate Adenocarcinoma" = "PRAD",
  "Follicular Lymphoma" = "FL",
  "Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma" = "CLLSLL"
)
