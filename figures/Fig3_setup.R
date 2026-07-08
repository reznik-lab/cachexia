library(data.table)
library(dplyr)

BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots", "fearon_definition")
REV_TABLES  <- file.path(BASE_REV, "rev_tables", "fearon_definition")
DATE_STAMP  <- "20260706"

FIG3_DIR <- file.path(REV_PLOTS, "Fig3")
dir.create(FIG3_DIR,   recursive = TRUE, showWarnings = FALSE)
dir.create(REV_TABLES, recursive = TRUE, showWarnings = FALSE)

dx_fp   <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")
bmi_fp  <- file.path(REV_INPUTS, "bmi_final_20260129.csv")
meds_fp <- file.path(REV_INPUTS, "regimen_medication_cleaned.csv")

eps5_fp  <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
eps10_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL10_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
eps15_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL15_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))

msk_clin <- fread(dx_fp)
meds     <- fread(meds_fp)
bmi      <- fread(bmi_fp)

eps5  <- fread(eps5_fp)
eps10 <- fread(eps10_fp)
eps15 <- fread(eps15_fp)

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

setDT(msk_clin)
msk_clin[, anchor_final := as.IDate(anchor_final)]
msk_clin[, PT_DEATH_DTE  := as.IDate(PT_DEATH_DTE)]
msk_clin[, CANCER_TYPE_DETAILED := as.character(CANCER_TYPE_DETAILED)]

msk_clin[, STAGE_TXT := toupper(trimws(as.character(STAGE_CDM_DERIVED)))]
msk_clin[STAGE_TXT %in% c("", "NA", "N/A"), STAGE_TXT := NA_character_]

msk_clin[, stage_group := fifelse(is.na(STAGE_TXT), "MISSING",
                                  fifelse(STAGE_TXT == "MISSING", "MISSING",
                                          fifelse(grepl("STAGE\\s*4", STAGE_TXT), "STAGE 4",
                                                  fifelse(grepl("STAGE\\s*[123]", STAGE_TXT), "STAGE 1-3",
                                                          "OTHER"))))]

clin_mrn <- msk_clin[order(MRN, is.na(anchor_final), anchor_final)][, .SD[1], by = MRN]
clin_mrn <- clin_mrn[!is.na(anchor_final),
                     .(MRN, anchor_final, CANCER_TYPE_DETAILED, PT_DEATH_DTE, stage_group)]
clin_mrn[, early_stage := (stage_group == "STAGE 1-3")]

setDT(meds)
meds[, APR_START_DTE := as.IDate(APR_START_DTE)]
meds_first <- meds[!is.na(APR_START_DTE)][order(MRN, APR_START_DTE), .SD[1], by = MRN]

tx_first <- merge(
  meds_first[, .(MRN, first_tx_date = APR_START_DTE)],
  clin_mrn[, .(MRN, anchor_final)],
  by = "MRN", all.x = TRUE
)
tx_first <- tx_first[!is.na(anchor_final)][, first_tx_day := as.integer(first_tx_date - anchor_final)]

setDT(bmi)
bmi[, datetime := as.IDate(datetime)]
last_bmi <- bmi[!is.na(datetime), .(last_bmi_date = max(datetime, na.rm = TRUE)), by = MRN]

eps_fix <- function(ep) {
  setDT(ep)
  ep[, start_day := as.integer(start_day)]
  ep[, end_day := as.integer(end_day)]
  ep[, has_cachexia_valid_edemaQC := as.integer(has_cachexia_valid_edemaQC)]
  ep
}
eps5  <- eps_fix(eps5)
eps10 <- eps_fix(eps10)
eps15 <- eps_fix(eps15)

valid_ctypes <- msk_clin[, .(n_patients = uniqueN(MRN)), by = CANCER_TYPE_DETAILED][n_patients >= 500, CANCER_TYPE_DETAILED]

add_cancer_code <- function(dt) {
  dt[, CANCER_TYPE_DETAILED := trimws(as.character(CANCER_TYPE_DETAILED))]
  dt[, Cancer_Code := unname(code_map[CANCER_TYPE_DETAILED])]
  dt[is.na(Cancer_Code), Cancer_Code := CANCER_TYPE_DETAILED]
  dt
}

build_or_table <- function(dt, flag_col) {
  d <- copy(dt)[!is.na(Cancer_Code)]
  d[, (flag_col) := as.integer(get(flag_col))]

  by_ct <- d[, .(CT_Total = .N, CT_Pos = sum(get(flag_col) == 1L, na.rm = TRUE)), by = Cancer_Code]
  total_pos <- sum(d[[flag_col]] == 1L, na.rm = TRUE)
  total_all <- nrow(d)

  res <- by_ct[, {
    a  <- CT_Pos
    b  <- CT_Total - CT_Pos
    c  <- total_pos - CT_Pos
    d2 <- (total_all - CT_Total) - c
    ft <- fisher.test(matrix(c(a, b, c, d2), nrow = 2))
    .(a=a, b=b, c=c, d=d2,
      OR=unname(ft$estimate),
      LCL=ft$conf.int[1], UCL=ft$conf.int[2],
      P=ft$p.value)
  }, by = Cancer_Code]

  res[, FDR := p.adjust(P, method = "BH")]
  res
}

mapped <- unname(code_map[valid_ctypes])
report_codes <- unique(ifelse(is.na(mapped), valid_ctypes, mapped))
