library(data.table)
library(dplyr)
library(lme4)

# =========================
# GLMM serology
# spans_labtests must contain:
#   PATIENT_ID, span (0/1 or factor), CANCER_TYPE_DETAILED, age_at_diag, sex
#   + lab columns in labs2use
# =========================

# spans_lab_fp <- "path/to/spans_labtests.csv"
# spans_labtests <- fread(spans_lab_fp)

labs2use <- c(
  'HGB', 'HCT', 'RBC', 'MCV', 'MCH', 'WBC', 'Platelets', 'MCHC', 'RDW', 'Neut',
  'Creatinine', 'CO2', 'Glucose', 'Sodium', 'Chloride', 'BUN', 'Calcium', 'Potassium',
  'Lymph', 'Mono', 'Eos', 'Baso', 'Albumin', 'ALK', 'ALT', 'Bilirubin, Total',
  'Protein, Total', 'AST', 'Nucleated RBC', 'Immature Granulocyte'
)


spans_labtests <- spans_labtests %>%
  mutate(
    PATIENT_ID= as.factor(PATIENT_ID),
    span = as.factor(span),
    sex = as.factor(sex),
    CANCER_TYPE_DETAILED = as.factor(CANCER_TYPE_DETAILED),
    age_at_diag = as.numeric(age_at_diag)
  )

cts <- levels(spans_labtests$CANCER_TYPE_DETAILED)

fit_one_cancer <- function(cancer_type, spans_labtests, labs2use) {
  
  dt <- spans_labtests %>% dplyr::filter(CANCER_TYPE_DETAILED == cancer_type)
  
  results <- data.frame(
    test = character(),
    cancer_type = character(),
    estimate = numeric(),
    logor = numeric(),
    lower_ci = numeric(),
    upper_ci = numeric(),
    p = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (test_name in labs2use) {
    
    current_data <- dt %>%
      dplyr::filter(!is.na(.data[[test_name]])) %>%
      dplyr::mutate(result = as.numeric(.data[[test_name]]))
    
    # keep your behavior: if only one sex in this cancer subset, drop sex
    unique_sex <- length(unique(current_data$sex))
    formula <- if (unique_sex > 1) {
      as.formula("span ~ result + age_at_diag + sex + (1 | PATIENT_ID)")
    } else {
      as.formula("span ~ result + age_at_diag + (1 | PATIENT_ID)")
    }
    
    model <- tryCatch(
      glmer(
        formula,
        data = current_data,
        family = binomial(link = "logit"),
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
      ),
      error = function(e) NULL
    )
    
    if (is.null(model)) next
    if (isSingular(model, tol = 1e-5)) next
    
    cs <- summary(model)$coefficients
    if (!("result" %in% rownames(cs))) next
    
    log_or <- cs["result", "Estimate"]
    se     <- cs["result", "Std. Error"]
    pval   <- cs["result", "Pr(>|z|)"]
    
    results <- rbind(
      results,
      data.frame(
        test = test_name,
        cancer_type = cancer_type,
        estimate = exp(log_or),
        logor = log_or,
        lower_ci = exp(log_or - 1.96 * se),
        upper_ci = exp(log_or + 1.96 * se),
        p = pval,
        stringsAsFactors = FALSE
      )
    )
  }
  
  results
}

# ---- run for all cancers  ----
res_list <- lapply(cts, function(ct) fit_one_cancer(ct, spans_labtests, labs2use))
results <- dplyr::bind_rows(res_list)
results$adjusted_p <- p.adjust(results$p, method = "BH")

# fwrite(results, "glmm_results.csv")