#!/usr/bin/env Rscript
# ============================================================
# Block B: CCE / IFE Estimators (R implementation)
# Stanford Reviewer Q1: "CCE (Pesaran) or IFE (Bai) estimators
#   should be central rather than ancillary"
#
# Implements:
#   1. Pesaran CCE (Common Correlated Effects) - handles cross-sectional dependence
#   2. Bai IFE (Interactive Fixed Effects) via phtt - latent factor structure
#   3. Driscoll-Kraay SE via plm - robust to CSD + serial correlation
#   4. Comparison table: OLS-FE / TWFE / CCE / IFE / DK-SE
#
# Dependencies:
#   install.packages(c("plm", "phtt", "lmtest", "sandwich"))
#
# Data: data/macro/panel_merged.csv (or equivalent)
# ============================================================

suppressPackageStartupMessages({
  library(plm)
  library(lmtest)
  library(sandwich)
})

# ---- Configuration ----
DATA_FILE <- file.path(dirname(sys.frame(1)$ofile %||% "."),
                       "..", "..", "data", "macro", "panel_merged.csv")

# Fallback paths
if (!file.exists(DATA_FILE)) {
  candidates <- c(
    "data/macro/panel_merged.csv",
    "data/macro/panel_with_inactivity.csv",
    "data/macro/macro_panel.csv",
    "../../data/macro/panel_merged.csv"
  )
  for (f in candidates) {
    if (file.exists(f)) { DATA_FILE <- f; break }
  }
}

# ============================================================
# 1. Load and prepare panel data
# ============================================================
load_panel <- function() {
  if (!file.exists(DATA_FILE)) {
    stop(paste("Panel data not found. Tried:", DATA_FILE,
               "\nSee data/README_data.md for assembly instructions."))
  }

  df <- read.csv(DATA_FILE, stringsAsFactors = FALSE)
  cat(sprintf("  Loaded: %s (%d rows)\n", DATA_FILE, nrow(df)))

  # Identify columns (flexible naming)
  entity_cols <- c("country", "entity", "iso3", "country_code")
  time_cols <- c("year", "time")
  dep_cols <- c("dep_rate", "depression", "depression_rate", "dep")
  sui_cols <- c("suicide_rate", "suicide", "sui_rate")
  proxy_cols <- c("ad_proxy", "proxy", "adproxy")
  gdp_cols <- c("gdp_pc", "gdp", "gdp_per_capita", "NY.GDP.PCAP.CD")
  inet_cols <- c("internet", "internet_pct", "IT.NET.USER.ZS")

  find_col <- function(candidates, label) {
    for (c in candidates) {
      if (c %in% names(df)) return(c)
    }
    warning(paste("Column not found for", label))
    return(NULL)
  }

  col_map <- list(
    entity = find_col(entity_cols, "entity"),
    year = find_col(time_cols, "year"),
    dep = find_col(dep_cols, "depression"),
    suicide = find_col(sui_cols, "suicide"),
    proxy = find_col(proxy_cols, "ad proxy"),
    gdp = find_col(gdp_cols, "GDP"),
    internet = find_col(inet_cols, "internet")
  )

  # Rename
  for (key in names(col_map)) {
    if (!is.null(col_map[[key]]) && col_map[[key]] != key) {
      names(df)[names(df) == col_map[[key]]] <- key
    }
  }

  # Construct proxy if missing
  if (!"proxy" %in% names(df) && "internet" %in% names(df) && "gdp" %in% names(df)) {
    df$proxy <- df$internet * df$gdp / 1000
    cat("  Constructed proxy = internet * gdp / 1000\n")
  }

  # Convert to pdata.frame
  df <- df[complete.cases(df[, c("entity", "year")]), ]
  pdf <- pdata.frame(df, index = c("entity", "year"), drop.index = FALSE)

  cat(sprintf("  Entities: %d, Periods: %d\n",
              length(unique(df$entity)), length(unique(df$year))))

  return(pdf)
}

# ============================================================
# 2. Standard FE and TWFE
# ============================================================
run_fe_models <- function(pdf, outcome, xvars) {
  fml <- as.formula(paste(outcome, "~", paste(xvars, collapse = " + ")))

  cat("\n--- 2a. Entity FE (within) ---\n")
  fe <- tryCatch({
    plm(fml, data = pdf, model = "within", effect = "individual")
  }, error = function(e) { cat("  ERROR:", e$message, "\n"); NULL })
  if (!is.null(fe)) print(summary(fe))

  cat("\n--- 2b. Two-Way FE (entity + time) ---\n")
  twfe <- tryCatch({
    plm(fml, data = pdf, model = "within", effect = "twoways")
  }, error = function(e) { cat("  ERROR:", e$message, "\n"); NULL })
  if (!is.null(twfe)) print(summary(twfe))

  return(list(fe = fe, twfe = twfe))
}

# ============================================================
# 3. Driscoll-Kraay SE (via plm + vcovSCC)
# ============================================================
run_driscoll_kraay <- function(pdf, outcome, xvars) {
  cat("\n--- 3. Driscoll-Kraay SE ---\n")
  fml <- as.formula(paste(outcome, "~", paste(xvars, collapse = " + ")))

  fe <- tryCatch({
    plm(fml, data = pdf, model = "within", effect = "twoways")
  }, error = function(e) { cat("  ERROR:", e$message, "\n"); NULL })

  if (is.null(fe)) return(NULL)

  # vcovSCC = Driscoll-Kraay (SCC = spatial correlation consistent)
  dk_vcov <- tryCatch({
    vcovSCC(fe, maxlag = 3)
  }, error = function(e) {
    cat("  vcovSCC failed, trying vcovHC...\n")
    vcovHC(fe, method = "arellano")
  })

  cat("\n  TWFE with Driscoll-Kraay SE:\n")
  ct <- coeftest(fe, vcov. = dk_vcov)
  print(ct)

  return(list(model = fe, vcov = dk_vcov, coeftest = ct))
}

# ============================================================
# 4. Pesaran CCE (Common Correlated Effects)
# ============================================================
run_cce <- function(pdf, outcome, xvars) {
  cat("\n--- 4. Pesaran CCE (Common Correlated Effects) ---\n")
  cat("  Augments regression with cross-sectional averages of y and X\n")
  cat("  to absorb unobserved common factors.\n\n")

  fml <- as.formula(paste(outcome, "~", paste(xvars, collapse = " + ")))

  cce <- tryCatch({
    pcce(fml, data = pdf, model = "mg")  # Mean Group CCE
  }, error = function(e) {
    cat("  pcce() failed:", e$message, "\n")
    cat("  Trying manual CCE implementation...\n")

    # Manual CCE: add cross-sectional means as regressors
    df <- as.data.frame(pdf)
    for (v in c(outcome, xvars)) {
      df[[paste0(v, "_bar")]] <- ave(df[[v]], df$year, FUN = function(x) mean(x, na.rm = TRUE))
    }
    bar_vars <- paste0(c(outcome, xvars), "_bar")
    fml_cce <- as.formula(paste(outcome, "~",
                                paste(c(xvars, bar_vars), collapse = " + ")))
    pdf_cce <- pdata.frame(df, index = c("entity", "year"))
    plm(fml_cce, data = pdf_cce, model = "within", effect = "individual")
  })

  if (!is.null(cce)) print(summary(cce))
  return(cce)
}

# ============================================================
# 5. Bai IFE (Interactive Fixed Effects) via phtt
# ============================================================
run_ife <- function(pdf, outcome, xvars) {
  cat("\n--- 5. Bai IFE (Interactive Fixed Effects) ---\n")
  cat("  y_it = alpha_i + lambda_i' * f_t + beta * X_it + e_it\n")
  cat("  Absorbs unit-specific loadings on common latent factors.\n\n")

  has_phtt <- requireNamespace("phtt", quietly = TRUE)

  if (!has_phtt) {
    cat("  phtt package not installed.\n")
    cat("  Install with: install.packages('phtt')\n")
    cat("  Falling back to proxy IFE (entity FE + factor-augmented regressors)...\n\n")

    # Proxy IFE: PCA on residuals from FE, then re-estimate with factors
    df <- as.data.frame(pdf)
    fml <- as.formula(paste(outcome, "~", paste(xvars, collapse = " + ")))
    fe <- plm(fml, data = pdf, model = "within", effect = "individual")
    resid_df <- data.frame(entity = df$entity, year = df$year, resid = as.numeric(residuals(fe)))

    # Reshape to entity × year matrix
    resid_wide <- reshape(resid_df, idvar = "entity", timevar = "year",
                          direction = "wide", v.names = "resid")
    resid_mat <- as.matrix(resid_wide[, -1])
    resid_mat[is.na(resid_mat)] <- 0

    # PCA for latent factors
    if (nrow(resid_mat) > 5 && ncol(resid_mat) > 5) {
      pc <- prcomp(resid_mat, center = TRUE, scale. = FALSE)
      n_factors <- min(3, ncol(pc$rotation))
      cat(sprintf("  Using %d principal components as latent factors\n", n_factors))
      cat(sprintf("  Variance explained: %.1f%%\n",
                  100 * sum(pc$sdev[1:n_factors]^2) / sum(pc$sdev^2)))

      # Map factors back to long format
      factors <- pc$rotation[, 1:n_factors, drop = FALSE]
      colnames(factors) <- paste0("F", 1:n_factors)
      years <- as.numeric(gsub("resid\\.", "", colnames(resid_wide)[-1]))
      factor_df <- data.frame(year = years, factors)

      df <- merge(df, factor_df, by = "year", all.x = TRUE)
      factor_vars <- paste0("F", 1:n_factors)
      fml_ife <- as.formula(paste(outcome, "~",
                                  paste(c(xvars, factor_vars), collapse = " + ")))
      pdf_ife <- pdata.frame(df, index = c("entity", "year"))
      ife <- plm(fml_ife, data = pdf_ife, model = "within", effect = "individual")
      cat("\n  Proxy IFE (FE + PCA factors):\n")
      print(summary(ife))
      return(ife)
    }
    return(NULL)
  }

  # phtt available
  library(phtt)
  fml <- as.formula(paste(outcome, "~", paste(xvars, collapse = " + ")))

  ife <- tryCatch({
    Kss(fml, data = as.data.frame(pdf),
        additive.effects = "twoways",
        consult.dim.crit = FALSE, d.max = 5)
  }, error = function(e) {
    cat("  Kss() failed:", e$message, "\n")
    NULL
  })

  if (!is.null(ife)) print(summary(ife))
  return(ife)
}

# ============================================================
# 6. Comparison table
# ============================================================
comparison_table <- function(results_list, xvars) {
  cat("\n", paste(rep("=", 70), collapse = ""), "\n")
  cat(" COMPARISON TABLE: Coefficient estimates across estimators\n")
  cat(paste(rep("=", 70), collapse = ""), "\n\n")

  cat(sprintf("  %-25s", "Variable"))
  for (name in names(results_list)) {
    cat(sprintf("  %12s", name))
  }
  cat("\n")
  cat(paste(rep("-", 25 + 14 * length(results_list)), collapse = ""), "\n")

  for (v in xvars) {
    cat(sprintf("  %-25s", v))
    for (name in names(results_list)) {
      res <- results_list[[name]]
      if (is.null(res)) {
        cat(sprintf("  %12s", "N/A"))
      } else {
        coefs <- tryCatch(coef(res), error = function(e) NULL)
        if (!is.null(coefs) && v %in% names(coefs)) {
          cat(sprintf("  %12.4f", coefs[v]))
        } else {
          cat(sprintf("  %12s", "—"))
        }
      }
    }
    cat("\n")
  }
  cat("\n")
  cat("  Interpretation:\n")
  cat("    - Sign consistency across estimators = robust direction\n")
  cat("    - Magnitude stability = robust effect size\n")
  cat("    - Significance under DK-SE/CCE/IFE = survives CSD correction\n")
}

# ============================================================
# Main
# ============================================================
main <- function() {
  cat(paste(rep("=", 70), collapse = ""), "\n")
  cat(" Block B: CCE / IFE / Driscoll-Kraay Estimators (R)\n")
  cat(" Stanford Reviewer response: cross-sectional dependence\n")
  cat(paste(rep("=", 70), collapse = ""), "\n\n")

  cat("Loading panel data...\n")
  pdf <- tryCatch(load_panel(), error = function(e) {
    cat("ERROR:", e$message, "\n")
    cat("\nThis script requires the assembled macro panel.\n")
    cat("See data/README_data.md for instructions.\n")
    return(NULL)
  })

  if (is.null(pdf)) return(invisible())

  # Check available outcomes
  outcomes <- intersect(c("dep", "suicide"), names(pdf))
  xvars <- intersect(c("proxy", "gdp"), names(pdf))

  if (length(xvars) == 0) {
    cat("ERROR: No explanatory variables found (proxy, gdp)\n")
    return(invisible())
  }

  for (outcome in outcomes) {
    cat("\n", paste(rep("=", 70), collapse = ""), "\n")
    cat(sprintf(" OUTCOME: %s | X = %s\n", toupper(outcome), paste(xvars, collapse = ", ")))
    cat(paste(rep("=", 70), collapse = ""), "\n")

    # Complete cases for this outcome
    vars_needed <- c(outcome, xvars, "entity", "year")
    sub <- pdf[complete.cases(pdf[, vars_needed]), ]
    cat(sprintf("  N = %d (entities = %d, periods = %d)\n",
                nrow(sub), length(unique(sub$entity)), length(unique(sub$year))))

    # Run all estimators
    fe_results <- run_fe_models(sub, outcome, xvars)
    dk_result <- run_driscoll_kraay(sub, outcome, xvars)
    cce_result <- run_cce(sub, outcome, xvars)
    ife_result <- run_ife(sub, outcome, xvars)

    # Comparison
    results_list <- list(
      FE = fe_results$fe,
      TWFE = fe_results$twfe,
      CCE = cce_result,
      IFE = ife_result
    )
    comparison_table(results_list, xvars)
  }

  cat("\n", paste(rep("=", 70), collapse = ""), "\n")
  cat(" ANALYSIS COMPLETE\n")
  cat(paste(rep("=", 70), collapse = ""), "\n")
  cat("\n  Files produced: console output (redirect with > results/r_estimators.log)\n")
  cat("  Usage: Rscript analysis/block_b/r_cce_ife_estimators.R > results/r_estimators.log 2>&1\n")
}

# Run
main()
