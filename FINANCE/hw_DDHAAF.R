library(zoo)
library(quantmod)


end_date <- Sys.Date()
start_5y <- end_date - 365 * 5
start_10y <- end_date - 365 * 10
output_folder <- "outputs_from_updated_script"
if (!dir.exists(output_folder)) dir.create(output_folder)

getSymbols("V", src = "yahoo", from = start_5y, to = end_date)
getSymbols("MA", src = "yahoo", from = start_5y, to = end_date)

visa <- V[, "V.Adjusted"]
mastercard <- MA[, "MA.Adjusted"]

visa_ret <- dailyReturn(visa)
ma_ret <- dailyReturn(mastercard)


pair_ret <- na.omit(merge(visa_ret, ma_ret))
names(pair_ret) <- c("Visa", "Mastercard")

pair_ret$Divergence <- pair_ret$Visa - pair_ret$Mastercard

top3 <- data.frame(
  Date = index(pair_ret),
  Visa_Return = as.numeric(pair_ret$Visa),
  Mastercard_Return = as.numeric(pair_ret$Mastercard),
  V_minus_MA = as.numeric(pair_ret$Divergence),
  Abs_Divergence = abs(as.numeric(pair_ret$Divergence))
)

top3 <- head(top3[order(-top3$Abs_Divergence), ], 3)

roll60_corr  <- rollapply(
  pair_ret[, c("Visa", "Mastercard")],
  width = 60,
  FUN = function(x) cor(x[, 1], x[, 2]),
  by.column = FALSE,
  align = "right",
  fill = NA
)
pair_ret$Rolling60_Correlation <- roll60_corr

png(file.path(output_folder, "visa_mastercard_rolling60_correlation.png"),
    width = 1000, height = 600)

plot(index(pair_ret), as.numeric(pair_ret$Rolling60_Correlation),
     type = "l",
     main = "Visa vs Mastercard, rolling 60 day corr",
     xlab = "Date",
     ylab = "Rolling 60-day correlation")

abline(v = as.Date(top3$Date), col = "red", lty = 2, lwd = 2)

dev.off()


day1 <- top3$Date[1]

i1 <- which(index(pair_ret) == as.Date(day1))

w1 <- pair_ret[(i1 - 59):i1, c("Visa", "Mastercard")]
w1_no_day <- w1[index(w1) != as.Date(day1)]

outlier1 <- data.frame(
  Date = as.Date(day1),
  Corr_With_Day = as.numeric(cor(w1$Visa, w1$Mastercard)),
  Corr_Without_Day = as.numeric(cor(w1_no_day$Visa, w1_no_day$Mastercard)),
  Cov_With_Day = as.numeric(cov(w1$Visa, w1$Mastercard)),
  Cov_Without_Day = as.numeric(cov(w1_no_day$Visa, w1_no_day$Mastercard)),
  SD_Visa_With_Day = as.numeric(sd(w1$Visa)),
  SD_Visa_Without_Day = as.numeric(sd(w1_no_day$Visa)),
  SD_MA_With_Day = as.numeric(sd(w1$Mastercard)),
  SD_MA_Without_Day = as.numeric(sd(w1_no_day$Mastercard))
)


day2 <- top3$Date[2]

i2 <- which(index(pair_ret) == as.Date(day2))

w2 <- pair_ret[(i2 - 59):i2, c("Visa", "Mastercard")]
w2_no_day <- w2[index(w2) != as.Date(day2)]

outlier2 <- data.frame(
  Date = as.Date(day2),
  Corr_With_Day = as.numeric(cor(w2$Visa, w2$Mastercard)),
  Corr_Without_Day = as.numeric(cor(w2_no_day$Visa, w2_no_day$Mastercard)),
  Cov_With_Day = as.numeric(cov(w2$Visa, w2$Mastercard)),
  Cov_Without_Day = as.numeric(cov(w2_no_day$Visa, w2_no_day$Mastercard)),
  SD_Visa_With_Day = as.numeric(sd(w2$Visa)),
  SD_Visa_Without_Day = as.numeric(sd(w2_no_day$Visa)),
  SD_MA_With_Day = as.numeric(sd(w2$Mastercard)),
  SD_MA_Without_Day = as.numeric(sd(w2_no_day$Mastercard))
)


day3 <- top3$Date[3]

i3 <- which(index(pair_ret) == as.Date(day3))

w3 <- pair_ret[(i3 - 59):i3, c("Visa", "Mastercard")]
w3_no_day <- w3[index(w3) != as.Date(day3)]

outlier3 <- data.frame(
  Date = as.Date(day3),
  Corr_With_Day = as.numeric(cor(w3$Visa, w3$Mastercard)),
  Corr_Without_Day = as.numeric(cor(w3_no_day$Visa, w3_no_day$Mastercard)),
  Cov_With_Day = as.numeric(cov(w3$Visa, w3$Mastercard)),
  Cov_Without_Day = as.numeric(cov(w3_no_day$Visa, w3_no_day$Mastercard)),
  SD_Visa_With_Day = as.numeric(sd(w3$Visa)),
  SD_Visa_Without_Day = as.numeric(sd(w3_no_day$Visa)),
  SD_MA_With_Day = as.numeric(sd(w3$Mastercard)),
  SD_MA_Without_Day = as.numeric(sd(w3_no_day$Mastercard))
)

outlier_table <- rbind(outlier1, outlier2, outlier3)

outlier_table


getSymbols("NVDA", src = "yahoo", from = start_10y, to = end_date)
getSymbols("^GSPC", src = "yahoo", from = start_10y, to = end_date)

nvda <- NVDA[, "NVDA.Adjusted"]
sp500 <- GSPC[, "GSPC.Close"]

nvda_ret <- dailyReturn(nvda)
sp500_ret <- dailyReturn(sp500)

beta_ret <- na.omit(merge(nvda_ret, sp500_ret))
names(beta_ret) <- c("NVDA", "SP500")

rolling_beta <- rollapply(
  beta_ret,
  width = 252,
  FUN = function(x) cov(x[, 1], x[, 2]) / var(x[, 2]),
  by.column = FALSE,
  align = "right",
  fill = NA
)

rolling_rho <- rollapply(
  beta_ret,
  width = 252,
  FUN = function(x) cor(x[, 1], x[, 2]),
  by.column = FALSE,
  align = "right",
  fill = NA
)

rolling_sd_nvda <- rollapply(beta_ret$NVDA, 252, sd, align = "right", fill = NA)
rolling_sd_sp500 <- rollapply(beta_ret$SP500, 252, sd, align = "right", fill = NA)
rolling_vol_ratio <- rolling_sd_nvda / rolling_sd_sp500

beta_parts <- merge(rolling_beta, rolling_rho, rolling_sd_nvda, rolling_sd_sp500, rolling_vol_ratio)
names(beta_parts) <- c("Beta", "Rho", "SD_NVDA", "SD_SP500", "Vol_Ratio")

png(file.path(output_folder, "nvda_sp500_rolling252_beta.png"), width = 1000, height = 600)
plot(index(beta_parts), as.numeric(beta_parts$Beta),
     type = "l",
     main = "NVDA vs S&P 500: Rolling 252-Day Beta",
     xlab = "Date",
     ylab = "Rolling 252-day beta")
abline(v = as.Date(c("2020-01-28", "2020-04-23")), col = "red", lty = 2, lwd = 2)
dev.off()

shift_start <- as.Date("2020-01-28")
shift_end <- as.Date("2020-04-23")

pre <- beta_parts[shift_start]
post <- beta_parts[shift_end]

pre
post

beta_change <- as.numeric(post$Beta) - as.numeric(pre$Beta)
beta_change

regime_shift <- data.frame(
  Shift_Start = shift_start,
  Shift_End = shift_end,
  Beta_Before = as.numeric(pre$Beta),
  Beta_After = as.numeric(post$Beta),
  Beta_Change = beta_change,
  Rho_Before = as.numeric(pre$Rho),
  Rho_After = as.numeric(post$Rho),
  Vol_Ratio_Before = as.numeric(pre$Vol_Ratio),
  Vol_Ratio_After = as.numeric(post$Vol_Ratio)
)

regime_shift

getSymbols("SPY", src = "yahoo", from = start_10y, to = end_date)

spy <- SPY[, "SPY.Adjusted"]
spy_ret <- na.omit(dailyReturn(spy))

var_99 <- -(mean(spy_ret) + qnorm(0.01) * sd(spy_ret))

breach_count <- sum(spy_ret < -var_99, na.rm = TRUE)
breach_percent <- breach_count / length(spy_ret)

var_summary <- data.frame(
  Index = "SPY",
  Start_Date = as.Date(first(index(spy_ret))),
  End_Date = as.Date(last(index(spy_ret))),
  Observations = length(spy_ret),
  Normal_99pct_Daily_VaR = as.numeric(var_99),
  Actual_Breach_Count = breach_count,
  Actual_Breach_Rate = as.numeric(breach_percent)
)

png(file.path(output_folder, "spy_daily_returns_var_histogram.png"), width = 1000, height = 600)
hist(as.numeric(spy_ret), breaks = 80, main = "SPY Daily Returns and Normal 99% VaR", xlab = "Daily return")
abline(v = -as.numeric(var_99), col = "red", lwd = 2)

var_99
breach_count
breach_percent


