rm(list=ls())
outcomeFile="../dataset/GCST90002232_buildGRCh37.tsv.gz" #https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90002001-GCST90003000/GCST90002232/GCST90002232_buildGRCh37.tsv.gz

library(SNPlocs.Hsapiens.dbSNP144.GRCh37)
snps <- SNPlocs.Hsapiens.dbSNP144.GRCh37
data <- data.table::fread(outcomeFile)
head(data)
for (i in unique(data$chromosome)){
  my_pos <- data$base_pair_location[data$chromosome == i]
  chr_snps <- snpsBySeqname(snps, as.character(i))
  idx <- match(my_pos,pos(chr_snps))
  rsids <- mcols(chr_snps)$RefSNP_id[idx]
  data$rsid[data$chromosome == i] <- rsids
  print(paste(as.character(i),"is ok"))
}
write.csv(data, "../dataset/GCST90002232_buildGRCh37_id.csv", row.names=F)


exposureFile="../dataset/ieu-b-111.vcf.gz" #https://gwas.mrcieu.ac.uk/files/ieu-b-111/ieu-b-111.vcf.gz
outcomeFile="../dataset/GCST90002232_buildGRCh37_id.csv"
exposureName="triglycerides"
outcomeName="Fasting_glucose"
# install.packages("devtools")
# remotes::install_github("MRCIEU/TwoSampleMR")
# devtools::install_github("explodecomputer/plinkbinr") 
# remotes::install_github("mrcieu/gwasvcf")
# remotes::install_github("mrcieu/gwasglue")
library(gwasvcf)
library(gwasglue)
library(VariantAnnotation)
library(TwoSampleMR)
library(stringr)
library(data.table)
library(TwoSampleMR)
library(MRInstruments)
library(ieugwasr)
library(MRPRESSO)
library(dplyr)
sumstats_dt <- readVcf(exposureFile) 
data = gwasvcf_to_TwoSampleMR(vcf = sumstats_dt, type="exposure")
head(data)
##############################
data <- data[!(data$chr.exposure == "chr6" & data$pos.exposure >= 28477797 & data$pos.exposure <= 33448354), ]
data$maf=ifelse(data$eaf.exposure>0.5,1-data$eaf.exposure, data$eaf.exposure)
head(data)
dim(data)
outTab<-subset(data, pval.exposure < 5e-08 & maf > 0.01)
dim(outTab)
write.csv(outTab, file="exposure.pvalue.csv", row.names=F)
##############################
exposure_dat<-read_exposure_data(filename="exposure.pvalue.csv",
                                 sep = ",",
                                 snp_col = "SNP",
                                 beta_col = "beta.exposure",
                                 se_col = "se.exposure",
                                 effect_allele_col = "effect_allele.exposure",
                                 other_allele_col = "other_allele.exposure",
                                 pval_col = "pval.exposure",
                                 eaf_col = "eaf.exposure",
                                 samplesize_col = "samplesize.exposure",
                                 clump = F)
exposure_dat_clumped <- ld_clump(
  dat = tibble(rsid = exposure_dat$SNP,
               pval = exposure_dat$pval.exposure, 
               id = exposure_dat$id.exposure), 
  clump_kb = 10000,      
  clump_r2 = 0.001,      
  clump_p = 5e-08,           
  bfile = "../dataset/1000_ld/EUR",      #http://fileserve.mrcieu.ac.uk/ld/1kg.v3.tgz
  plink_bin = "../dataset/plink"  #https://github.com/MRCIEU/genetics.binaRies/tree/master/binaries
)
exposure_dat_clumped<-exposure_dat[which(exposure_dat$SNP%in%exposure_dat_clumped$rsid),]
dim(exposure_dat_clumped)
write.csv(exposure_dat_clumped, file="exposure.LD.csv", row.names=F)
rm(exposure_dat)
rm(exposure_dat_clumped)
##############################
Ffilter = 10
dat<-read.csv("exposure.LD.csv", header=T, sep=",", check.names=F)
N=dat[1,"samplesize.exposure"]
dat=transform(dat,R2=2*((beta.exposure)^2)*eaf.exposure*(1-eaf.exposure))
dat=transform(dat,F=(N-2)*R2/(1-R2))
outTab=dat[dat$F>Ffilter,]
head(outTab)
write.csv(dat, "exposure.F.csv", row.names=F)
rm(dat)
rm(outTab)
rm(N)
rm(Ffilter)
exposure_dat<-read_exposure_data("exposure.F.csv", sep = ",",
                                 snp_col = "SNP",
                                 beta_col = "beta.exposure",
                                 se_col = "se.exposure",
                                 effect_allele_col = "effect_allele.exposure",
                                 pval_col = "pval.exposure",
                                 other_allele_col = "other_allele.exposure",
                                 eaf_col = "eaf.exposure",
                                 samplesize_col = "samplesize.exposure",
                                 clump = F)

outcome_dat <- fread(outcomeFile)
intersection_dat <- merge(exposure_dat,outcome_dat,by.x="SNP",by.y="rsid")
write.csv(intersection_dat,file="intersection_dat.csv")
rm(intersection_dat)
##############################
outcome_dat <- read_outcome_data(
  snps = exposure_dat$SNP,
  filename = "intersection_dat.csv", sep = ",",
  snp_col = "SNP",
  beta_col = "beta",
  se_col = "standard_error",
  effect_allele_col = "effect_allele",
  other_allele_col = "other_allele",
  pval_col = "p_value",
  eaf_col = "effect_allele_frequency",
  samplesize_col = "sample_size"
)
dim(outcome_dat)
outcome_dat<- subset(outcome_dat,pval.outcome>=5e-05)
dim(outcome_dat)
write.csv(outcome_dat, file="outcome_dat.csv", row.names=F)
##############################
exposure_dat$exposure=exposureName
outcome_dat$outcome=outcomeName
dat<-harmonise_data(exposure_dat=exposure_dat,
                    outcome_dat=outcome_dat,action= 2)
dim(dat)
write.csv(dat, file="dat.csv", row.names=F)
dat=dat[dat$mr_keep=="TRUE",]
write.csv(dat, file="table.SNP.csv", row.names=F)
##############################

my_mr_presso <- function(BetaOutcome, BetaExposure, SdOutcome, SdExposure, data, OUTLIERtest = FALSE, DISTORTIONtest = FALSE, SignifThreshold = 0.05, NbDistribution = 1000, seed = NULL){
  
  if(!is.null(seed))
    set.seed(seed)
  
  if(SignifThreshold > 1)
    stop("The significance threshold cannot be greater than 1")
  
  if(length(BetaExposure) != length(SdExposure))
    stop("BetaExposure and SdExposure must have the same number of elements")
  
  if(class(data)[1] != "data.frame")
    stop("data must be an object of class data.frame, try to rerun MR-PRESSO by conversing data to a data.frame \'data = as.data.frame(data)\'")
  
  # Functions
  "%^%" <- function(x, n) with(eigen(x), vectors %*% (values^n * t(vectors)))
  
  getRSS_LOO <- function(BetaOutcome, BetaExposure, data, returnIV){
    dataW <- data[, c(BetaOutcome, BetaExposure)] * sqrt(data[, "Weights"])
    X <- as.matrix(dataW[ , BetaExposure])
    Y <- as.matrix(dataW[ , BetaOutcome])
    CausalEstimate_LOO <- sapply(1:nrow(dataW), function(i) {
      (t(X[-i, ]) %*% X[-i, ])%^%(-1) %*% t(X[-i, ]) %*% Y[-i, ]
    })
    
    if(length(BetaExposure) == 1)
      RSS <- sum((Y - CausalEstimate_LOO * X)^2, na.rm = TRUE)
    else
      RSS <- sum((Y - rowSums(t(CausalEstimate_LOO) * X))^2, na.rm = TRUE)
    
    if(returnIV)
      RSS <- list(RSS, CausalEstimate_LOO)
    return(RSS)
  }
  
  
  getRandomData <- function(BetaOutcome, BetaExposure, SdOutcome, SdExposure, data){
    mod_IVW <- lapply(1:nrow(data), function(i) lm(as.formula(paste0(BetaOutcome, " ~ -1 + ", paste(BetaExposure, collapse=" + "))), weights = Weights, data = data[-i, ]))
    dataRandom <- cbind(eval(parse(text = paste0("cbind(", paste0("rnorm(nrow(data), data[, \'", BetaExposure, "\'], data[ ,\'", SdExposure, "\'])", collapse = ", "), ", sapply(1:nrow(data), function(i) rnorm(1, predict(mod_IVW[[i]], newdata = data[i, ]), data[i ,\'", SdOutcome,"\'])))"))), data$Weights)
    colnames(dataRandom) <- c(BetaExposure, BetaOutcome, "Weights")
    return(dataRandom)
  }
  
  # 0- Transforming the data + checking number of observations
  data <- data[, c(BetaOutcome, BetaExposure, SdOutcome, SdExposure)]
  data <- data[rowSums(is.na(data)) == 0, ]
  data[, c(BetaOutcome, BetaExposure)] <- data[, c(BetaOutcome, BetaExposure)] * sign(data[, BetaExposure[1]])
  data$Weights <- data$Weights <- 1/data[, SdOutcome]^2
  
  if(nrow(data) <= length(BetaExposure) + 2)
    stop("Not enough intrumental variables")
  
  if(nrow(data) >= NbDistribution)
    stop("Not enough elements to compute empirical P-values, increase NbDistribution")
  
  # 1- Computing the observed residual sum of squares (RSS)
  RSSobs <- getRSS_LOO(BetaOutcome = BetaOutcome, BetaExposure = BetaExposure, data = data, returnIV = OUTLIERtest)
  
  # 2- Computing the distribtion of expected residual sum of squares (RSS)
  randomData <- replicate(NbDistribution, getRandomData(BetaOutcome = BetaOutcome, BetaExposure = BetaExposure, SdOutcome = SdOutcome, SdExposure = SdExposure, data = data), simplify = FALSE)
  RSSexp <- sapply(randomData, getRSS_LOO, BetaOutcome = BetaOutcome, BetaExposure = BetaExposure, returnIV = OUTLIERtest)
  if(OUTLIERtest)
    GlobalTest <- list(RSSobs = RSSobs[[1]], Pvalue = sum(RSSexp[1, ] > RSSobs[[1]])/NbDistribution)
  else
    GlobalTest <- list(RSSobs = RSSobs[[1]], Pvalue = sum(RSSexp > RSSobs[[1]])/NbDistribution)
  
  # 3- Computing the single IV outlier test
  if(GlobalTest$Pvalue < SignifThreshold & OUTLIERtest){
    OutlierTest <- do.call("rbind", lapply(1:nrow(data), function(SNV){
      randomSNP <- do.call("rbind", lapply(randomData, function(mat) mat[SNV, ]))
      if(length(BetaExposure) == 1){
        Dif <- data[SNV, BetaOutcome] - data[SNV, BetaExposure] * RSSobs[[2]][SNV]
        Exp <- randomSNP[, BetaOutcome] - randomSNP[, BetaExposure] * RSSobs[[2]][SNV]
      } else {
        Dif <- data[SNV, BetaOutcome] - sum(data[SNV, BetaExposure] * RSSobs[[2]][, SNV])
        Exp <- randomSNP[, BetaOutcome] - rowSums(randomSNP[, BetaExposure] * RSSobs[[2]][, SNV])
      }
      pval <- sum(Exp^2 > Dif^2)/length(randomData)
      pval <- cbind.data.frame(RSSobs = Dif^2, Pvalue = pval)
      return(pval)
    }))
    row.names(OutlierTest) <- row.names(data)
    # OutlierTest$Pvalue <- apply(cbind(OutlierTest$Pvalue*nrow(data), 1), 1, min) # Bonferroni correction
    OutlierTest$Pvalue <- apply(cbind(OutlierTest$Pvalue, 1), 1, min) # Bonferroni correction
  } else{
    OUTLIERtest <- FALSE
  }
  
  # 4- Computing the test of the distortion of the causal estimate
  mod_all <- lm(as.formula(paste0(BetaOutcome, " ~ -1 + ", paste(BetaExposure, collapse = "+"))), weights = Weights, data = data)
  if(DISTORTIONtest & OUTLIERtest){
    getRandomBias <- function(BetaOutcome, BetaExposure, SdOutcome, SdExposure, data, refOutlier){
      indices <- c(refOutlier, replicate(nrow(data)-length(refOutlier), sample(setdiff(1:nrow(data), refOutlier))[1]))
      mod_random <- lm(as.formula(paste0(BetaOutcome, " ~ -1 + ", paste(BetaExposure, collapse = "+"))), weights = Weights, data = data[indices[1:(length(indices) - length(refOutlier))], ])
      return(mod_random$coefficients[BetaExposure])
    }
    refOutlier <- which(OutlierTest$Pvalue <= SignifThreshold)
    
    if(length(refOutlier) > 0){
      if(length(refOutlier) < nrow(data)){
        BiasExp <- replicate(NbDistribution, getRandomBias(BetaOutcome = BetaOutcome, BetaExposure = BetaExposure, data = data, refOutlier = refOutlier), simplify = FALSE)
        BiasExp <- do.call("rbind", BiasExp)
        
        mod_noOutliers <- lm(as.formula(paste0(BetaOutcome, " ~ -1 + ", paste(BetaExposure, collapse=" + "))), weights = Weights, data = data[-refOutlier, ])
        BiasObs <- (mod_all$coefficients[BetaExposure] - mod_noOutliers$coefficients[BetaExposure]) / abs(mod_noOutliers$coefficients[BetaExposure])
        BiasExp <- (mod_all$coefficients[BetaExposure] - BiasExp) / abs(BiasExp)
        BiasTest <- list(`Outliers Indices` = refOutlier, `Distortion Coefficient` = 100*BiasObs, Pvalue = sum(abs(BiasExp) > abs(BiasObs))/NbDistribution)
      } else {
        BiasTest <- list(`Outliers Indices` = "All SNPs considered as outliers", `Distortion Coefficient` = NA, Pvalue = NA)
      }
    } else{
      BiasTest <- list(`Outliers Indices` = "No significant outliers", `Distortion Coefficient` = NA, Pvalue = NA)
    }
  }
  
  # 5- Formatting the results
  GlobalTest$Pvalue <- ifelse(GlobalTest$Pvalue == 0, paste0("<", 1/NbDistribution), GlobalTest$Pvalue)
  if(OUTLIERtest){
    OutlierTest$Pvalue <- replace(OutlierTest$Pvalue, OutlierTest$Pvalue == 0, paste0("<", nrow(data)/NbDistribution))
    if(DISTORTIONtest){
      BiasTest$Pvalue <- ifelse(BiasTest$Pvalue == 0, paste0("<", 1/NbDistribution), BiasTest$Pvalue)
      res <- list(`Global Test` = GlobalTest, `Outlier Test` = OutlierTest, `Distortion Test` = BiasTest)
    } else {
      res <- list(`Global Test` = GlobalTest, `Outlier Test` = OutlierTest)
    }
    if(nrow(data)/NbDistribution > SignifThreshold)
      warning(paste0("Outlier test unstable. The significance threshold of ", SignifThreshold, " for the outlier test is not achievable with only ", NbDistribution, " to compute the null distribution. The current precision is <", nrow(data)/NbDistribution, ". Increase NbDistribution."))
  } else {
    res <- list(`Global Test` = GlobalTest)
  }
  
  OriginalMR <- cbind.data.frame(BetaExposure, "Raw", summary(mod_all)$coefficients)
  colnames(OriginalMR) <- c("Exposure", "MR Analysis", "Causal Estimate", "Sd", "T-stat", "P-value")
  if(exists("mod_noOutliers")){
    OutlierCorrectedMR <- cbind.data.frame(BetaExposure, "Outlier-corrected", summary(mod_noOutliers)$coefficients, row.names = NULL)
  } else {
    warning("No outlier were identified, therefore the results for the outlier-corrected MR are set to NA")
    OutlierCorrectedMR <- cbind.data.frame(BetaExposure, "Outlier-corrected", t(rep(NA, 4)), row.names = NULL)
  }
  colnames(OutlierCorrectedMR) <- colnames(OriginalMR)
  MR <- rbind.data.frame(OriginalMR, OutlierCorrectedMR)
  row.names(MR) <- NULL
  
  res <- list(`Main MR results` = MR, `MR-PRESSO results` = res)
  return(res)
}
presso=my_mr_presso(BetaOutcome = "beta.outcome", BetaExposure = "beta.exposure", SdOutcome = "se.outcome", SdExposure = "se.exposure", 
                 OUTLIERtest = TRUE,DISTORTIONtest = TRUE, data = dat, NbDistribution = 1000,  
                 SignifThreshold = 0.05, seed=1234)
del_SNP<-dat$SNP[presso$`MR-PRESSO results`$`Distortion Test`$`Outliers Indices`]
dat<-dat[!(dat$SNP%in%del_SNP),]
write.csv(presso[[1]]$`MR-PRESSO results`$`Outlier Test`, file="table.MR-PRESSO.csv")
presso[[1]]$`MR-PRESSO results`$`Distortion Test`$Pvalue
##############################
mrResult=mr(dat)
mrTab=generate_odds_ratios(mrResult)
write.csv(mrTab, file="table.MRresult.csv", row.names=F)

heterTab=mr_heterogeneity(dat)
write.csv(heterTab, file="table.heterogeneity.csv", row.names=F)

pleioTab=mr_pleiotropy_test(dat)
write.csv(pleioTab, file="table.pleiotropy.csv", row.names=F)
##############################

mr_scatter_plot(mr_results=mrResult, dat=dat)
ggsave("mr_scatter.pdf",width = 6,height = 6)
pdf("mr_scatter.pdf",width = 6,height = 6)
mr_scatter_plot(mr_results=mrResult, dat=dat)

dev.off()

res_single=mr_singlesnp(dat)
mr_forest_plot(res_single)
ggsave("mr_forest.pdf",width = 8,height = 30)
pdf("mr_forest.pdf",width = 8,height = 30)
mr_forest_plot(res_single)
dev.off()
 
mr_funnel_plot(singlesnp_results = res_single)
ggsave("mr_funnel.pdf",width = 6,height = 6)
pdf("mr_funnel.pdf",width = 6,height = 6)
mr_funnel_plot(singlesnp_results = res_single)
dev.off()

mr_leaveoneout_plot(leaveoneout_results = mr_leaveoneout(dat))
ggsave("mr_leaveoneout.pdf",width = 8,height = 30)
pdf("mr_leaveoneout.pdf",width = 8,height = 30)
mr_leaveoneout_plot(leaveoneout_results = mr_leaveoneout(dat))
dev.off()

