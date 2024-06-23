library(nhanesA)
library(data.table)

dataSummary <- read.csv('../dataset/dataSummary.csv')
length(unique(dataSummary$V))

tableName <- function(Year,tableN){

  nameDict <- list("2011" = "_G","2013" = "_H","2015" = "_I")

  if (Year == "2017") {
    tableN_C <- paste0('P_', tableN)
  } else{
    tableN_C <- paste0(tableN,nameDict[Year])
  }
  
  return(tableN_C)
}

# tableName(Year = '2011',tableN = 'DIQ')

acquireData <- function(Year){
  # DEMO Table
  dataDEMO <- nhanes(tableName(Year,'DEMO'), translate = FALSE)
  dataDEMOUse <- dataDEMO[,c('SEQN',dataSummary[dataSummary$Table == 'DEMO','V'])]
  dataDEMOUse <- dataDEMOUse[dataDEMOUse$RIDSTATR == '2',] # accept exam
  
  # Q Table
  QClassSummary <- dataSummary[dataSummary$Tclass == 'Q',]
  for (Tname_ in unique(QClassSummary$Table)) {
    data <- nhanes(tableName(Year,Tname_), translate = FALSE)
    dataUse <- data[,c('SEQN',dataSummary[dataSummary$Table == Tname_,'V'])]
    if (Tname_ == 'DIQ') {
      dataUse <- dataUse[dataUse$DIQ010 %in% c(1,2),]
    }
    dataDEMOUse <- merge(dataDEMOUse, dataUse, by = "SEQN")
  }
  
  # Exam Table
  EXAMClassSummary <- dataSummary[dataSummary$Tclass == 'EXAM',]
  for (Tname_ in unique(EXAMClassSummary$Table)) {
    if ((Year != '2017') & (Tname_ == "BPXO")) {
      Tname_ <- "BPX"
      data <- nhanes(tableName(Year,Tname_), translate = FALSE)
      dataUse <- data[,c('SEQN',"BPXSY1","BPXDI1","BPXPLS")]
      names(dataUse) <- c('SEQN',"BPXOSY1","BPXODI1","BPXOPLS1")
    } else{
      data <- nhanes(tableName(Year,Tname_), translate = FALSE)
      dataUse <- data[,c('SEQN',dataSummary[dataSummary$Table == Tname_,'V'])]
    }
    dataDEMOUse <- merge(dataDEMOUse, dataUse, by = "SEQN", all.x = TRUE, all.y = FALSE)
  }
  
  # LAB Table
  LABClassSummary <- dataSummary[dataSummary$Tclass == 'LAB',]
  for (Tname_ in unique(LABClassSummary$Table)) {
    if ((Year == '2011') & (Tname_ == "INS")) {
      Tname_ <- "GLU"
      data <- nhanes(tableName(Year,Tname_), translate = FALSE)
      dataUse <- data[,c('SEQN','LBDINSI')]
    } else{
      data <- nhanes(tableName(Year,Tname_), translate = FALSE)
      dataUse <- data[,c('SEQN',dataSummary[dataSummary$Table == Tname_,'V'])]
    }
    
    dataDEMOUse <- merge(dataDEMOUse, dataUse, by = "SEQN", all.x = TRUE, all.y = FALSE)
  }
  return(dataDEMOUse)
}

Nhanes_2011 <- acquireData(Year = '2011')
Nhanes_2013 <- acquireData(Year = '2013')
Nhanes_2015 <- acquireData(Year = '2015')
Nhanes_2017 <- acquireData(Year = '2017')

NhanesData <- rbind(Nhanes_2011, Nhanes_2013, Nhanes_2015,Nhanes_2017)

# Delete cols which naRate > 0.4 after merge
nSample <- length(NhanesData$SEQN)

naRateD = 0.4
naRate <- lapply(NhanesData,function(x) round(sum(is.na(x)) / nSample,2))
deletNames <- names(naRate[naRate > naRateD])
ncol(NhanesData)
length(deletNames)

NhanesDataPro <- NhanesData[,-which(names(NhanesData) %in% names(naRate[naRate > naRateD]))]
ncol(NhanesDataPro)
# 'LBDGLUSI' is the target
NhanesDataPro['LBDGLUSI'] <- NhanesData[,'LBDGLUSI']

nrow(NhanesDataPro)
NhanesDataPro <- na.omit(NhanesDataPro)
nrow(NhanesDataPro)
summary(NhanesDataPro$DIQ010)
round(sum(NhanesDataPro$DIQ010 == 1)/nrow(NhanesDataPro),2)

# Save 
write.csv(NhanesData, './NhanesData.csv', row.names = FALSE)
write.csv(NhanesDataPro, './NhanesDataPro.csv', row.names = FALSE)