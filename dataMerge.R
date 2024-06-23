# install.packages("data.table")
library(data.table)

#clinicalData
clinicalDataFile <- '../dataset/clinical.tsv'

dataSource <- fread(clinicalDataFile, header = TRUE)
dataClinical <- dataSource[,c('case_submitter_id','project_id','age_at_index','gender','race','ajcc_pathologic_stage')]

dataClinical[] <- lapply(dataClinical, function(x) sub("'--", NA, x))
dataClinical <- dataClinical[complete.cases(dataClinical$ajcc_pathologic_stage),]
dataClinical <- unique(dataClinical)

collectionI <- c("Stage I", "Stage IA", "Stage IB","Stage IC")
collectionII <- c("Stage II", "Stage IIA", "Stage IIB","Stage IIC")
collectionIII <- c("Stage III", "Stage IIIA", "Stage IIIB","Stage IIIC")
collectionIV <- c("Stage IV", "Stage IVA", "Stage IVB","Stage IVC")

rmRows <- c()
for (i in seq(length(dataClinical$ajcc_pathologic_stage))) {
  if (dataClinical[i,'ajcc_pathologic_stage'] %in% collectionI) {
    dataClinical[i,'ajcc_pathologic_stage'] = 1
  } else if (dataClinical[i,'ajcc_pathologic_stage'] %in% collectionII) {
    dataClinical[i,'ajcc_pathologic_stage'] = 2
  } else if (dataClinical[i,'ajcc_pathologic_stage'] %in% collectionIII){
    dataClinical[i,'ajcc_pathologic_stage'] = 3
  } else if (dataClinical[i,'ajcc_pathologic_stage'] %in% collectionIV){
    dataClinical[i,'ajcc_pathologic_stage'] = 4
  } else {
    print(dataClinical[i,'ajcc_pathologic_stage'])
    rmRows <- c(rmRows,i)
  }
}

# unique(dataClinical$ajcc_pathologic_stage)
dataClinical <- dataClinical[-rmRows,]

# genes
## geneID and geneName match
matchIDNameFile <- '../dataset/gene_ensg_type.csv'
matchIDName <- fread(matchIDNameFile, header = TRUE)
# matchIDName <- matchIDName[matchIDName$gene_type == 'protein_coding',]

filterMatch <- function(cancerName, foldC = 3, FDR = 0.005){
  ## filter
  ### filter geneID
  filterFile <- paste0('../dataset/',cancerName,'_edgeR_filter_gene_deg_res.csv')
  
  col_types <- c("character", rep("numeric", 9))
  filterData <- fread(filterFile, header = TRUE, colClasses = col_types)
  names(filterData)[1] <- 'geneID'
  filterData <- filterData[log2foldChange != Inf]
  filterInd <- ((filterData$log2foldChange > foldC) | (filterData$log2foldChange < -foldC)) & (filterData$FDR < FDR)
  filterGene <- filterData[filterInd,'geneID']
  
  ### match
  matchExpFile_1 <- paste0('../dataset/',cancerName,'_ensg_tpm_data.csv')
  matchExp <- fread(matchExpFile_1, header = TRUE)
  matchExp <- na.omit(matchExp)
  matchExp <- matchExp[which(matchExp$geneID %in% filterGene$geneID),]
  
  
  matchIDName <- matchIDName[which(matchIDName$gene_id %in% filterGene$geneID),c('gene_id','gene_name')]
  names(matchIDName) <- c('geneID','geneName')
  if (length(unique(matchIDName$geneID)) != length(unique(matchIDName$geneName))
  ) {
    print("There are some gene_ids match to one gene_name!")
  }
  matchGeneResult <- merge(matchExp,matchIDName,by = 'geneID')
  matchGeneResult$geneID <- NULL
  
  #### process duplicated gene_name
  
  geneNames <- matchGeneResult$geneName
  
  if (length(unique(geneNames[duplicated(geneNames)])) > 0) {
    for (geneName_ in unique(geneNames[duplicated(geneNames)])) {
      nrows <- nrow(matchGeneResult[matchGeneResult$geneName == geneName_,'geneName'])
      matchGeneResult[matchGeneResult$geneName == geneName_,'geneName'] <- paste0(geneName_,seq(nrows))
    } 
  }
  
  geneNames <- matchGeneResult$geneName
  
  IDs <- names(matchGeneResult)[1:length(names(matchGeneResult)) - 1]
  
  dataGenes <- data.table::transpose(matchGeneResult[,-'geneName'])
  names(dataGenes) <- geneNames
  dataGenes$case_submitter_id <- substr(IDs,1,12)
  dataGenes$cancer <- substr(IDs,14,15)
  dataGenes <- dataGenes[dataGenes$cancer < '10',] # cancer case
  dataGenes$cancer <- NULL
  
  # merge 
  mergeClinicalGenes <- merge(dataClinical, dataGenes, by = "case_submitter_id")
  writeFile <- paste0('../dataset/', cancerName, '.csv')
  write.csv(mergeClinicalGenes, writeFile, row.names = FALSE)
}

filterMatch(cancerName = 'LUAD')

