# This R code uses the twang package to calculate the ATE between multiple groups. Specifically, after inputting the results of the grouping under each split point, we can obtain the corresponding ATE.

library(twang)
library(survey)
set.seed(1)

psmfunction <- function(data,target,col,way=TRUE) {
    
  #input: 
    # data:sample features 
    # target: grouping features
    # col:target feature
  #output: save the intergroup ATE in the '. /data/result/' directory 
    
  genrecolumns = stringr::str_subset(names(data), "^Genre")
  genrecolumns = paste(genrecolumns,collapse="+")
  tempf <- paste0(target,paste("~pre_citCount+firstyear+Ncount+",genrecolumns)) #
  f <- as.formula(tempf)  

  mnps.AOD <- mnps(f,
                 data = data,
                 estimand = "ATE",
                 verbose = TRUE,
                 stop.method = c("ks.mean"), #stop method
                 n.trees = 10000)  #default number

  r <- bal.table(mnps.AOD, digits = 4)
  write.csv(r, file = paste0('./data/result/',year,'_p.csv'), row.names = F, quote = F,sep = ",")
  
  data$w <- get.weights(mnps.AOD, stop.method = "ks.mean")
  design.mnps <- svydesign(ids = ~1, weights = ~w, data = data)
  glm1 <- svyglm(logCit ~ as.factor(quantile_50), design = design.mnps)
  
  write.table(summary(glm1)$coefficients, file = paste0('./data/result/',year,'_coef.csv'), sep = ",")
  write.csv(confint(glm1), file = paste0('./data/result/',year,'_inter.csv'), row.names = T, quote = F,sep=",")

  for( group_labels in c(1:3)){
      datatemp<-data
      datatemp$quantile_50[which(datatemp$quantile_50 == group_labels)] <--1
      datatemp$quantile_50[which(datatemp$quantile_50 == 0)] <- group_labels
      datatemp$quantile_50[which(is.na(datatemp$quantile_50))] <- 0
      design.mnps <- svydesign(ids = ~1, weights = ~w, data = datatemp)
      glm1 <- svyglm(logCit ~ as.factor(quantile_50), design = design.mnps)
      write.table(group_labels, file = paste0('./data/result/',year,'_coef.csv'), append = TRUE, sep = ",")
      write.table(summary(glm1)$coefficients, file = paste0('./data/result/',year,'_coef.csv'), append = TRUE, sep = ",")
      
      write.table(group_labels, file = paste0('./data/result/',year,'_inter.csv'), append = TRUE, sep = ",")
      write.table(confint(glm1), file = paste0('./data/result/',year,'_inter.csv'), append = TRUE,sep = ",")
  }
  
  
}

args <- (commandArgs(TRUE))
group <- as.numeric(args[1])

#year as split point
system.time({
for (label in c('quantile_50')){ 
    for (year in c(group:group)){
        year = paste0('_year_',year)
        print(year)
        
        pathname = paste0('./data/quantile/pair_switch',year,'.csv')
        print(pathname)
        mag_data <- read.csv(pathname)
        mag_data <- mag_data[!is.na(mag_data[label]),]
        mag_data[,'firstyear'] <- factor(mag_data[,'firstyear'])
        mag_data[,label] <- factor(mag_data[,label],levels = 0:3)
        psmfunction(mag_data,label,'logCit')
    }
}
})

#N as split point
for (label in c('quantile_50')){ 
    for (year in c(group:group)){
        year = paste0('_N_',year)
        print(year)
        
        pathname = paste0('./data/quantile/pair_switch',year,'.csv')
        print(pathname)
        mag_data <- read.csv(pathname)
        mag_data <- mag_data[!is.na(mag_data[label]),]
        mag_data[,'firstyear'] <- factor(mag_data[,'firstyear'])
        mag_data[,label] <- factor(mag_data[,label],levels = 0:3)
        psmfunction(mag_data,label,'logCit')
    }
}
})

