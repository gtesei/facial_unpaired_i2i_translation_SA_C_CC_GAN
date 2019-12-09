library(ggplot2)
library(dplyr)
library(corrplot) 
library(Hmisc)
library(plyr)
library(ggthemes)
## funcs 
plot_corrplot <- function(df,m = "pearson") {
  
  mcor <- cor(df , method = m)
  
  col <- colorRampPalette( c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA")) 
  corrplot( mcor, method ="shade", shade.col = NA, tl.col ="black", tl.srt = 20, 
            col = col(200), 
            addCoef.col ="black", 
            addcolorlabel ="no", order ="AOE" , mar=c(0,0,1,0), title=m) 
}

plot_corrplot_sign_test <- function(df,m = "pearson",sig.level=0.05) {
  mcor <- cor(df,method = m) 
  res1 <- cor.mtest(df,sig.level)
  ##res2 <- cor.mtest(df,0.99)
  corrplot(mcor, p.mat = res1[[1]], low=res1[[2]], upp=res1[[3]], order="hclust",
           pch.col="red", sig.level = sig.level, addrect=3, rect.col="navy",
           plotC="rect",cl.pos="n")  
}

cor.mtest <- function(mat, conf.level = 0.95){
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat <- lowCI.mat <- uppCI.mat <- matrix(NA, n, n)
  diag(p.mat) <- 0
  diag(lowCI.mat) <- diag(uppCI.mat) <- 1
  for(i in 1:(n-1)){
    for(j in (i+1):n){
      tmp <- cor.test(mat[,i], mat[,j], conf.level = conf.level) 
      p.mat[i,j] <- p.mat[j,i] <- tmp$p.value
      lowCI.mat[i,j] <- lowCI.mat[j,i] <- tmp$conf.int[1]
      uppCI.mat[i,j] <- uppCI.mat[j,i] <- tmp$conf.int[2]
    }
  }
  return(list(p.mat, lowCI.mat, uppCI.mat))
}

## load 
setwd('C:/Users/TESEIG/Desktop/PMR/top_n_sent_sentim')
#df = read.csv("sentiment_sample.csv", header = TRUE)
df = read.csv("sentiment_all_interviews.csv", header = TRUE)
