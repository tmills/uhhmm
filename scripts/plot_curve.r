#!/usr/bin/Rscript

library(ggplot2)

cliargs = commandArgs(TRUE)

xix = match('-x', cliargs) + 1
xaxis = cliargs[xix]
if (is.na(xix)) {
    xaxis = 'Independent variable'
} else {
    cliargs = cliargs[-c(xix-1, xix)]
}

yix = match('-y', cliargs) + 1
yaxis = cliargs[yix]
if (is.na(yix)) {
    yaxis = 'Dependent variable'
} else {
    cliargs = cliargs[-c(yix-1,yix)]
}

oix = match('-o', cliargs) + 1
out = cliargs[oix]
if (is.na(oix)) {
    out = NULL
} else {
    cliargs = cliargs[-c(oix-1,oix)]
}

tix = match('-t', cliargs) + 1
title = paste0(toupper(substr(cliargs[tix], 1, 1)), substr(cliargs[tix], 2, nchar(cliargs[tix])))
if (!is.na(tix)) {
    cliargs = cliargs[-c(tix-1,tix)]
}

hix = match('-h', cliargs)
if (is.na(hix)) {
    h = FALSE
} else {
    h = TRUE
    cliargs = cliargs[-hix]
}

data <- read.table(cliargs[1],header=h,quote='',comment.char='')
colnames(data) <- c(xaxis,yaxis)
newplot <- ggplot(data, aes(x=data[,xaxis],y=data[,yaxis])) +
    geom_line() + theme_classic() + xlab(xaxis) + ylab(yaxis)
if (!is.na(tix)) {
    newplot <- newplot + ggtitle(title)
}
ggsave(filename=out, plot=newplot, width=5, height=3)

