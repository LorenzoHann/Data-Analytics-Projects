#basic data preparation

data = read.csv(file = 'jpm_cluster.csv',stringsAsFactors = F)

names(data) = c('id','performance','fees_commissions','depth_of_products','ability_resolve_problems','online_services','choice_of_providers','quality_of_advice','knowledge_of_reps','rep_knowing_your_needs','professional_services','provider_knows_me','quality_of_service',
                'age','marital_status','education')
data$age = factor(data$age,labels = c('27-57','58-68','69-75','75+'))
data$marital_status = factor(data$marital_status,labels=c('married','not married'))
data$education = factor(data$education,labels=c('no college','some college','college graduate','some graduate school','masters degree','doctorate'))
str(data)

data_cluster = data[,2:13]
head(data_cluster[,1:5])

#prepare data for clustering

library(mice)
set.seed(617)
data_cluster = mice::complete(mice(data_cluster,use.matcher=T))

data_cluster = scale(data_cluster)
head(data_cluster[,1:4])

#hierarchical 

d = dist(x = data_cluster,method = 'euclidean') 
clusters = hclust(d = d,method='ward.D2')
plot(clusters)
cor(cophenetic(clusters),d)
library(factoextra)
fviz_dend(x = clusters,k=2)

h_segments = cutree(tree = clusters,k=4)
table(h_segments)

library(psych)
temp = data.frame(cluster = factor(h_segments),
           factor1 = fa(data_cluster,nfactors = 2,rotate = 'varimax')$scores[,1],
           factor2 = fa(data_cluster,nfactors = 2,rotate = 'varimax')$scores[,2])
ggplot(temp,aes(x=factor1,y=factor2,col=cluster))+
  geom_point()
  
 library(cluster)
clusplot(data_cluster,
         h_segments,
         color=T,shade=T,labels=4,lines=0,main='Hierarchical Cluster Plot')
         
         
#k-means

set.seed(617)
km = kmeans(x = data_cluster,centers = 3,iter.max=10000,nstart=25)
table(km$cluster)
noquote(paste(km$totss,'=',km$betweenss,'+',km$tot.withinss,sep = ' '))
km$totss == km$betweenss + km$tot.withinss

within_ss = sapply(1:10,FUN = function(x){
  set.seed(617)
  kmeans(x = data_cluster,centers = x,iter.max = 1000,nstart = 25)$tot.withinss})
  
ggplot(data=data.frame(cluster = 1:10,within_ss),aes(x=cluster,y=within_ss))+
  geom_line(col='steelblue',size=1.2)+
  geom_point()+
  scale_x_continuous(breaks=seq(1,10,1))
  
ratio_ss = sapply(1:10,FUN = function(x) {
  set.seed(617)
  km = kmeans(x = data_cluster,centers = x,iter.max = 1000,nstart = 25)
  km$betweenss/km$totss} )
ggplot(data=data.frame(cluster = 1:10,ratio_ss),aes(x=cluster,y=ratio_ss))+
  geom_line(col='steelblue',size=1.2)+
  geom_point()+
  scale_x_continuous(breaks=seq(1,10,1))
  
  
library(cluster)
pam(data_cluster,k = 3)$silinfo$avg.width
library(cluster)
pam(data_cluster,k = 4)$silinfo$avg.width

library(cluster)
silhoette_width = sapply(2:10,
                         FUN = function(x) pam(x = data_cluster,k = x)$silinfo$avg.width)
ggplot(data=data.frame(cluster = 2:10,silhoette_width),aes(x=cluster,y=silhoette_width))+
  geom_line(col='steelblue',size=1.2)+
  geom_point()+
  scale_x_continuous(breaks=seq(2,10,1))
  
set.seed(617)
km = kmeans(x = data_cluster,centers = 4,iter.max=10000,nstart=25)
k_segments = km$cluster
table(k_segments)

library(psych)
temp = data.frame(cluster = factor(k_segments),
           factor1 = fa(data_cluster,nfactors = 2,rotate = 'varimax')$scores[,1],
           factor2 = fa(data_cluster,nfactors = 2,rotate = 'varimax')$scores[,2])
ggplot(temp,aes(x=factor1,y=factor2,col=cluster))+
  geom_point()
  

#model-based clustering
library(mclust)
clusters_mclust = Mclust(data_cluster)
summary(clusters_mclust)

clusters_mclust_3 = Mclust(data_cluster,G=3)
summary(clusters_mclust_3)

clusters_mclust_3$bic

mclust_bic = sapply(1:10,FUN = function(x) -Mclust(data_cluster,G=x)$bic)
mclust_bic

ggplot(data=data.frame(cluster = 1:10,bic = mclust_bic),aes(x=cluster,y=bic))+
  geom_line(col='steelblue',size=1.2)+
  geom_point()+
  scale_x_continuous(breaks=seq(1,10,1))
  
m_clusters = Mclust(data = data_cluster,G = 4)
m_segments = m_clusters$classification
table(m_segments)

library(psych)
temp = data.frame(cluster = factor(m_segments),
           factor1 = fa(data_cluster,nfactors = 2,rotate = 'varimax')$scores[,1],
           factor2 = fa(data_cluster,nfactors = 2,rotate = 'varimax')$scores[,2])
ggplot(temp,aes(x=factor1,y=factor2,col=cluster))+
  geom_point()
  
  
#profile by needs

data2 = cbind(data,h_segments, k_segments,m_segments)
library(dplyr)
data2 %>%
  select(performance:quality_of_service,k_segments)%>%
  group_by(k_segments)%>%
  summarize_all(function(x) round(mean(x,na.rm=T),2))%>%
  data.frame()
  
  library(dplyr); library(ggplot2); library(tidyr)
data2 %>%
  select(performance:quality_of_service,k_segments)%>%
  group_by(k_segments)%>%
  summarize_all(function(x) round(mean(x,na.rm=T),2))%>%
  gather(key = var,value = value,performance:quality_of_service)%>%
  ggplot(aes(x=var,y=value,fill=factor(k_segments)))+
  geom_col(position='dodge')+
  coord_flip()
  
  
  #profile by demo
  
  prop.table(table(data2$k_segments,data2[,14]),1)
  
  library(ggplot2)
tab = prop.table(table(data2$k_segments,data2[,14]),1)
tab2 = data.frame(round(tab,2))
library(RColorBrewer)
ggplot(data=tab2,aes(x=Var2,y=Var1,fill=Freq))+
  geom_tile()+
  geom_text(aes(label=Freq),size=6)+
  xlab(label = '')+
  ylab(label = '')+
  scale_fill_gradientn(colors=brewer.pal(n=9,name = 'Greens'))
  
  lapply(14:16,function(x) round(prop.table(table(data2$k_segments,data2[,x]),1),2)*100)
  
  lapply(14:16,function(x) {
  dat = round(prop.table(table(data2$k_segments,data2[,x]),1),2)*100
dat = data.frame(dat)
ggplot(data=dat,aes(x=Var2,y=Var1,fill=Freq))+
  geom_tile()+
  geom_text(aes(label=Freq),size=6)+
  xlab(label = '')+
  ylab(label = '')+
  scale_fill_gradientn(colors=brewer.pal(n=9,name = 'Greens'))
})


  
  
  
  
  
  
  
  
  
  




