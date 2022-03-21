#read data and exploration 

videogame = read.csv('video_game_reviews.csv')
str(videogame)
library(dplyr); library(magrittr)
videogame%>%
summarize(average_rating = mean(review_rating), median_rating = median(review_rating))

#distribution of reviews

library(ggplot2); library(ggthemes)
ggplot(data=videogame,aes(x=review_rating))+
  geom_histogram(fill='sienna3')+
  theme_bw()+
  scale_x_reverse()+
  xlab('Review Rating')+
  coord_flip()
  
#character,word,sentence summary 

library(stringr)
library(dplyr)
videogame %>%
  select(review)%>%
  mutate(characters = nchar(review),
         words = str_count(review,pattern='\\S+'),
         sentences = str_count(review,pattern="[A-Za-z,;'\"\\s]+[^.!?]*[.?!]"))%>%
  summarize_at(c('characters','words','sentences'),.funs = mean,na.rm=T)
  
#grammar

percentUpper = 100*str_count(videogame$review,pattern='[A-Z]')/nchar(videogame$review)
summary(percentUpper)
percentExclamation = 100*str_count(videogame$review,pattern='!')/nchar(videogame$review)
summary(percentExclamation)
#impact on rating
r_upper = cor.test(percentUpper,videogame$review_rating)
r_exclamation = cor.test(percentExclamation,videogame$review_rating)
correlations2 = data.frame(r = c(r_upper$estimate, r_exclamation$estimate),p_value=c(r_upper$p.value, r_exclamation$p.value))
rownames(correlations2) = c('Upper Case','Exclamation Marks')
correlations2


#key words
library(stringr)
mean(str_detect(string = tolower(videogame$review),pattern = 'minecraft|mine craft'))*100
library(dplyr); library(tidytext); library(magrittr)
videogame%>%
  unnest_tokens(input = review, output = word)%>%
  select(word)%>%
  group_by(word)%>%
  summarize(count = n())%>%
  ungroup()%>%
  arrange(desc(count))%>%
  top_n(25)
  
videogame%>%
  unnest_tokens(input = review, output = word)%>%
  select(word)%>%
  group_by(word)%>%
  summarize(count = n())%>%
  ungroup()%>%
  arrange(desc(count))%>%
  top_n(25)%>%
  ggplot(aes(x=reorder(word,count), y=count, fill=count))+
    geom_col()+
    xlab('words')+
    coord_flip()
# the top 25 list after removing the stopwords 
 videogame%>%
  unnest_tokens(input = review, output = word)%>%
  select(word)%>%
  anti_join(stop_words)%>%
  group_by(word)%>%
  summarize(count = n())%>%
  ungroup()%>%
  arrange(desc(count))%>%
  top_n(25)
  
  videogame%>%
  unnest_tokens(input = review, output = word)%>%
  select(word)%>%
  anti_join(stop_words)%>%
  group_by(word)%>%
  summarize(count = n())%>%
  ungroup()%>%
  arrange(desc(count))%>%
  top_n(25)%>%
  ggplot(aes(x=reorder(word,count), y=count, fill=count))+
    geom_col()+
    xlab('words')+
    coord_flip()
    
 #tokenize
 #counting number of words
 library(dplyr); library(tidytext)
videogame %>%
  select(id,review)%>%
  group_by(id)%>%
  unnest_tokens(output = word,input=review)%>%
  ungroup()%>%
  group_by(id)%>%
  summarize(count = n())
  
  library(dplyr); library(tidytext)
videogame %>%
  select(id,review)%>%
  group_by(id)%>%
  unnest_tokens(output = word,input=review)%>%
  ungroup()%>%
  group_by(id)%>%
  summarize(count = n())%>%
  ggplot(aes(x=count))+geom_histogram(bins = 40)+xlab('Number of Words')
  
  
  videogame %>%
  select(id,review)%>%
  group_by(id)%>%
  unnest_tokens(output = word,input=review)%>%
  ungroup()%>%
  count()
  
  
  #Categorize emotions
  #NRC lexicon
  library(lexicon)
  nrc = get_sentiments('nrc')
  videogame%>%
  group_by(id)%>%
  unnest_tokens(output = word, input = review)%>%
  inner_join(nrc)%>%
  group_by(sentiment)%>%
  count()%>%
  arrange(desc(n))
  
  videogame%>%
  group_by(id)%>%
  unnest_tokens(output = word, input = review)%>%
  inner_join(nrc)%>%
  group_by(sentiment)%>%
  count()%>%
  ggplot(aes(x=reorder(sentiment,X = n), y=n, fill=sentiment))+
  geom_col()+
  guides(fill=F)+
  coord_flip()+
  theme_wsj()
  
  #emotions in each review
videogame%>%
  group_by(id, review_rating)%>%
  unnest_tokens(output = word, input = review)%>%
  inner_join(nrc)%>%
  group_by(id,review_rating, sentiment)%>%
  count()%>%
  pivot_wider(names_from = sentiment,values_from=n)%>%
  select(id, review_rating, positive, negative, trust, anticipation, joy, fear, anger, sadness, surprise, disgust)%>%
  mutate_at(.vars = 3:12, .funs = function(x) replace_na(x,0))%>%
  ungroup()
  
  #emotions and ratings
  
  videogame%>%
  group_by(id, review_rating)%>%
  unnest_tokens(output = word, input = review)%>%
  inner_join(nrc)%>%
  group_by(id,sentiment,review_rating)%>%
  count()%>%
  group_by(id,sentiment, review_rating)%>%
  pivot_wider(names_from = sentiment,values_from = n)%>%
  mutate_at(.vars = 3:12, .funs = function(x) replace_na(x,0))%>%
  ungroup()%>%
  pivot_longer(cols = anticipation: sadness, names_to = 'sentiment',values_to = 'n')%>%
  group_by(sentiment, review_rating)%>%
  summarize(n = mean(n))%>%
  ggplot(aes(x=review_rating,y=n,fill=review_rating))+
  geom_col()+
  facet_wrap(~sentiment)+
  guides(fill=F)+
  coord_flip()+
  theme_bw()

videogame%>%
  group_by(id, review_rating)%>%
  unnest_tokens(output = word, input = review)%>%
  inner_join(nrc)%>%
  group_by(id,review_rating, sentiment)%>%
  count()%>%
  pivot_wider(names_from = sentiment,values_from=n)%>%
  select(id, review_rating, positive, negative, trust, anticipation, joy, fear, anger, sadness, surprise, disgust)%>%
  mutate_at(.vars = 3:12, .funs = function(x) replace_na(x,0))%>%
  ungroup()%>%
  pivot_longer(cols = 3:12, names_to = 'sentiment',values_to = 'n')%>%
  group_by(sentiment)%>%
  summarize(r = cor(n,review_rating))
  
  #affin sentiment
  
  afinn = get_sentiments('afinn')
  
  videogame %>%
  select(id,review)%>%
  group_by(id)%>%
  unnest_tokens(output=word,input=review)%>%
  inner_join(afinn)%>%
  summarize(reviewSentiment = mean(value))%>%
  ungroup()%>%
  summarize(min=min(reviewSentiment),
            max=max(reviewSentiment),
            median=median(reviewSentiment),
            mean=mean(reviewSentiment))
  
  videogame %>%
  select(id,review)%>%
  group_by(id)%>%
  unnest_tokens(output=word,input=review)%>%
  inner_join(afinn)%>%
  summarize(reviewSentiment = mean(value))%>%
  ungroup()%>%
  ggplot(aes(x=reviewSentiment,fill=reviewSentiment>0))+
  geom_histogram(binwidth = 0.1)+
  scale_x_continuous(breaks=seq(-5,5,1))+
  scale_fill_manual(values=c('tomato','seagreen'))+
  guides(fill=F)+
  theme_wsj()
  
  
  #word cloud visualizations
  
  library(wordcloud)
wordcloudData = 
  videogame%>%
  group_by(id)%>%
  unnest_tokens(output=word,input=review)%>%
  ungroup()%>%
  select(id,word)%>%
  anti_join(stop_words)%>%
  group_by(word)%>%
  summarize(freq = n())%>%
  arrange(desc(freq))%>%
  ungroup()%>%
  data.frame()


library(wordcloud)
set.seed(617)
wordcloud(words = wordcloudData$word,wordcloudData$freq,scale=c(2,0.5),max.words = 100,colors=brewer.pal(9,"Spectral"))


library(tidyr)
wordcloudData = 
  videogame%>%
  group_by(id)%>%
  unnest_tokens(output=word,input=review)%>%
  ungroup()%>%
  select(id,word)%>%
  anti_join(stop_words)%>%
  inner_join(get_sentiments('bing'))%>%
  ungroup()%>%
  count(sentiment,word,sort=T)%>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)%>%
  data.frame()
rownames(wordcloudData) = wordcloudData[,'word']
wordcloudData = wordcloudData[,c('positive','negative')]
set.seed(617)
comparison.cloud(term.matrix = wordcloudData,scale = c(2,0.5),max.words = 200, rot.per=0)
  
  
















