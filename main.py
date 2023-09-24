import numpy as np # linear algebra
import pandas as pd # data processing
import os
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

#import the first dataset; seperated by tabs and doesn't have any columns names so this code addresses that 
columns_names=['user_id','item_id','rating','timestamp']
df=pd.read_csv("ml-100k/u.data", sep="\t",names=columns_names)        

#import the second dataset; similar to the first so we added column names
columns_names=['item_id','title','date', "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21"]
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', names=columns_names)

#we only need the first two columns to setup before we merge the two datasets together 
movies=movies.iloc[: , 0:2]


#merge the datasets together
df = pd.merge(df, movies, on='item_id')
#drop the timestamps since we don't need it anymore
df.drop(["timestamp"], axis = 1, inplace = True)



#start setting up the reccomendation system 
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')

#get user input 
selection = st.text_input("Input your movie name and year to find movies with similar rating ex: 'Star Wars (1977)' "+"\n")
selection_user_ratings = moviemat[selection]
selection_user_ratings.head()

#We calculate the correlation of all films with the input movie using the corrwith() method:
similar_to_selection = moviemat.corrwith(selection_user_ratings)

#list the data frame we obtained and see which film it suggests as the closest to input film.
corr_selection = pd.DataFrame(similar_to_selection, columns=['Correlation'])
corr_selection.dropna(inplace=True)
corr_selection.sort_values('Correlation', ascending=False).head(10)

#set it up to show the best rated and voted on films in the list
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.sort_values('rating', ascending=False).head(20)
ratings['rating_vote_number'] = pd.DataFrame(df.groupby('title')['rating'].count())

#output the most voted on films
st.write("Most voted on films")
st.write(ratings.sort_values('rating_vote_number',ascending=False).head())

#add ratings number to our recommendation selection
corr_selection = corr_selection.join(ratings['rating_vote_number'])



#output the final that gets the best fit based on initial choice
final = corr_selection[corr_selection['rating_vote_number']>100].sort_values('Correlation',ascending=False).head()
st.write(final)
