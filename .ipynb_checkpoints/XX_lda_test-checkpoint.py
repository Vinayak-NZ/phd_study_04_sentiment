#!/usr/bin/env python
# coding: utf-8

# In[116]:

if __name__ == '__main__':
    import pandas as pd
    import re
    import os
    os.chdir('..')
    from wordcloud import WordCloud
    import gensim
    from gensim.utils import simple_preprocess
    import gensim.corpora as corpora
    import nltk
    from nltk.corpus import stopwords
    from pprint import pprint
    import pyLDAvis.gensim
    import pickle
    import pyLDAvis
    from multiprocessing import Process

    # In[117]:

    os.chdir('..')


    df = pd.read_csv('C:/Users/vanandkuma/Documents/projects/sentiment_analysis/input/app_v1_feedback.txt', sep="\t", header=0, encoding = "latin1")


    # In[118]:


    df.head()


    # In[119]:


    df = df.drop(columns=['person_id', 'segment_id', 'tags'], 
                 axis = 1)


    # In[120]:


    df.head()


    # In[121]:


    df['text_processed'] = \
    df['text'].map(lambda x: re.sub('[,\.!?]', '', x))


    # In[122]:


    df['text_processed'] = \
    df['text_processed'].map(lambda x: x.lower())


    # In[123]:


    df['text_processed'].head()


    # In[124]:


    long_string = ','.join(list(df['text_processed'].values))


    # In[125]:


    wordcloud = WordCloud(background_color = "white", max_words = 500, 
                          contour_width = 3, 
                          contour_color = 'steelblue')


    # In[126]:


    wordcloud.generate(long_string)


    # In[127]:


    wordcloud.to_image()


    # In[128]:


    stop_words = stopwords.words('german')


    # In[129]:


    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), 
                                                 deacc = True))


    # In[130]:


    def remove_stopwords(texts):
        return[[word for word in simple_preprocess(str(doc)) 
                if word not in stop_words] for doc in texts]


    # In[131]:


    data = df.text_processed.values.tolist()
    data_words = list(sent_to_words(data))


    # In[132]:


    data_words = remove_stopwords(data_words)


    # In[133]:


    #print(data_words)


    # In[134]:


    id2word = corpora.Dictionary(data_words)


    # In[135]:


    texts = data_words


    # In[136]:


    corpus = [id2word.doc2bow(text) for text in texts]


    # In[137]:


    #pprint(corpus)


    # In[138]:


    num_topics = 3


    # In[139]:


    lda_model = gensim.models.LdaMulticore(corpus = corpus, 
                                           id2word = id2word, 
                                           num_topics = num_topics)


    # In[140]:


    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # In[141]:
    
    lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.display(lda_display)
    pyLDAvis.save_html(lda_display, 'C:/Users/vanandkuma/Documents/projects/sentiment_analysis/output/lda.html')

#     # In[141]:

#     LDAvis_data_filepath = os.path.join('C:/Users/vanandkuma/Documents/projects/sentiment_analysis/input/ldavis_prepared_'+str(num_topics))


#     # In[142]:

#     if 1 == 1:
#         LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, 
#                                                   id2word)
        
#         with open(LDAvis_data_filepath, 'wb') as f:
#             pickle.dump(LDAvis_prepared, f)


#     # In[143]:


#     with open(LDAvis_data_filepath, 'rb') as f:
#         LDAvis_prepared = pickle.load(f)


#     # In[144]:


#     pyLDAvis.save_html(LDAvis_prepared, 'C:/Users/vanandkuma/Documents/projects/sentiment_analysis/output/ldavis_prepared_'+str(num_topics)+'.html')


#     # In[145]:


#     LDAvis_prepared

