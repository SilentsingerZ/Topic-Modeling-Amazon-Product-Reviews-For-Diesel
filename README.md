# Topic-Modeling-Amazon-Product-Reviews-For-Diesel
Using method of Latent Dirichlet Allocation (LDA) topic modeling to deliver business insight for Italian retail clothing company - Diesel S.p.A.

Original Dataset: [Amazon Product Data from Prof. Julian McAuley at UC-San Diego](http://jmcauley.ucsd.edu/data/amazon/links.html)(~80gb)
To remedy this, Prof. Vargo has picked to two smaller datasets that only contain \
⋅⋅* [meta-data about products](https://www.dropbox.com/s/r6z2gt7xyok1ztt/meta_Clothing_Shoes_and_Jewelry.json?dl=1) that are in categorized as “Clothing, Shoes & Jewelry” and \
⋅⋅* [reviews about products](https://www.dropbox.com/s/f3a7o8svixw7zqh/reviews_Clothing_Shoes_and_Jewelry.json?dl=1) that are in the “Clothing, Shoes & Jewelry” category.

Live Demo: https://silentsingerz.github.io/topicmodeling/

## File Description:
Code.py: Python project script.\
alldieselreviews.json: All product reviews for Diesel on Amazon.\
classified_reviews.jsonl: Classified reviews from SUPER REVIEWERS (Top 5%)\
lda_topics.txt: Topic Model from LDA modeling\
pyBestLDAvis.html: Interactive Visualization from PyLDAvis
