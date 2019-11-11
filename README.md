# CS7CS4 ML 19-20 CourseWork 2

This repository applies Data Science techniques on the dataset sourced from recommender-system as-a-service for the module [Machine Learning](https://scss.tcd.ie/modules/2019-2020/CS7CS4.MACHINE_LEARNING.2019-2020.5.SEM101.pdf) at the Trinity College Dublin [School of Computer Science and Statistics](https://www.scss.tcd.ie/).

## Task Overview 
[TCD ML Comp. 2019/20 - Rec Alg Click Pred (Group)](https://www.kaggle.com/c/tcd-ml-comp-201920-rec-alg-click-pred-group/overview)

## Python Libraries used
 - [Numpy](https://numpy.org/)
 - [Pandas](https://pandas.pydata.org/)
 - [Scikit Learn](https://scikit-learn.org/)
 
### Data Exploration
 
 #### Probably Useful Columns
 - recommendation_set_id -- Unique numeric identifier
 - query_identifier -- Source document title
 - query_word_count -- words in the source document's title
 - query_detected_language -- detected language
 
 #### Probably Not So Useful Columns (MetaData)
 - user_id -- Hash string, 199 unique users, a few users with many thousands of entries 
 
|     |     |
|----------------------------------|------|
| 3ce5de6574b74d9e6f6657547091755b | 4035 |
| 46015d40bb2433595af1410c746e3471 | 3449 |
| 3902677038066845d9c1efc4073ccbad | 3153 |
| a0bcf1241d17237ac6d3dea2fea82cc0 | 2179 |
| 9110dfa0476a9dcf0abbc054a28fb386 | 2131 |
| 2df91cd846e905cc5ec478cc35416fdf | 2123 |
| eeb1480101f13f4221eb8843c83061a8 | 2019 |
| 149e42d40f9e69f42ac011e6efd4efad | 1647 |
| 05e198370527fed40d6448aca1002ca2 | 1564 |
| 66d4664ac18c308c84ead20e52c40e7b | 1440 |
| e89e3a76f3780dd18a044e081b95b095 | 1379 |
| b4bbcd9153f40edfea60b4005a884880 | 1328 |
| b4425f41e5160aa062436cdaf6e88e04 | 1216 |
| fb1e00b65a76f46eb56b0650728e0aff | 1174 |
| f6811db56838495a9cfa0930f5df8c29 | 1160 |
 
 - session_id -- Hash string, same session id for a lot of users which probbaly doesn't make sense
 |     |     |
 |----------------------------------|------|
 | 2b87d422871b0b10908dd0dcf662fdb0 | 2619 |
| c762ddf095b3fc2a60270ddda082c6ad | 2239 |
| 471e4b86e3560c6feb474def098169b6 | 1354 |
| bcf1814caa6afe84eeebef28ff236a7f | 1335 |
| 666332998c62658cc43216116351bf1f | 1159 |
| 5eaddbe64bb311a7ba788adfd9ffdfcb | 1001 |
| 40d2af28a4c309bbb824dc957af59b11 | 997  |
| 01b80466de9751fc3c1cfc72f0950804 | 708  |


 - query_char_count -- The number of characters in the source document's title
 
 - query_document_id -- The unique ID of the source document. For MyVolts and the Blog, this should be redundant with the 'query identifier'. However, for some requests for JabRef, there is such an ID given, for others not. If the ID is not given, this means Darwin & Goliath does not have the document in its database and can only use the submitted title for generating recommendations. If the ID is given, it means Darwin & Goliath has the source document in its database and look u information beyond the title (e.g. author names or description/abstract).
 
 - document_language_provided -- for some documents, the data owner (JabRef, MyVolts…) provides the document language (not 100% reliable).
 
 - year_published -- The year in which the source items was released/published
 
 - number_of_authors -- In case of JabRef and research articles, this field describes the number of authors of the source document-
 
 - abstract_word_count -- The number of words of the description/abstract of the source item
 
 - abstract_char_count -- The number of characters of the description/abstract of the source item
 
 - abstract_detected_language -- The automatically detected language of the source document's abstract
 
 - first_author_id -- The unique ID of the first author of the source document
 
 - num_pubs_by_first_author -- The number of documents in Darwin & Goliath database published by the source document's first author. Actually, we have an algorithm that recommends documents by the 'same author'. So, if a source document is published by 'author X', the 'same author' algorithm recommends other documents authored by that person. Potentially, the number of documents that the person has authored is a good predictor of whether the 'same author' algorithm will perform well.
 
 - organization_id -- the ID of the recommendation partner, i.e. JabRef, MyVolts or our Blog
 
 - application_type -- Digital Library, E-Commerce, or Blog. This should 100% correlate with 'organizationid. item_type -- For JabRef, this value is always 'academicpublication', for our Blog, this value is always 'article'. For MyVolts, the value differs (e.g. 'Hard drives & NAS' or 'Music making & pedals').
 
 - request_received -- The local Irish time when the request for recommendations was received
 - hour_request_received -- The hour when the request for recommendations was received
 - response_delivered -- The time when the server returned recommendations. If this time is long (a few seconds) after the request_received, chances are users have closed the web page already, and won't see any recommendations. This feature is not available in the test dataset. This feature can only help you to filter data but not to train a model!!!
 - rec_processing_time -- The duration in seconds it took the server to calculate recommendations. This should be equal to the difference of request_received and response_delivered.
 - app_version -- The version of the application that requested recommendations. Probably nA for MyVolts and the Blog but given for JabRef.
 - app_lang -- The language of the application that requested recommendations (again, nA for MyVolts and the Blog, but given for JabRef)
 - user_os -- The operating system of the user that recommendations are given to
 - user_os_version -- The operating version of the user that recommendations are given to
 - user_java_version -- The Java version of the user that recommendations are given to
 - user_timezone -- The local time zone of the user that recommendations are given to
 - country_by_ip -- The country of the user, based on the user's IP
 - timezone_by_ip -- The local time zone of the user that recommendations are given to (should be identical with user_timezone)
 - local_time_of_request -- The local time of the user that recommendations are given to
 - local_hour_of_request -- The local hour of the user that recommendations are given to
 - number_of_recs_in_set -- The number of recommendations in the recommendation set. This data is not available in the test set. So, you can use this field to analyze and e.g. filter data, but not for training the model.
 - algorithm_class -- Darwin & Goliath created recommendations with 5 different recommendation approaches/classes: contentbasedfiltering, sentenceembeddings, stereotype, sameauthor, and random. contentbasedfiltering and sentenceembeddings are rather similar and create recommendations based on the terms in the items' titles and abstracts/descriptions. 'Stereotype' recommendations are items that were manually selected by the operators (e.g. MyVolts and JabRef). For instance, 'stereotype recommendations' in JabRef recommend books about academic writing and research because we assume the users of JabRef will like this. 'sameauthor' recommends documents authored by the first author of the source document. Random recommendations randomly select items from our database.
 - recommendation_algorithm_id_used -- For the aforementioned 5 recommendation approaches, Darwin & Goliath has a total of 23 variations. For instance, contentbasedfiltering can be applied to the source documents' titles only, the abstract only, to the abstract and title, and so on. Also, for stereotype recommendations, we have a few slight variations.
cbf_parser, search_title, search_keywords, and search_abstract provide more details on what fields the contentbasedfiltering algorithm used.
 - time_recs_recieved -- The time when the Java Script client received recommendations. This data is only available since a few weeks and not for JabRef. This data will not be available in the test set. Do not use it for training but only to filter data.
 - time_recs_displayed -- The time when recommendations were displayed on the website (should be usually identical to time_recs_recieved or one second later)
 - time_recs_viewed -- The time the recommendations were displayed in the visible area of the screen. For instance, on MyVolts, recommendations are often displayed out-of-sight at the bottom of the page. Only when a user scrolls down the web page, the user will see recommendations. Again, this data will not be available in the test set. Do not use it for training but only to filter data. It may well be that training your model only on 'viewed' recommendations will deliver the best performance because it will strongly reduce noise (but also the amount of training data).
 - clicks -- This is the number of total clicks for the delivered recommendation set. For instance, if a set had 7 recommendations, and 3 recommendations were clicked, then this field is '3' (multiple clicks on the same recommendation are only counted once). This number should never be larger than number_of_recs_in_set (for a few instances it actually is larger, but that must be a bug)
 - ctr -- The Click-Through rate of the set, i.e. clicks divided by number_of_recs_in_set. This means CTR should usually be between 0 and 1. However, a few rows are larger than 1. We do not know why.
 - set_clicked -- '1' if at least one recommendation was clicked, '0' otherwise. 