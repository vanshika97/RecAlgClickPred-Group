# CS7CS4 ML 19-20 CourseWork 2

This repository applies Data Science techniques on the dataset sourced from recommender-system as-a-service for the module [Machine Learning](https://scss.tcd.ie/modules/2019-2020/CS7CS4.MACHINE_LEARNING.2019-2020.5.SEM101.pdf) at the Trinity College Dublin [School of Computer Science and Statistics](https://www.scss.tcd.ie/).

## Task Overview 
[TCD ML Comp. 2019/20 - Rec Alg Click Pred (Group)](https://www.kaggle.com/c/tcd-ml-comp-201920-rec-alg-click-pred-group/overview)

## Python Libraries used
 - [Numpy](https://numpy.org/)
 - [Pandas](https://pandas.pydata.org/)
 - [Scikit Learn](https://scikit-learn.org/)
 
### Data Exploration
 
|          Dataset Columns         | JabRef | MyVolts | Blog | SplitTrain |
|:--------------------------------:|:------:|:-------:|:----:|:----------:|
| recommendation_set_id            |    ✔   |    ✔    |   ✔  |            |
| user_id                          |    ✖   |    ✔    |   ✖  |            |
| session_id                       |    ✖   |    ✔    |   ✖  |            |
| query_identifier                 |    ✖   |    ✔    |   ✔  |            |
| query_word_count                 |    ✔   |    ✔    |   ✔  |            |
| query_char_count                 |    ✖   |    ✖    |   ✖  |            |
| query_detected_language          |    ✔   |    ✔    |   ✔  |            |
| query_document_id                |    ✔   |    ✖    |   ✖  |            |
| document_language_provided       |    ✖   |    ✖    |   ✖  |            |
| year_published                   |    ✔   |    ✔    |   ✔  |            |
| number_of_authors                |    ✔   |    ✖    |   ✖  |            |
| abstract_word_count              |    ✔   |    ✔    |   ✔  |            |
| abstract_char_count              |    ✖   |    ✖    |   ✖  |            |
| abstract_detected_language       |    ✔   |    ✔    |   ✔  |            |
| first_author_id                  |    ✔   |    ✖    |   ✖  |            |
| num_pubs_by_first_author         |    ✔   |    ✖    |   ✖  |            |
| organization_id                  |    ✖   |    ✖    |   ✖  |            |
| application_type                 |    ✖   |    ✖    |   ✖  |            |
| item_type                        |    ✖   |    ✔    |   ✖  |            |
| request_received                 |    ✖   |    ✖    |   ✖  |            |
| hour_request_received            |    ✔   |    ✔    |   ✔  |            |
| response_delivered               |    ✖   |    ✖    |   ✖  |      ✔     |
| rec_processing_time              |    ✔   |    ✔    |   ✔  |            |
| app_version                      |    ✔   |    ✖    |   ✖  |            |
| app_lang                         |    ✔   |    ✖    |   ✖  |            |
| user_os                          |    ✔   |    ✖    |   ✖  |            |
| user_os_version                  |    ✔   |    ✖    |   ✖  |            |
| user_java_version                |    ✔   |    ✖    |   ✖  |            |
| user_timezone                    |    ✔   |    ✖    |   ✖  |            |
| country_by_ip                    |    ✔   |    ✔    |   ✔  |            |
| timezone_by_ip                   |    ✖   |    ✖    |   ✖  |            |
| local_time_of_request            |    ✖   |    ✖    |   ✖  |            |
| local_hour_of_request            |    ✔   |    ✔    |   ✔  |            |
| number_of_recs_in_set            |    ✖   |    ✖    |   ✖  |      ✔     |
| recommendation_algorithm_id_used |    ✔   |    ✔    |   ✔  |            |
| algorithm_class                  |    ✔   |    ✔    |   ✔  |            |
| cbf_parser                       |    ✔   |    ✔    |   ✔  |            |
| search_title                     |    ✔   |    ✔    |   ✔  |            |
| search_keywords                  |    ✔   |    ✔    |   ✔  |            |
| search_abstract                  |    ✔   |    ✔    |   ✔  |            |
| time_recs_recieved               |    ✖   |    ✖    |   ✖  |      ✔     |
| time_recs_displayed              |    ✖   |    ✖    |   ✖  |      ✔     |
| time_recs_viewed                 |    ✖   |    ✖    |   ✖  |      ✔     |
| clicks                           |    ✖   |    ✖    |   ✖  |      ✔     |
| ctr                              |    ✖   |    ✖    |   ✖  |      ✔     |
| set_clicked                      |    ✖   |    ✖    |   ✖  |      ✔     |  
