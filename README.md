# MPlatform

Hello. This is MPlatform project. The continuation of the SPlatform project here: https://github.com/Ivnng/SPlatform.
The purpose of this project is to create a Recommendation System based on some movie datasets:

The source dataset is composed of the following files:

* source_data:
  * amazon_prime_titles.csv
  * disney_plus_titles.csv
  * hulu_titles.csv
  * netflix_titles.csv
  * ratings:
    * 1.csv
    * 2.csv
    * 3.csv
    * 4.csv
    * 5.csv
    * 6.csv
    * 7.csv
    * 8.csv
    
An EDA (Exploratory Data Analysis) process was done on the source files using pandas and ydata_profiling, which you can review in:
* 01_transform.ipynb
* 02_eda.ipynb

The resulting files are listed:
* transformed_data:
  * title.csv
  * rating.csv
  * cast.csv
  * categories.csv
  
 Which you can also find in the link: https://mega.nz/folder/umIgGLTb#GtdfJXpJJGORmujvGbJMZg
 
 Then a SVD algorithm was used, with pandas and surprise library, to create a very simple Movie Recommendation System. You can review the code in:
 * 03_model.ipynb
 
 Regards.
 
 By: Ivanna Villa.
