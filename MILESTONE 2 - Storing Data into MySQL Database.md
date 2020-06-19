
# MILESTONE 2 - Storing Data into MySQL Database

## Topic : Twitter Sentiment Analysis on Covid19 and Depression

### Matric No : 17198801/1  (Aimi Nabilah Hassin)

Link : https://github.com/aimihassin/WQD7005---Data-Mining-Milestones

### MySQL Installation

* sudo apt-get install mysql-server
* systemctl start mysql
* /usr/bin/mysql -u root -p

### Importing CSV file to MySQL

* **In MySQL:**
    
    
    - CREATE DATABASE COVIDEP;
    - USE COVIDEP;
    - create table tweets (id varchar(6), location varchar(50),
    - tweetcreatedts DATETIME, text varchar(280),
    - hashtags varchar(280));
    - exit

* **Import 'covidep_tweets.csv' to the created table:**


    - mysql -uroot -proot --local_infile=1 COVIDEP -e "LOAD DATA LOCAL INFILE
      '~/Downloads/covidep_tweets.csv' INTO TABLE tweets
      FIELDS TERMINATED BY ','"


### Display the created table

* **Log in again:**

    * /usr/bin/mysql -u root -p
    
    
* **In MySQL:**


    - USE COVIDEP;
    - select * from tweets;
    

**result:**

![1.PNG](attachment:1.PNG)
