
# MILESTONE 3 - Accessing Data from MySQL Database by using Python

## Topic : Twitter Sentiment Analysis on Covid19 and Depression

### Matric No : 17198801/1  (Aimi Nabilah Hassin)

Link : https://github.com/aimihassin/WQD7005---Data-Mining-Milestones

### Installing MySQLdb into the terminal


   * sudo apt-get install python-mysqldb

### Log in into python

* /usr/bin/python


### In python:

**Connect**
    
    db = MySQLdb.connect(host="localhost",
                         user="your_username",
                         passwd="your_password",
                         db="COVIDEP")

    cursor = db.cursor()


**Execute SQL select statement**
    cursor.execute("SELECT * FROM tweets")


**Commit your changes if writing**
**In this case, we are only reading data**
**db.commit()**


**Get the number of rows in the resultset**

    numrows = cursor.rowcount


**Get and display one row at a time**

    for x in range(0, numrows):
        row = cursor.fetchone()
        print row[0], "-->", row[1]


**Close the connection**

    db.close()

**Results**

![mysqldb4.PNG](attachment:mysqldb4.PNG)

![mysqldb6.PNG](attachment:mysqldb6.PNG)
