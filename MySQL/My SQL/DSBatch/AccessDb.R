# Load the RMySql package
#install.packages("RMySQL")
library(RMySQL)
# Connect to the database
#mydb<-dbConnect(MySQL(),dbname='exadb',host='localhost',
#              user='root',password='secret')
conn<-dbConnect(MySQL(),dbname='practicedb',host='localhost',
                user='root')

# Drop a Table
dbSendQuery(conn, "DROP TABLE IF EXISTS RBooks ;")

# Create a table
dbSendQuery(conn, "
CREATE TABLE if not exists Rbooks (
book_id INT,
title VARCHAR(50),
author VARCHAR(50));")

# Insert a record
dbSendQuery(conn, "INSERT INTO RBooks
VALUES(1,'My R Book','Geethika');")

cur = dbSendQuery(conn, "select * from RBooks");
data = fetch(cur, n=-1)

# Executing Queries
cur = dbSendQuery(conn, "select * from Books");
data = fetch(cur, n=-1)

# List the tables in your database
dbListTables(conn)
dbListFields(conn, 'Books')

# Close Connection
dbDisconnect(conn)


