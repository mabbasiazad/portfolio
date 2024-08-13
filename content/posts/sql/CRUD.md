
+++
title = "CRUD = Create Read Update Delete"
date = 2024-08-13
author= ["Mehdi Azad"]
summary = ""
+++

# CRUD Operations

## Create a database

```sql
CREATE DATABASE database_name;
DROP DATABASE database_name; 
```

**USE database_name;** in Cloud Console workspaces

When you run the USE <database_name> statement in a Cloud Console workspace, **it sets the database that will be used in any subsequent CREATE statement requests for the specific editor cell**. Different cells can use different databases within the same workspace.

**SHOW TABLES;** show all the tabls in a database

## Creating new table

```jsx
CREAT TABLE ...; 
```

```jsx
CRATE TABLE shoes (
ID            char(5)         PRIMARY KEY
,type         varchar(250)       NOT NULL
,price        decimal(8, 2)   NOT NULL
,discription  varchar(750)       NULL); 

CREARTE TABLE players (
	name varchar(50) NOT NULL
	,city varchar(30) DEFAULT "Barcelona"
);

# if not specified the deault value is NULL
```

**SHOW COLUMNS FROM table_name;** displays information about the columns in a given table

**Data Types**

- Numeric
- String
- Date and Time

Numeric Data Types

- TINYINT
- INT
- Decimal

**String data types**

- char and varchar
- TINYTEXT             (like paragraphs)
- TEXT                      (like an article)
- MEDIUM TEXT    (like text of a book)
- LONGTEXT           (~4GB of data)

## Adding and Droping Columns in a Database

```sql
ALTER TABLE table_name ADD (column_name DATA TYPE);
ALTER TABLE table_name DROP COLUMN column_name;

ALTER TABLE students ADD (age INT, country VARCHAR(50), nationality VARCHAR(250));
ALTER TABLE students DROP COLUMN nationality;
ALTER TABLE studnets MODIFY country VARCHAR(100);
```

## Insert in to table

```jsx
INSERT INTO shoes(
ID,
type,
price,
discription)
VALUES('1234','slippers','8900',NULL) , ('1235','running','10000',NULL);
```

## Create temporary table

```jsx
CREATE TEMPORARY TABLE sandals AS
(
	SELECT *
	FROM shoes
	WHERE type = 'sandals'
);
```

## Update/correct a value in a table

```sql
Update Student_table
SET date_of_birth = '2000-10-12'
WHERE ID = 02;
```

## Delete data form a table

```sql
DELETE FROM Student_table
WHERE ID = 02; 
```