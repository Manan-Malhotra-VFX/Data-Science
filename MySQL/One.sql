show databases;
create database practicedb;
use practicedb;
show tables;
create table Students(sid integer,sname char(30),age integer,course char(30));
desc students;
insert into students values(1,"Thomas",32,"MySQL");
insert into students values(2,"Loona",24,"DS");
select * from students;
alter table students add column marks integer;
alter table students drop column age;
alter table students change column sid std_id integer;
alter table students rename to mystudents;
select * from mystudents;
rename table mystudents to students;
select * from students;
create table mytab(id integer, name character(30));
show tables;
drop table mytab;
create table patients(pid integer,name varchar(30),dob date, toa datetime); #instead of datetime its better to use time stapm as its utc format
desc patients;
insert into patients values(1,"Harry","1997-07-11","2021-08-12 08:30:00");
insert into patients values(2,"Potter","1998-01-24","2021-08-12 20:30:00");
select * from patients;
select adddate(dob,interval 31 day) from patients; # to add 31 days within the dob column
select curdate(); # gives today's date
select now();# gives today's date and current time
insert into patients values(3,"Granger","1995-08-04",now());
select 1*2; #select methods
select 1+null; #null is unknown value
select 5=5;
insert into patients values(4,"hhh","1958-05-04",now()),(5,"sadsaf",null,now()); #insert methods
insert into patients (pid,name,toa) values(6,"655",now());
set sql_safe_updates=0; # this is used to allow update commands,just need to run it once to enable updates on workbench
update patients set name="hhhii" where pid=4;
select * from patients;
update patients set name="hi",dob="1878-04-15" where pid=4;
update patients set dob="1877-05-05" where dob is null;
delete from patients where pid=6;
select * from students;
delete from students; # delete  without where command
desc students;
insert into students values(3,"Thomas","MySQL",32); # check the order or variable with desc it changes(doesn't stay same like when we created table) 
truncate students; # works like delete without where command however it deletes and recreates the entire table
select * from myemp;
select * from myemp limit 10;
select first_name,last_name,dep_id from myemp limit 10;
select first_name,last_name,salary,salary*0.15 from myemp limit 10; #salalry*.15 shows 15% of total salary
select concat(first_name," ",last_name),salary,salary*0.15 from myemp limit 10;
select concat(first_name," ",last_name) as name,salary,salary*0.15 as 15th_of_sal from myemp limit 10;
select concat(first_name," ",last_name) as name,salary,round(salary*0.15,2) as 15th_of_sal from myemp limit 10;
select distinct dep_id from myemp; # gives unique values
select * from person;
select distinct fname from person;
select distinct lname from person;
select distinct fname,lname from person;#unique values of combination of multiple features
select distinct dep_id from myemp order by dep_id; #ascending
select distinct dep_id from myemp order by dep_id desc; #descending
select * from myemp where salary between 10000 and 15000; #between operator
select * from myemp where salary>=10000 and salary<=15000;
select * from myemp where first_name like "j%"; # use of like to find values (first name starts with j)
select * from myemp where first_name like "%a"; # use of like to find values (first name ends with a)
select * from myemp where first_name like "%a%"; # use of like to find values (first name includes has a)
select * from myemp where first_name like "j____"; # use of like to find values (first name starts with j and has 4 more char after j)
select * from myemp where dep_id IN(30,40,50);
select * from myemp where dep_id=30 or dep_id=40 or dep_id=50;# works same like in
select * from myemp where dep_id=30 and dep_id=40 and dep_id=50; # won't show any thing
select * from myemp where dep_id not IN(30,40,50); # except these values
select year(hire_date) from myemp limit 10;
select min(salary) from myemp;
select max(salary),avg(salary)from myemp;
select count(salary),sum(salary),stddev(salary) from myemp;
select dep_id,mgr_id,avg(salary) from myemp group by dep_id,mgr_id;
select * from movies;
select * from members;
# inner join
select id, title, first_name,last_name from movies inner join members on id=movieid;
select * from movies inner join members on id=movieid;
# left join- gives all records from left table and common once from right table
select id, title, first_name,last_name from movies left join members on id=movieid order by id;
select id, title,ifnull(first_name,"_"),ifnull(last_name,"_") from movies left join members on id=movieid order by id; # just formatting
# right join- gives all records from right table and common once from left table
select id, title, first_name,last_name from movies right join members on id=movieid order by id;
# cross join
select * from meals;
select * from drinks;
select mealname,drinkname,meals.rate,drinks.rate from meals cross join drinks;
select * from meals cross join drinks;
select m.mealname,d.drinkname,m.rate,d.rate from meals as m cross join drinks as d;#assigning short names
# self join
select emp.emp_id, emp.first_name, emp.last_name, mgr.emp_id, mgr.first_name, mgr.last_name from myemp as emp join myemp as mgr on emp.mgr_id=mgr.emp_id;
# constrains
create table students (sid integer unique, sname char(30) not null, age integer check(age>18), course char(30));
desc students;
insert into students values(1,"Thomas",32,"MySQL");
insert into students values(2,"Loona",24,"DS");
select * from students;
alter table students modify column sname char(30) not null default "Kumar";# re-editing sname constrains
desc students;
insert into students(sid,age,course) values(3,27,"Dtd");
select * from authors;
select * from books;
create table books(bookid integer primary key, title varchar(255), aid integer, FOREIGN KEY(aid) REFERENCES authors(authorid) ON UPDATE CASCADE ON DELETE CASCADE); # to give permission to the parent be updated and deleted in last 
# rollback
set autocommit = 0; # to stop server from auto commiting
insert into authors values(10,"Ashu");
rollback; # it only working when auto commit is off as after commiting it doesn't rollback
set autocommit = 1; # to start server to auto commit
create table tt (id char(10));
insert into tt values('a');
insert into tt values('b');
start transaction;# turns off auto commit
insert into tt values('c');
insert into tt values('d');
insert into tt values('e');
select * from tt;
rollback;
delete from tt;
start transaction;
insert into tt values('a');
insert into tt values('b');
savepoint sb;
insert into tt values('c');
insert into tt values('d');
savepoint sd;
insert into tt values('e');
insert into tt values('f');
select * from tt;
rollback to sd;
rollback to sb;
commit;
rollback;# this rollback won't work as we have commited 

# procedures
call myproc(10); #this func was created in stored procedure and applied
call proc_if(15); # it will show multiple outputs so click result1/2.....n or use mysql cmd
call proc_simple_loop(3);
call proc_repeat(3);
call proc_while(3);
call proc_except1(); 

CREATE TABLE products(
pcode varchar(15) default null,
pname varchar(70) default null,
productline varchar(50) not null,
pvendor varchar(15) default null,
qtyInStock smallint(6) default null,
MRP decimal(7,2) default null
);

select * from products;

# functions
select emp_id, first_name,last_name,experience(emp_id) as experience from myemp limit 10;

create table book_sales
(
sale_id int(11) not null auto_increment,
book_id int(11) default null,
title varchar(100) default null,
qty_sold int(11) default null,
tos datetime default null,
primary key (sale_id)
);

select * from book_sales;
select * from books;

alter table books add column sales integer default 10;
update books set sales=sales+3 where bookid=5;
desc book_sales;
update books set sales=sales+4 where bookid=6;# use this after creating a trigger
select * from book_sales;
