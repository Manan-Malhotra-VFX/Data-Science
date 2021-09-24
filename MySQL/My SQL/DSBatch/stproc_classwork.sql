CREATE DEFINER=`root`@`localhost` PROCEDURE `myproc`(aid integer)
BEGIN
    if aid is null then
      select "Null not allowed" as message;
	elseif aid > 10 then
		select "Invalid Author" as message;
	else 
        select "Selecting author details" as message;
		select authorid, name from authors where authorid = aid;
	end if;
    
    select "Outside If" as status;
	
END


-- LOOPS
-- a. Simple Loop

CREATE DEFINER=`root`@`localhost` PROCEDURE `proc_simpleloop`(num integer)
BEGIN
declare cnt integer default 0;
myloop:LOOP

   set cnt = cnt + 1;
   select concat("Hello ", cnt);
   
   -- Termination condition
   if cnt = num then
	leave myloop;
   end if;

END LOOP myloop;
select "Outside loop" as message;
END



--- b. Repeat Until Loop

CREATE DEFINER=`root`@`localhost` PROCEDURE `proc_repeat`(num integer)
BEGIN
declare cnt integer default 0;
myloop:REPEAT

   set cnt = cnt + 1;
   select concat("Hello Repeat ", cnt) as message;
   
-- Termination Condition   
until cnt = num
END REPEAT myloop;
select "Outside loop" as message;
END


-- c. While Do Loop

CREATE DEFINER=`root`@`localhost` PROCEDURE `proc_while`(num integer)
BEGIN
declare cnt integer default 0;
-- Continuation condition
myloop:WHILE cnt < num DO

   set cnt = cnt + 1;
   
   if cnt = 3 then
      iterate myloop;
   end if;
   
   select concat("Hello While ", cnt) as message;
   

END WHILE myloop;
select "Outside loop" as message;
END



-- Exception Handling

CREATE DEFINER=`root`@`localhost` PROCEDURE `make_payment`(cid integer, amt integer)
BEGIN
    DECLARE AmountNull CONDITION FOR 1048;
    declare exit handler for AmountNull
    BEGIN
    insert into fraud (client_id, message, tot)
        values (cid, 'Fraudulent Transaction',  now());
    END;
    
  insert into payments values(cid,amt);
    select "Transaction Completed" as Message;
END

-- Cursor

CREATE DEFINER=`root`@`localhost` PROCEDURE `proc_cursor`()
BEGIN
    declare lcl_pcode varchar(15);
    declare lcl_pvendor varchar(50);
    declare lcl_qty, finished integer default 0;
    
    
  declare prod_cur cursor for 
    select pcode, pvendor, qtyinstock from products;
    declare continue handler for NOT FOUND
    BEGIN
    set finished = 1;
    END;
    
    open prod_cur;
    
    prodloop:LOOP

        fetch prod_cur into lcl_pcode, lcl_pvendor, lcl_qty;
            if finished = 1 then
          leave prodloop;
        end if;
            
            if lcl_qty < 2000 then
          insert into orders (product_code, vendor,status)
                values(lcl_pcode, lcl_pvendor, 'Immediate');
        elseif lcl_qty between 2000 and 5000 then
          insert into orders (product_code, vendor,status)
                values(lcl_pcode, lcl_pvendor, 'Order Soon');
        end if ;
        
    END LOOP prodloop;
    
    close prod_cur;
    select "Process Completed" as Message;
END

-- Function

CREATE DEFINER=`root`@`localhost` FUNCTION `experience`(eid integer) RETURNS int(11)
BEGIN
  declare exp integer default 0;
    
  select year(now())- year(hire_date) into exp
    from myemp where emp_id = eid;

RETURN exp;
END


-- Trigger
CREATE DEFINER=`root`@`localhost` TRIGGER `bk_trg` 
AFTER UPDATE ON `books` 
FOR EACH ROW
BEGIN
    declare qty integer default 0;
    set qty = new.sales - old.sales;
  insert into book_sales (bookid, title, qty_sold, tos)
    values (old.bookid, old.title,qty, now());
    
END


