#relations

create table address (
   address_id varchar(10) primary key, 
   street varchar(50) not null, 
   city varchar(50) not null, 
   state char(2) not null, 
   zip char(5) not null);

create table customer (
   customer_id varchar(10) primary key,
   first_name varchar(20) not null,
   last_name varchar(20) not null,
   email varchar(64), 
   phone char(11) not null,
   time_joined timestamp with time zone not null);


create table customer_address (
   id varchar(10) primary key, 
   customer_id varchar(10) not null references customer, 
   address_id varchar(10) not null references address) ;

create table restaurant (
   restaurant_id varchar(10) primary key, 
   restaurant_name varchar(50) not null, 
   cuisine varchar(50) not null, 
   address_id varchar(10) not null references address);

create table menu_item (
   menu_item_id varchar(10) primary key,
   item_name varchar(50) not null, 
   restaurant_id varchar(10) not null references restaurant, 
   description varchar(200), 
   price numeric, 
   is_active boolean);

create table placed_order (
   order_id serial primary key, 
   restaurant_id varchar(10) not null references restaurant, 
   customer_id varchar(10) not null references customer, 
   order_time timestamp with time zone not null, 
   total_price numeric);


create table in_order_item (
   in_order_item_id varchar(10) primary key, 
   order_id serial not null references placed_order, 
   menu_item_id varchar(10) not null references menu_item, 
   quantity numeric);
   
   
#resoning and assumptions

customer table
customer_id is the primary key of the table,so it should be not NULL and unique.
It could set to be varchar(10), which allows app provider to label customers in a flexible manner.
first_name and last_name could set to be varchar(20) not null, 
since lengths of names vary and name is generally required on recipts, no matter it is fake or not.
email could set to be varchar(64) and can be null, 
since the standard maximum length of email is 64 and someone might not have an email.
phone could set to be char(11) and not null, since phone number is required for contact,
here I also assume that all users have U.S phone numbers.
time_joined could set to be timestamp with time zone and not null, 
since it should record the date, and within U.S there are different time zones;
on top of that, as long as you join the app, this cell cannot be null.

address table
address_id is the primary key of the table,so it should be not NULL and unique.
It could set to be varchar(10), which allows app provider to label them in a flexible manner.
Based on my life experience, street, city, state, and zip are all required in a food deliver app.
In this case, I will set them all as not null.
For street and city, varchar(50) should be enough to hold the address.
For state and zip, the standard lengths are 2 and 5, so I will set them as char(2) and char(5).

customer_address table
id is the primary key of the table,so it should be not NULL and unique.
It could set to be varcahr(10), which allows app provider to label them in a flexible manner.
combinations of customer and address is complicated and prone to mistakes.
customer_id is the foreign key references to cutomer table, since this is a piece of information that
tied with customer, which allows people to look up related information in customer table.
address_id is the foreign key references to  address table, since this is a piece of information that
tied with address, which allows people to look up related information in address table.

restaurant table
restaurant_id is the primary key of the table,so it should be not NULL and unique.
It could set to be varchar(10), which allows app provider to label them in a flexible manner.
restaurant_name and cuisine could set to be varchar(50), since lengths of those names vary.
address_id is the foreign key to address table, which can be used to look up restaurant''s specific
address in another relation and to avoid duplicates.

menu_item table
menu_item_id is the primary key of the table,so it should be not NULL and unique.
It could set to be varchar(10), which allows app provider to label them in a flexible manner.
item_name and description lengths vary, so they can be set to varchar(50) and varchar(200),
description is optional so it can be null.
price could set to be numeric and can be NULL, since prices might not be integers and the price range is large;
on top of that, some outlets might change the price and leave it blank.
is_active could set to be boolean and can be NULL, since it only needs two options to tell wether the item is 
available now; on top of that, if the outlet is not sure about the status, it can leave the cell blank.
restaurant_id id a foreign key references to table restaurant, which can be used to tied to a specific restaurant
and to look up restaurant''s specific information and to avoid duplicates.

placed_order table
order_id is the primary key of the table,so it should be not NULL and unique.
It could set to be serial, since based on my life experience, order ids are usually automatically generated
with fixed increment and this process is not done manually.
restaurant_id and customer_id are foreign keys reference to restaurant and cutomer tables respectively.
They can be used to tied to specific restaurant and customer to associate with order price and time,
and to look up specific information of each restaurant and customer who are involved in an order,
and to avoid duplicates.
order_time could set to be timestamp with time zone and not null, 
since it should record the date, and within U.S there are different time zones.
total_price could set to be numeric and can be NULL, since prices might not be integers and the 
range of total price is large;on top of that, some outlets might have some promotions and
will give some complimentary meals, which leave the total price value as null.

in_order_item table
in_order_item_id is the primary key of the table,so it should be not NULL and unique.
It could set to be varchar(10), which allows app provider to label them in a flexible manner.
quantity could set to be numeric and can be null, since this should be a number and sometimes the quantity
might be hard to calculate so the outlet can leave it as null as well.


#Cardinalities

customer_address table is a relationship between address table and customer table
in_order_item table is a relationship between placed_order table and menu_item table
placed_order table is a relationship between customer table and restaurant table
address vs customer_address: one to many
address vs restaurant: one to many
customer_address vs customer: many to one
customer vs placed_order: one to many
placed_order vs restaurant: many to one 
placed_order vs in_order_item: one to many
in_order_item vs menu_item: many to one 
restaurant vs menu_item: one to many


#data retrieval example

select restaurant_name, sum(total_price) as revenue
from restaurant, placed_order
where extract(year from order_time) = 2021
and restaurant.restaurant_id = placed_order.restaurant_id
group by restaurant.restaurant_id
order by revenue desc
limit(1);
