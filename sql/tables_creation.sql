create table if not exists users (
    id
        serial 
        primary key,
	timestamp 
		timestamp with time zone 
		not null 
		default current_timestamp,
	user_id 
		integer 
		not null,
	user_name 
		text,
	first_name 
		text, 
	height
		integer,
	weight
		integer,
	age
		integer,
	daily_calories_goal
		integer		
);

create table if not exists meals (
    id
        serial 
        primary key,
    timestamp
        timestamp with time zone 
		not null 
		default current_timestamp, 
    user_id
        integer, 
    dish_name
        text, 
    calories
        integer,
    mass
        integer,
    protein
        float,
    carb
        float,
    fat
        float
);
