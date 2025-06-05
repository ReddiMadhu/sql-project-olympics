USE Olympics;

#oly_data is table of olympics data
#noc_regions is table of regions data



#1.How many olympics games have been held?

SELECT COUNT(DISTINCT Games) AS OlympicGames From oly_data;

#2.List down all Olympics games held so far.
  
SELECT DISTINCT Games AS Olympic_Games From oly_data;


#3.Mention the total no of nations who participated in each olympics game?

SELECT DISTINCT Games AS Olympic_Games , COUNT(DISTINCT Noc) as Total_Nations FROM oly_data GROUP BY Games ;


#4.Which year saw the highest and lowest no of countries participating in olympics?

 WITH  NationsCount AS(
	SELECT 
		DISTINCT Games AS Olympic_Games , 
        COUNT(DISTINCT Noc) as Total_Nations 
	FROM 
		oly_data 
	GROUP BY 
    Games
)
SELECT 
	Olympic_Games, 
    Total_Nations, 
	CASE 
		WHEN Total_Nations = (SELECT MAX(Total_Nations) from NationsCount) THEN 'Highest'
		WHEN Total_Nations = (SELECT MIN(Total_Nations) from NationsCount) THEN 'Lowest'
	END AS Participation_Level
FROM 
	NationsCount
WHERE
	Total_Nations = (SELECT MAX(Total_Nations) from NationsCount)
    or
	Total_Nations = (SELECT MIN(Total_Nations) from NationsCount);


#5.Which nation has participated in all of the olympic games?

SELECT 
    nr.region AS Country
FROM 
    oly_data od
JOIN 
    noc_regions nr ON od.NOC = nr.NOC
GROUP BY 
    od.NOC, nr.region
HAVING 
    COUNT(DISTINCT od.Games) = (
        SELECT COUNT(DISTINCT Games) 
        FROM oly_data
    );


#6.Identify the sport which was played in all summer olympics.
SELECT 
	sport
FROM 
	oly_data
WHERE
	season='summer'
GROUP BY
	sport,years,season
HAVING 
	COUNT(sport) =(SELECT COUNT(DISTINCT years) FROM oly_data WHERE season ='summer');


7.Which Sports were just played only once in the olympics?

SELECT 
	sport
FROM
	oly_data
GROUP BY
	sport,Games
HAVING
	COUNT(Sport)=1;

#8.Fetch the total no of sports played in each olympic games.
	
SELECT 
	Games,
	COUNT(DISTINCT Sport) AS No_Of_Sports
FROM
	oly_data
GROUP BY
	Games;


#9.Fetch details of the oldest athletes to win a gold medal.


10.Find the Ratio of male and female athletes participated in all olympic games.
11.Fetch the top 5 athletes who have won the most gold medals.
12.Fetch the top 5 athletes who have won the most medals (gold/silver/bronze).
13.Fetch the top 5 most successful countries in olympics. Success is defined by no of medals won.
14.List down total gold, silver and broze medals won by each country.
15.List down total gold, silver and broze medals won by each country corresponding to each olympic games.
16.Identify which country won the most gold, most silver and most bronze medals in each olympic games.
17.Identify which country won the most gold, most silver, most bronze medals and the most medals in each olympic games.
18.Which countries have never won gold medal but have won silver/bronze medals?
19.In which Sport/event, India has won highest medals.
20.Break down all olympic games where india won medal for Hockey and how many medals in each olympic games.




