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
SELECT 
    *
FROM 
    oly_data
WHERE 
    Sport = 'Athletics'
    AND Medal = 'Gold'
    AND Age = (
        SELECT MAX(Age)
        FROM oly_data
        WHERE Sport = 'Athletics' AND Medal = 'Gold'
);

#10.Find the Ratio of male and female athletes participated in all olympic games.

SELECT 
    SUM(CASE WHEN Sex = 'M' THEN 1 ELSE 0 END) AS Males,
    SUM(CASE WHEN Sex = 'F' THEN 1 ELSE 0 END) AS Females,
    ROUND(
        SUM(CASE WHEN Sex = 'M' THEN 1 ELSE 0 END) / 
        SUM(CASE WHEN Sex = 'F' THEN 1 ELSE 0 END), 
    2) AS Male_to_Female_Ratio
FROM (
    SELECT DISTINCT Name, Sex
    FROM oly_data
) AS unique_athletes;

	
#11.Fetch the top 5 athletes who have won the most gold medals.

SELECT 
	Name,
    COUNT(Medal) AS GOLD_MEDALS
FROM
	oly_data
WHERE
	Medal ='Gold'
GROUP BY
	Name
ORDER BY
	GOLD_MEDALS DESC
LIMIT 5;


#12.Fetch the top 5 athletes who have won the most medals (gold/silver/bronze).

SELECT 
	Name,
    COUNT(Medal) AS MEDALS_COUNT
FROM
	oly_data
WHERE
	Medal IS NOT NULL
GROUP BY
	Name
ORDER BY
	MEDALS_COUNT DESC
LIMIT 5;


#13.Fetch the top 5 most successful countries in olympics. Success is defined by no of medals won.

SELECT 
    noc_regions.region AS Country,
    COUNT(oly_data.Medal) AS Total_MEDALS
FROM
    oly_data
LEFT JOIN
    noc_regions
ON
    oly_data.NOC = noc_regions.NOC
WHERE
    oly_data.Medal IS NOT NULL
GROUP BY
    noc_regions.region
ORDER BY
    Total_MEDALS DESC
LIMIT 5;


#14.List down total gold, silver and broze medals won by each country.

SELECT 
    noc_regions.region AS Country,
    SUM(CASE WHEN oly_data.Medal = 'Gold' THEN 1 ELSE 0 END) AS Gold_Medals,
    SUM(CASE WHEN oly_data.Medal = 'Silver' THEN 1 ELSE 0 END) AS Silver_Medals,
    SUM(CASE WHEN oly_data.Medal = 'Bronze' THEN 1 ELSE 0 END) AS Bronze_Medals,
    COUNT(oly_data.Medal) AS Total_MEDALS
FROM
    oly_data
LEFT JOIN
    noc_regions
ON
    oly_data.NOC = noc_regions.NOC
WHERE
    oly_data.Medal IN ('Gold','Silver','Bronze')
GROUP BY
    noc_regions.region
ORDER BY
    Total_MEDALS DESC

#15.List down total gold, silver and broze medals won by each country corresponding to each olympic games.


SELECT 
    noc_regions.region AS Country,
    oly_data.Games,
    SUM(CASE WHEN oly_data.Medal = 'Gold' THEN 1 ELSE 0 END) AS Gold_Medals,
    SUM(CASE WHEN oly_data.Medal = 'Silver' THEN 1 ELSE 0 END) AS Silver_Medals,
    SUM(CASE WHEN oly_data.Medal = 'Bronze' THEN 1 ELSE 0 END) AS Bronze_Medals,
    COUNT(oly_data.Medal) AS Total_MEDALS
FROM
    oly_data
LEFT JOIN
    noc_regions
ON
    oly_data.NOC = noc_regions.NOC
WHERE
    oly_data.Medal IN ('Gold','Silver','Bronze')
GROUP BY
    noc_regions.region,oly_data.Games
ORDER BY
    Games,Total_MEDALS DESC

#16.Identify which country won the most gold, most silver and most bronze medals in each olympic games.

WITH Medals_Table AS (
    SELECT
        noc_regions.region AS Country,
        oly_data.Games,
        SUM(CASE WHEN oly_data.Medal = 'Gold' THEN 1 ELSE 0 END) AS Gold_Medals,
        SUM(CASE WHEN oly_data.Medal = 'Silver' THEN 1 ELSE 0 END) AS Silver_Medals,
        SUM(CASE WHEN oly_data.Medal = 'Bronze' THEN 1 ELSE 0 END) AS Bronze_Medals,
        COUNT(oly_data.Medal) AS Total_MEDALS
    FROM
        oly_data
    LEFT JOIN
        noc_regions ON oly_data.NOC = noc_regions.NOC
    WHERE
        oly_data.Medal IN ('Gold', 'Silver', 'Bronze')
    GROUP BY
        noc_regions.region, oly_data.Games
)
SELECT 
    Country,
    Games,
    Gold_Medals,
    Silver_Medals,
    Bronze_Medals
FROM 
    Medals_Table AS Mt
WHERE 
    Gold_Medals = (
        SELECT MAX(ST.Gold_Medals)
        FROM Medals_Table AS ST
        WHERE ST.Games = Mt.Games
    )
    OR Silver_Medals = (
        SELECT MAX(ST.Silver_Medals)
        FROM Medals_Table AS ST
        WHERE ST.Games = Mt.Games
    )
    OR Bronze_Medals = (
        SELECT MAX(ST.Bronze_Medals)
        FROM Medals_Table AS ST
        WHERE ST.Games = Mt.Games
    )
ORDER BY 
    Games;


#17.Identify which country won the most gold, most silver, most bronze medals and the most medals in each olympic games.

WITH Medals_Table AS (
    SELECT
        noc_regions.region AS Country,
        oly_data.Games,
        SUM(CASE WHEN oly_data.Medal = 'Gold' THEN 1 ELSE 0 END) AS Gold_Medals,
        SUM(CASE WHEN oly_data.Medal = 'Silver' THEN 1 ELSE 0 END) AS Silver_Medals,
        SUM(CASE WHEN oly_data.Medal = 'Bronze' THEN 1 ELSE 0 END) AS Bronze_Medals,
        COUNT(oly_data.Medal) AS Total_MEDALS
    FROM
        oly_data
    LEFT JOIN
        noc_regions ON oly_data.NOC = noc_regions.NOC
    WHERE
        oly_data.Medal IN ('Gold', 'Silver', 'Bronze')
    GROUP BY
        noc_regions.region, oly_data.Games
)
SELECT 
    Country,
    Games,
    Gold_Medals,
    Silver_Medals,
    Bronze_Medals,
    Total_MEDALS
FROM 
    Medals_Table AS Mt
WHERE 
    Gold_Medals = (
        SELECT MAX(ST.Gold_Medals)
        FROM Medals_Table AS ST
        WHERE ST.Games = Mt.Games
    )
    OR Silver_Medals = (
        SELECT MAX(ST.Silver_Medals)
        FROM Medals_Table AS ST
        WHERE ST.Games = Mt.Games
    )
    OR Bronze_Medals = (
        SELECT MAX(ST.Bronze_Medals)
        FROM Medals_Table AS ST
        WHERE ST.Games = Mt.Games
    )
    OR Total_MEDALS = (
        SELECT MAX(ST.Total_MEDALS)
        FROM Medals_Table AS ST
        WHERE ST.Games = Mt.Games
    )
ORDER BY 
    Games,Total_MEDALS DESC,Gold_Medals DESC,Silver_Medals DESC,Bronze_Medals DESC;


#18.Which countries have never won gold medal but have won silver/bronze medals?

SELECT DISTINCT nr.region AS Country
FROM oly_data oly
JOIN noc_regions nr ON oly.NOC = nr.NOC
WHERE oly.Medal IN ('Silver', 'Bronze')
  AND NOT EXISTS (SELECT 1 FROM oly_data g WHERE g.NOC = oly.NOC AND g.Medal = 'Gold')
ORDER BY Country;


#19.In which Sport/event, India has won highest medals.

SELECT Sport ,COUNT(DISTINCT CONCAT(Games, Medal)) as Highest_Medals 
FROM oly_data 
WHERE NOC='IND' AND Medal  IN ('Gold','Silver','Bronze') 
GROUP BY Sport 
ORDER BY Highest_Medals DESC 
LIMIT 1;

#20.Break down all olympic games where india won medal for Hockey and how many medals in each olympic games.
SELECT 
    Games,
    COUNT(DISTINCT CONCAT(Games ,Event)) AS Medals_Won
FROM 
    oly_data
WHERE 
    NOC = 'IND'
    AND Sport = 'Hockey'
    AND Medal IS NOT NULL
GROUP BY 
    Games
ORDER BY 
    Games;



