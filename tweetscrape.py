import snscrape.modules.twitter as stw
import pandas as pd
import pickle as pkl

query = "(#ABB) lang:en until:2022-12-25 since:2022-09-01"
tweets = list()

# for twee in stw.TwitterSearchScraper(query).get_items():
#     print(vars(twee))
#     break


for i,tweet in enumerate(stw.TwitterSearchScraper(query).get_items()):
    # if i>100:
    #     break
    # print(tweet.date)
    tweets.append([tweet.date, tweet.username, tweet.content])
    
# Creating a dataframe from the tweets list above 
tweets_df = pd.DataFrame(tweets, columns=["Date Created", "Source of Tweet", "Tweets"])
with open("abb_raw_UL_df.pkl","wb") as f:
    pkl.dump(tweets_df, f)
print(tweets_df.head())

#@ABBgroupnews
#@ABBRobotics
#@ABBNorthAmerica
#@ABBItalia - Not Eng
#@ABB_Robotica_ES
#@ABBRobotics_UK
#@ABB_EVCharging
#@ABB_EVCharging
#@ABBMeasurement
#@ABBMotorDriveUS
#@ABBMotorDriveUS
#@ABBenPeru
#@ABBMiddleEast
#@ABBRoboticsUSA