import pandas as pd

data1=pd.read_excel("./data/data1_with_difficulty_levels.xlsx",
                    names=["task_id","gps_0","gps_1","pricing","condition", "difficulty", "difficulty_d"])
data2=pd.read_excel("./data/data2_with_city.xlsx",
                    names=["mem_id", "gps_0", "gps_1", "task_limit", "start_time", "credit", "city"])

data2=data2[data2['city']!="未知"]

E=[0.362613268,0.299414978,0.176031131,0.161940623] # 依次为广东、深圳、东莞和佛山

