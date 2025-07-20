import pandas as pd
import numpy as np

# NLP 结果
a = 0.0000140312
b = 0.0001186378
P0 = 366.9976831255


def cal_pricing(params, h_values, MD, TD, e):
    a, b, P0 = params
    C=np.zeros(len(MD))

    for i in range(len(MD)):
        C[i] = P0 / ((1 + a * MD[i]) * (1 + b * TD[i])) + h_values[i] + e[i]

    return C


data1 = pd.read_excel("../data/data1_all.xlsx")

# === E 映射 ===
city_e_map = {
    "广州": 0,
    "深圳": 5,
    "东莞": -5,
    "佛山": 0
}
data1["E"] = data1["city"].map(city_e_map)
pricing_new = cal_pricing(params=(a, b, P0),
                          h_values=data1["difficulty"].values,
                          MD=data1["MD"].values,
                          TD=data1["TD"].values,
                          e=data1["E"].values)

data1["pricing_new"] = pricing_new
data1.to_excel("../data/data1_result_2.xlsx", index=False)
