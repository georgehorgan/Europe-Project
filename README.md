# European Analysis Project

This project aims to answer questions about Europe by using statistical methods and showing the data through visualisations. This project is aimed to reflect my ability to analyse data, it isn’t intended to be entirely factual to this present day.

I have put this data together myself from the following online sources:

- EU membership status : https://europa.eu/european-union/about-eu/countries_en
- Euro currency users : https://europa.eu/european-union/about-eu/countries_en#countries-using-the-euro
- Population : https://www.worldometers.info/world-population/population-by-country/
- GDP per country : https://www.worldometers.info/gdp/gdp-by-country/
- Defence Budget : https://www.globalfirepower.com/defense-spending-budget.asp
- External Debt : https://www.globalfirepower.com/external-debt-by-country.asp
- Net exports : https://en.wikipedia.org/wiki/List_of_countries_by_net_exports
- Quality of life index : https://www.numbeo.com/quality-of-life/rankings_by_country.jsp

Disclaimer: This data is collected from the sources above, therefore the credibility of the results within this project are based on my findings using this data.

I haven’t included all countries in Europe once I had finished cleaning the data as data wasn’t found for their country, these countries include:

- Andorra
- Kosovo
- Liechtenstein
- Luxembourg
- Monaco
- San Marino
- Vatican City

I filled data in for countries which also had null values which I thought would be more impactful on the data should it have been missing, they are:

- Cyprus
- Iceland
- Malta

CSV files used:

[European Dataset.csv](https://www.dropbox.com/scl/fi/fcq6us7ba0ir50fsjc8o6/European-Dataset.csv?rlkey=1x8y3z5t95tdwd3cgu2nzptai&dl=0)

[world_population.csv](https://www.dropbox.com/scl/fi/7cpvz15ah8z18dfoa2z8h/world_population.csv?rlkey=dyq1lxflgptcsa76rtigrni62&dl=0)

### **Import the data**

```python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# plt.rcParams["figure.figsize"] = [5, 5]
euro = pd.read_csv(r"C:\Users\georg\DataPython\European Dataset.csv")
euro.head()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img1.png)

### **Clean the data**

```python
null_data = euro[euro.isnull().any(axis = 1)]
null_data.count()
```

```
Country Name                        10
EU Membership Status                10
Euro Currency Users                 10
Population                           8
GDP (Billions)($)                    6
Military Spending (Millions)($)      1
External Debt (Billions)($)          0
Net Exports (Millions)($)            9
Quality of life index (mid 2020)     3
dtype: int64
```

We can see the missing data is mostly for the small nation states, therefore I am going to drop the countries which don't have a quality of life index value.

For Cyprus, Iceland and Malta I will find data from an external source to fill the values.

Sources:

- External Debt Cyprus : https://www.ceicdata.com/en/indicator/cyprus/external-debt
- External Debt Iceland : https://tradingeconomics.com/iceland/external-debt#:~:text=Looking%20forward%2C%20we%20estimate%20External,according%20to%20our%20econometric%20models.
- Military Spending Iceland : https://www.icelandreview.com/news/iceland-ups-defence-budget-by-37/
- Military Spending Malta : https://en.wikipedia.org/wiki/Armed_Forces_of_Malta
- External Debt Malta : https://www.ceicdata.com/en/indicator/malta/external-debt--of-nominal-gdp#:~:text=The%20country%27s%20External%20Debt%20reached,USD%20bn%20in%20Jun%202020.

```python
# Cyprus
euro.at[10, "External Debt (Billions)($)"] = 209

# Iceland

euro.at[20, "Military Spending (Millions)($)"] = 17
euro.at[20, "External Debt (Billions)($)"] = 19

# Malta

euro.at[28, "Military Spending (Millions)($)"] = 64
euro.at[28, "External Debt (Billions)($)"] = 102
euro.dropna(inplace = True)
```

```python
check_null = euro[euro.isnull().any(axis = 1)]
check_null
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img2.png)

```python
euro.set_index("Country Name", inplace = True)
euro["Population"] = euro["Population"].astype(int)
euro["GDP (Billions)($)"] = euro["GDP (Billions)($)"].astype(int)
euro["Military Spending (Millions)($)"] = euro["Military Spending (Millions)($)"].astype(int)
euro["External Debt (Billions)($)"] = euro["External Debt (Billions)($)"].astype(int)
euro["Net Exports (Millions)($)"] = euro["Net Exports (Millions)($)"].astype(int)
```

### **Visualise the data**

What percentage of European nations are in the European Union?

```python
eu_members = euro["EU Membership Status"].value_counts()
eu_members
Member          26
Not a Member    12
Candidate        5
Name: EU Membership Status, dtype: int64
explode = (0.1, 0.1, 0.1)
plt.pie(eu_members, labels = eu_members.index, startangle = 90, 
        shadow = True, explode = explode, autopct='%1.1f%%')
```

Here we can see that 60.5% are members, whilst only 28% of Europe are not at all.

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img3.png)

What percentage of members are in the monetary union?

```python
EU_countries = euro[euro["EU Membership Status"] == "Member"]
euro_users = EU_countries["Euro Currency Users"].value_counts()
euro_users
```

```
Yes    18
No      8
Name: Euro Currency Users, dtype: int64
```

```python
explode = (0, 0.1)
plt.pie(euro_users, labels = euro_users.index, startangle = 90, 
        shadow = True, explode = explode, autopct='%1.1f%%')
```

69% of members in the union are using the Euro as their official currency.

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img4.png)

```python
plt.figure(figsize = (10, 5))
sns.barplot(x = euro.index, y = euro["Population"])
plt.xlabel("Country Name")
plt.xticks(rotation = 90)
plt.ylabel("Population")
plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img5.png)

Fig [1] - Population of European nations against one another. 

```python
w_euro = euro.loc[["Iceland", "Ireland", "United Kingdom", "France", "Spain ", "Italy", "Portugal",
                    "Sweden", "Finland", "Norway", "Denmark", "Netherlands", "Belgium", "Germany",
                   "Czechia", "Switzerland", "Austria", "Malta "]]

e_euro = euro.loc[["Russia", "Estonia", "Latvia", "Lithuania", "Belarus", "Ukraine",
                   "Romania", "Moldova", "Bulgaria", "North Macedonia", "Greece", "Turkey",
                   "Cyprus", "Montenegro", "Bosnia & Herzegovina", "Croatia", "Slovakia",
                   "Slovenia", "Hungary", "Poland", "Armenia", "Azerbaijan", "Georgia",
                   "Albania", "Serbia"]]
```

Creating two data frames that will allow comparisons between western and eastern Europe.

```python
sns.barplot(x = e_euro.index, y = e_euro["GDP (Billions)($)"])
plt.xticks(rotation = 90)
plt.xlabel("Eastern European Nations")
plt.ylabel("GDP in Billions of USD")
plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img6.png)

Fig [2] - GDP in Billions of USD against selected eastern European nations. 

```python
x = e_euro.index
y1 = e_euro["GDP (Billions)($)"]
y2 = e_euro["Military Spending (Millions)($)"]

plt.figure(figsize = (12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x = x, y = y1)
plt.xticks(rotation = 90)
plt.xlabel("Eastern European Nation")
plt.ylabel("GDP in Billions of USD")

plt.subplot(1, 2, 2)
sns.barplot(x = x, y = y2)
plt.xticks(rotation = 90)
plt.xlabel("Eastern European Nation")
plt.ylabel("Defence Budget in Millions of USD")

plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img7.png)

Fig [3] - Visualising the difference between GDP and spending on defence against selected eastern European nations.

```python
euro["GDP Spent On Defence (%)"] = ((euro["Military Spending (Millions)($)"] / 1000) / 
                                    euro["GDP (Billions)($)"] * 100)
```

```python
euro["GDP Spent On Defence (%)"].sort_values(ascending = False)
```

```
Country Name
Armenia                 12.590909
Azerbaijan               7.012500
Ukraine                  4.821429
Russia                   3.041825
Estonia                  2.634615
Latvia                   2.413333
Greece                   2.386207
Romania                  2.382075
Lithuania                2.355319
Poland                   2.281369
Slovakia                 2.232632
Turkey                   2.230047
Serbia                   2.212195
Georgia                  2.180000
United Kingdom           2.088704
Albania                  1.923077
Bulgaria                 1.860345
Norway                   1.799248
Montenegro               1.625000
France                   1.606659
Portugal                 1.533333
Cyprus                   1.522727
Netherlands              1.494465
Hungary                  1.485714
Croatia                  1.454545
Denmark                  1.442424
Italy                    1.430041
Finland                  1.416667
Czechia                  1.374537
Germany                  1.353913
Slovenia                 1.210417
Sweden                   1.180224
Belarus                  1.153704
Spain                    1.149163
Belgium                  0.994141
North Macedonia          0.981818
Bosnia & Herzegovina     0.916667
Austria                  0.810552
Switzerland              0.736377
Malta                    0.533333
Moldova                  0.375000
Ireland                  0.262840
Iceland                  0.070833
Name: GDP Spent On Defence (%), dtype: float64
```

```python
sns.barplot(x= w_euro.index, y= w_euro["GDP (Billions)($)"])
plt.xticks(rotation= 90)
plt.xlabel("Western European Nations")
plt.ylabel("GDP in Billions of USD")
plt.show()
```

Fig [4] - GDP in Billions of USD against selected western European nations. 

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img8.png)

```python
sns.barplot(x = w_euro.index, y = w_euro["Military Spending (Millions)($)"])
plt.xticks(rotation = 90)
plt.xlabel("Western European Nations")
plt.ylabel("Defence Budget in Millions of USD")
plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img9.png)

Fig [5] - Defence budget in USD against selected western European countries.

```python
non_EU_countries = euro[euro["EU Membership Status"] != "Member"]
EU_countries = euro[euro["EU Membership Status"] == "Member"]
```

```python
x1 = EU_countries.index
x2 = non_EU_countries.index
y1 = EU_countries["GDP Spent On Defence (%)"]
y2 = non_EU_countries["GDP Spent On Defence (%)"]

plt.figure(figsize = (12,5))
plt.subplot(1, 2, 1)
sns.barplot(x = x1, y = y1)
plt.xticks(rotation = 90)
plt.xlabel("European Union Members")
plt.ylabel("Percentage of GDP Spent on Defence")

plt.subplot(1, 2, 2)
sns.barplot(x = x2, y = y2)
plt.xticks(rotation = 90)
plt.xlabel("Non European Union Members")
plt.ylabel("Percentage of GDP Spent on Defence")

plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img10.png)

Fig [6] - Percentage of GDP spent on defence, EU members vs Non-members

The combined sum of military spending - EU vs Non-EU

```python
total_non_eu= non_EU_countries["Military Spending (Millions)($)"].sum()
total_eu= EU_countries["Military Spending (Millions)($)"].sum()
data_eu= total_eu, total_non_eu
data_eu
```

```
(208443, 146361) 

208 billion USD vs 146 billion USD
```

```python
res = data_eu[0] - data_eu[1] 
res / data_eu[1] * 100
```

```
42.41703732551704 

42% higher in EU countries vs non-EU
```

Combined population of EU members vs non-EU members

```python
total_non_eu= non_EU_countries["Population"].sum()
total_eu= EU_countries["Population"].sum()
sum_eu= total_eu, total_non_eu
sum_eu
```

```
(444624536, 404494276) 

445 million people vs 404 million people
```

```python
res = sum_eu[0] - sum_eu[1]
res / sum_eu[1] * 100
```

```
9.9210946559847

10% higher population in the EU than not in the EU
```

Average military spending, EU vs non-EU

```python
total_non_eu = non_EU_countries["Military Spending (Millions)($)"].mean()
total_eu = EU_countries["Military Spending (Millions)($)"].mean()
data_eu = total_eu, total_non_eu
data_eu
```

```
(8017.038461538462, 8609.470588235294)

8 Billion vs 8.6 Billion
```

```
res= data_eu[0]- data_eu[1]
res/ data_eu[1]* 100
```

```
-6.881167902546539

6.9% less in EU than non-EU
```

```python
plt.figure(figsize = (12, 5))
sns.barplot(x = euro.index, y = euro["External Debt (Billions)($)"])
plt.xticks(rotation = 90)
plt.xlabel("European Nation")
plt.ylabel("External Debt in Billions of USD")
plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img11.png)

Fig [7] - European nations measured on their external debt

The measure of exporters and importers

```python
euro["Net Exports (Millions)($)"].sort_values(ascending = False)
```

```
Country Name
Germany                 297000
Ireland                 129070
Russia                  124100
Netherlands              91000
Italy                    72400
Switzerland              50100
Norway                   22900
Denmark                  18590
Sweden                   14900
Czechia                  10000
Azerbaijan                6956
Hungary                   5440
Belgium                   3000
Slovakia                  2610
Slovenia                  1530
Poland                    -400
Iceland                  -1074
Armenia                  -1128
Estonia                  -1290
Belarus                  -2000
North Macedonia          -2170
Montenegro               -2196
Moldova                  -2406
Finland                  -2500
Malta                    -2564
Latvia                   -2790
Albania                  -3123
Lithuania                -4010
Bulgaria                 -4240
Bosnia & Herzegovina     -4310
Georgia                  -4313
Serbia                   -4550
Cyprus                   -4979
Austria                  -6500
Ukraine                  -7570
Croatia                  -8810
Portugal                -11720
Romania                 -13200
Greece                  -21000
Spain                   -31900
Turkey                  -39500
France                  -73100
United Kingdom         -166000
Name: Net Exports (Millions)($), dtype: int32
```

Net between EU and non-EU countries

```python
net_non_eu= non_EU_countries["Net Exports (Millions)($)"].mean()
net_eu= EU_countries["Net Exports (Millions)($)"].mean()
total= net_non_eu, net_eu
total
```

```
(-2134.3529411764707, 17559.115384615383)

-2.1 billion USD non-EU vs 17.6 billion USD EU
```

```python
total[1] - total[0] 
total
```

```
19693.468325791855

Difference in EU net exports and non-EU net exports
```

Creating a quality of life dataframe

```python
QOL = euro["Quality of life index (mid 2020)"].sort_values(ascending = False)
QOL
```

```
Country Name
Denmark                 192.53
Switzerland             190.92
Finland                 186.40
Netherlands             184.18
Austria                 181.68
Iceland                 180.74
Germany                 177.25
Estonia                 175.99
Norway                  174.55
Sweden                  172.18
Slovenia                169.81
Spain                   167.05
Portugal                162.46
United Kingdom          161.20
Lithuania               159.77
Croatia                 156.77
Czechia                 154.70
France                  150.68
Ireland                 150.16
Slovakia                149.93
Belgium                 149.75
Latvia                  148.68
Cyprus                  146.90
Italy                   138.97
Montenegro              134.94
Belarus                 133.52
Romania                 131.69
Greece                  131.51
Hungary                 128.40
Bulgaria                127.14
Turkey                  126.40
Poland                  125.20
Bosnia & Herzegovina    123.05
Malta                   121.82
Georgia                 116.37
Serbia                  116.08
Armenia                 114.09
Moldova                 110.53
North Macedonia         108.99
Ukraine                 105.26
Azerbaijan              102.69
Russia                  101.57
Albania                  99.20
Name: Quality of life index (mid 2020), dtype: float64
```

```python
plt.figure(figsize = (12, 5))
sns.barplot(x = QOL.index, y = QOL)
plt.xticks(rotation = 90)
plt.xlabel("European Nation")
plt.ylabel("Quality of Life Index")
plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img12.png)

Fig [8] - Quality of life index visualised in order from highest to lowest. 

```python
x1 = EU_countries.index
x2 = non_EU_countries.index
y1 = EU_countries["Quality of life index (mid 2020)"]
y2 = non_EU_countries["Quality of life index (mid 2020)"]

plt.figure(figsize = (12,5))
plt.subplot(1, 2, 1)
sns.barplot(x = x1, y = y1)
plt.xticks(rotation = 90)
plt.xlabel("European Union Members")
plt.ylabel("Quality of life Ranking")

plt.subplot(1, 2, 2)
sns.barplot(x = x2, y = y2)
plt.xticks(rotation = 90)
plt.xlabel("Non European Union Members")
plt.ylabel("Quality of life Ranking")

plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img13.png)

Fig [9] - Quality of life index, EU vs non-EU.

Mean happiness between The EU and non EU nations

```python
mean_happiness = y1.mean(), y2.mean()
mean_happiness   
```

```
(155.44615384615383, 129.41764705882352)

155 EU vs 129 non-EU
```

```python
x1 = w_euro.index
x2 = e_euro.index
y1 = w_euro["Quality of life index (mid 2020)"]
y2 = e_euro["Quality of life index (mid 2020)"]

plt.figure(figsize = (12,5))
plt.subplot(1, 2, 1)
sns.barplot(x = x1, y = y1)
plt.xticks(rotation = 90)
plt.xlabel("Western European Nations")
plt.ylabel("Quality of life Ranking")

plt.subplot(1, 2, 2)
sns.barplot(x = x2, y = y2)
plt.xticks(rotation = 90)
plt.xlabel("Eastern European Nations")
plt.ylim(0, 200)
plt.ylabel("Quality of life Ranking")

plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img14.png)

Fig [10] - Quality of life index, western vs eastern Europe. 

```python
QOL_w_euro= w_euro["Quality of life index (mid 2020)"].mean()
QOL_e_euro= e_euro["Quality of life index (mid 2020)"].mean()
d= {"Mean Quality of Life": [QOL_w_euro, QOL_e_euro]}
df= pd.DataFrame(data= d, index= ["Western Europe", "Eastern Europe"])
df
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img15.png)

```python
plt.pie(df["Mean Quality of Life"], labels= df.index, startangle= 90,
        shadow=True, autopct='%1.1f%%')
```

Fig [11] - Pie visualisation of the quality of life index

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img16.png)

Creating another dataset to use for further insights.

```python
europe = euro.copy() 
```

```python
europe["EU Membership Status"] = europe["EU Membership Status"].map({"Not a Member": 0, 
                                                                    "Candidate": 0, 
                                                                    "Member": 1})
```

```python
europe["EU Membership Status"].value_counts()
```

```
1    26
0    17
Name: EU Membership Status, dtype: int64
```

```python
europe["Euro Currency Users"] = europe["Euro Currency Users"].map({"Yes": 1, "No": 0})
europe["Euro Currency Users"].value_counts()
```

```
No    25
Yes    18
Name: Euro Currency Users, dtype: int64
```

### Regression Analysis

```python
import statsmodels.api as sm

y = europe["GDP (Billions)($)"]
x1 = europe[["Military Spending (Millions)($)", "External Debt (Billions)($)"]]
x = sm.add_constant(x1)

results = sm.OLS(y, x).fit()
results.summary()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img17.png)

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img18.png)

Predicting nations outside of Europe GDP's based on their military spending and external debt:

MILITARY SPENDING (Millions)($):

- Japan: 49000
- Canada: 22500
- Australia: 26300
- Argentina: 4200
- Ivory Coast: 550

EXTERNAL DEBT (Billions)($):

- Japan: 3240
- Canada: 1608
- Australia: 1714
- Argentina: 215
- Ivory Coast: 13

```python
x.head()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img19.png)

```python
compare = pd.DataFrame(
{"const": 1, 
 "Military Spending (Millions)($)": [49000, 22500, 26300, 4200, 550], 
 "External Debt (Billions)($)": [3240, 1608, 1714, 215, 13]}
)
```

```python
compare.rename({0: "Japan", 
								1: "Canada", 
								2: "Australia", 
								3: "Argentina", 
								4: "Ivory Coast"})

prediction = results.predict(compare)

predicted_values = pd.DataFrame({"Predicted GDP (Billions)($)": prediction})
joined = compare.join(predicted_values)
joined.rename(index = {0: "Japan", 
											 1: "Canada", 
										   2: "Australia", 
											 3: "Argentina", 
											 4: "Ivory Coast"})
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img20.png)

```python
actual_values = pd.DataFrame({"Actual GDP (Billions)($)": [4872, 1647, 1323, 637, 37]})
joined = joined.join(actual_values)
joined.rename(index = {0: "Japan", 
											 1: "Canada", 
											 2: "Australia", 
											 3: "Argentina", 
											 4: "Ivory Coast"})
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img21.png)

```python
y = europe["Quality of life index (mid 2020)"]
x1 = europe.drop(["Quality of life index (mid 2020)"], axis = 1)
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
results.summary()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img22.png)

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img23.png)

```python
europe["GDP per Capita ($)"] = europe["GDP (Billions)($)"] / europe["Population"]
europe["GDP per Capita ($)"] = europe["GDP per Capita ($)"] * 1000000000
europe["GDP per Capita ($)"] = europe["GDP per Capita ($)"].astype(int)
```

```python
y = europe["Quality of life index (mid 2020)"]
x1 = europe[["GDP per Capita ($)", "EU Membership Status"]]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
```

```python
plt.scatter(europe["GDP per Capita ($)"], y, c = europe["EU Membership Status"], cmap = "RdYlGn_r")
yhat_no = 111.7948 + 0.0009 * europe["GDP per Capita ($)"] + 0
yhat_yes = 111.7948 + 0.0009 * europe["GDP per Capita ($)"] + 16.9577
fig = plt.plot(europe["GDP per Capita ($)"], yhat_no, lw = 2, c = "green")
fig = plt.plot(europe["GDP per Capita ($)"], yhat_yes, lw = 2, c = "red")
plt.xlabel("GDP per Capita ($)")
plt.ylabel("Quality of life index (mid 2020)")
plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img24.png)

### K-Means Clustering

```python
plt.scatter(europe["GDP per Capita ($)"], europe["Military Spending (Millions)($)"])
plt.xlabel("GDP per Capita in USD")
plt.ylabel("Defence Budget in Millions of USD")
plt.show()
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img25.png)

```python
data = europe[["GDP per Capita ($)", "Military Spending (Millions)($)"]]
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img26.png)

```python
from sklearn.cluster import KMeans
kmeans = KMeans(3)
kmeans.fit(data)
```

```python
clusters = data.copy()
clusters["Cluster prediction"] = kmeans.fit_predict(data)
plt.scatter(clusters["GDP per Capita ($)"], clusters["Military Spending (Millions)($)"], 
           c = clusters["Cluster prediction"], cmap = "rainbow")
plt.xlabel("GDP per Capita in USD")
plt.ylabel("Defence Budget in Millions of USD")
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img27.png)

```python
from sklearn import preprocessing
x_scaled = preprocessing.scale(data)
x_scaled
```

```python
wcss = []

for i in range(1, 10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
```

```python
plt.plot(range(1, 10), wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS value")
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img28.png)

To find the optimal number of clusters, we use the elbow method in which we find the Within-Cluster-Sum-of-Squares value (WCSS) which tells you the distance between data points and their centroids, a centroid is essentially the centre of gravity between a group of points and all the points will gather to their closest centroid. We create a plot of the WCSS value against the number of clusters we can use. If there is a sharp bend in the resultant graph, this tells us that it's corresponding x value is the optimal number of clusters to use.

```python
kmeans_new = KMeans(3)
kmeans_new.fit(x_scaled)
clust_new = data.copy()
clust_new["Cluster prediction"] = kmeans_new.fit_predict(x_scaled)
plt.scatter(clust_new["GDP per Capita ($)"],clust_new["Military Spending (Millions)($)"], 
            c = clust_new["Cluster prediction"], cmap = "rainbow")
plt.xlabel("GDP per Capita")
plt.ylabel("Military Spending in Millions of USD")
```

![](https://github.com/georgehorgan/Europe-Project/blob/main/images/europe_img29.png)

We can identify these clusters into three groups.
Poorer nations who don’t spend big (bottom right), richer nations who don’t spend big (bottom left) and the big economies who do spend big (top centre).

The poorer nations who don’t spend big, we can assume most of them are central and eastern European countries judging by the previous data we have found. Possibly in the area of the ex-Soviet Union members and Balkan nations.
The bottom right countries I believe we could find those in western and central Europe, countries I expect you would find high on the quality of life index, such as Denmark, Finland, Austria, Ireland. These types of countries don’t spend a lot of money on their military but are rich in GDP per capita.
Lastly, we have the big economies who do spend big. If I had to guess I would say Russia is the point at the top left of the graph, only because of their low GDP per Capita and high spending on defence. Other nations in this bloc would be Europe’s big economies such as the UK, France and Germany for example.
