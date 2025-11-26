
#region PythonAi1
from statistics import variance

import numpy
import pandas as pd



# Lesson1 Exercices

df = pd.read_csv('Preview__houses_day1__first_20_rows_.csv')

# 1.	Sürətli baxış:
#
# ○	head(5), tail(5) və sample(3) ilə datasetə bax.

# print(df.head(5))
# print(df.tail(5))
#print(df.sample(3))

# 2.	Struktur yoxlaması:
#
# ○	info() nəticəsinə əsasən hansı sütunlarda boş dəyər var?
#
# ○	Hər sütunun dtype-ını qeyd et.


# df.info()


# 3.	Statistik icmal:
#
# ○	describe() nəticəsinə bax və Area_m2, Price_AZN üçün mean, median, std dəyərlərini müqayisə et.

# print(df.describe())


#
# 4.	Tip düzəlişi:
#
# ○	Price_AZN-də string dəyər olub-olmadığını yoxla (var!). Bunu rəqəmə çevir (error='coerce' istifadə edə bilərsən).
#
# ○	Çevirmədən əvvəl və sonra df.dtypes müqayisə et.

# if df['Price_AZN'].dtype == object:   # Pandas-da string-lər object tipində olur
#     print("string-dir")
# else:
#     print("numeric-dir")



#
# 5.	Qiymət outlier-ləri (təxmini):
#
# ○	Price_AZN-i sortla (azalan). İlk 10 sətirdə outlier təsiri verən hansı dəyərləri görürsən?
#
# ○	“Ən bahalı 3 m²” ideyasını qeyd et (hələ hesablamaya ehtiyac yoxdur).


# data=df.sort_values(by='Price_AZN',ascending=False)
# print(data.Price_AZN)


#
# 6.	Kateqorik balans:
#
# ○	District üçün value_counts() çıxar.
#
# ○	Sual: Hansi rayon(lar) çox/az təmsil olunub? Bu imbalance nə yarada bilər?

# print(df['District'].value_counts())




# 7.	Rooms distribusiyası:
#
# ○	Rooms üçün value_counts().sort_index() çıxar.
#
# ○	Sual: 1, 2, 3 otaqlılarda paylanma necədir?

# print(df['Rooms'].value_counts().sort_index())


#
# 8.	Mean vs Median (Price):
#
# ○	Price_AZN üçün mean və median müqayisə et.

#
# ○	Fikir: Niyə fərq var? Outlier-lərin rolu nədir?


# print(df['Price_AZN'].mean())
# print(df['Price_AZN'].median())



# 9.	Mode və yayılma ölçüləri:
#
# ○	Rooms üçün mode (ən çox görünən), Price_AZN üçün variance və std hesabla.

#
# ○	Yekun: Hansı rayonun qiymətlərində yayılma daha çox ola bilər (hipotez)?

# print(df['Rooms'].mode())

# print(df['Price_AZN'].std())
# print(df['Price_AZN'].var())
# #



# 10.	Filter + seçim:
#
# ○	Rooms >= 3 və Area_m2 >= 100 olan sətirləri seç. Bu alt-kəsikdə Price_AZN orta qiyməti neçədir?

# filtered_df = df[(df['Rooms'] >= 3) & (df['Area_m2'] >= 100)]
# print(filtered_df['Price_AZN'].mean())
#


# 11.	District üzrə mərkəz ölçüləri:
#
# ○	groupby("District")["Price_AZN"].agg(["mean","median","count"]) hesabla.

# data=df.groupby('District')['Price_AZN'].agg(['mean','median','count'])
# print(data)
# ○	Sual: Harada median mean-dən xeyli fərqlənir və niyə?
#
# 12.	Outlier aşkarlanması (IQR):
#
# ○	Price_AZN üçün Q1, Q3, IQR, lower/upper bound hesabla və “çıxışda” qalan sətirləri göstər.

# q1=df['Price_AZN'].quantile(0.25)
# q3=df['Price_AZN'].quantile(0.75)
# print(q1)
# print(q3)
# iqr=q3-q1
# lower,upper=q1-1.5*iqr,q3+1.5*iqr
# iqr_outliers=df[(df['Price_AZN']<lower) | (df['Price_AZN']>upper)]
# print(iqr_outliers)


#
# ○	Qeyd: Bunları avtomatik filtr kimi tətbiq et.
#
# 13.	Outlier aşkarlanması (Z-score):----------------------------------------------------
#
# ○	zscore(Price_AZN) hesabla və |z|>3 sətirləri tap.



# Məqsəd: Hər bir dəyərin orta dəyərdən neçə standart sapma uzaqda olduğunu göstərmək.
# z=x−mean/std
#
# ○	Nəticə: IQR və Z-score nəticələri eyni sətirləri göstərirmi?
#
# 14.	Top 10 ən bahalı və ən ucuz evlər:
#
# ○	İki ayrı cədvəl ilə göstər.
#
# ○	Qeyd: Outlier-ləri ayrıca qeyd et (əlavə sütun “IsOutlier” ola bilər).


# top_expensive = df.sort_values(by='Price_AZN', ascending=False).head(10)
# print("Top 10 Ən Bahalı Evlər:")
# print(top_expensive)


# top_cheap = df.sort_values(by='Price_AZN', ascending=True).head(10)
# print("\nTop 10 Ən Ucuz Evlər:")
# print(top_cheap)
# #
# 15.	Room-Effect ideyası:
#
# ○	Rooms ilə Price_AZN arasında “orta qiymətə təsir”i hiss etmək üçün groupby("Rooms")["Price_AZN"].median() çıxar.
#
# ○	Qeyd: Median niyə daha məntiqli ola bilər?

# data=df.groupby('Rooms')['Price_AZN'].median()
# print(data)
#
# 16.	Price per m² (ppm):
#
# ○	Yeni sütun: ppm = Price_AZN / Area_m2 (təhlükəsiz bölmə və boş dəyərləri nəzərə al!).
#
# ○	ppm-ə görə ilk 10 sətiri çıxar. Sual: Hansı rayon önə çıxır?

# df['PPM'] = df['Price_AZN'] / df['Area_m2']
# df.to_csv("Preview__houses_day1__first_20_rows_.csv", index=False)
#
# sorted_ppm=df.sort_values(by=['PPM'], ascending=False)
# print(sorted_ppm)
# 17.	Kateqorik təmizləmə (map):
#
# ○	District-ləri region map ilə qrupla (məs: Sabayil=“Prime”, Yasamal/Nizami/Nasimi/Nerimanov=“Central”, Khatai/Binagadi=“Outer”).
#
# ○	groupby("region")["Price_AZN"].median() müqayisə et.


# district_to_region = {
#     "Sabayil": "Prime",
#     "Yasamal": "Central",
#     "Nizami": "Central",
#     "Nasimi": "Central",
#     "Nerimanov": "Central",
#     "Khatai": "Outer",
#     "Binagadi": "Outer"
# }
#
# # Yeni sütun əlavə edirik
# df["Region"] = df["District"].map(district_to_region)
#
# median_prices = df.groupby("Region")["Price_AZN"].median()
# print(median_prices)

#
# 18.	Tip problemləri və boşluqların təsiri:
#
# ○	Price_AZN-də boş/NaN olan sətirləri tap; bunların District/Rooms/Area paylanmasını təhlil et.
#
# ○	Qeyd: Boş dəyərləri necə imputasiya edərdin (niyə median daha yaxşıdır)?
# nan_rows = df[df['Price_AZN'].isna()]
# print(nan_rows)

#
# 19.	Simulyasiya “təmiz” qiymət medianı:
#
# ○	Outlier-ləri IQR ilə filtr edib təmiz subset üçün Price_AZN medianını hesabla.
#
# ○	Təmiz medianı ümumi medianla müqayisə et.


# Q1 = df['Price_AZN'].quantile(0.25)
# Q3 = df['Price_AZN'].quantile(0.75)
# IQR = Q3 - Q1
#
# lower_limit = Q1 - 1.5*IQR
# upper_limit = Q3 + 1.5*IQR
#
# clean_df = df[(df['Price_AZN'] >= lower_limit) & (df['Price_AZN'] <= upper_limit)]
# median_clean = clean_df['Price_AZN'].median()
# print(median_clean)

#
# 20.	Kiçik “mini-profil” hesabatı yaz:-------------------------------------------------
#
# ○	shape, nulls per column, numeric describe, District count, mean/median Price, top-ppm 5 rows.
#
# ○	5 sətirlik nəticə şərhi əlavə et: “Nə gördün? Nələr risk/ fürsət yaradır?
#endregion