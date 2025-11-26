
#region PythonAi1
from statistics import variance

import numpy
import pandas as pd


# df=pd.DataFrame({
#    'City':['Baku','Ganja','Sumqayit'],
#    'Population':[2300000,330000,340000]
# })

# List: Kiçik və sadə datalarda rahatdır, amma dövrlərlə işləmək lazımdır → yavaş ola bilər.
#
# DataFrame: Böyük dataset və analitik əməliyyatlar üçün optimallaşdırılıb → daha sürətli və daha rahat.

# print(df)
# print(df.head(2))
# print(df.tail(1))
# print(df.sample())


# print(df.info())
# print(df.describe())


# df["Population"]=df["Population"].astype('int64')
# df['Density_guess']=df['Population']/100


# print(df)


# data={
#    "Area_m2":[50,60,80,100,120,200],
#    "Rooms":[1,2,2,3,3,5],
#    "District":["Yasamal","Nizami","Nizami","Sebayil","Nerimanov","Sebayil"],
#    "Price_AZN":[60000,75000,95000,120000,150000,500000]
# }


# houses=pd.DataFrame(data)
# print(houses)
# print(houses[["Area_m2","Price_AZN"]])
# print(houses[houses['Rooms']>=3])


# print(houses.sort_values(by='Price_AZN',ascending=False))
# print(houses['District'].value_counts())


import numpy as np


# print("Mean : ",np.mean(prices))     //ededi orta
# print("Median : ",np.median(prices)) //azalandan arta  siralayib ortada olani goturur.
# from statistics import mode,variance //en cox tekrarlanan (ilk duzgun olani qaytarir)
#
# print("Rooms",mode(houses['Rooms']))

# print("Variance",variance(houses['Price_AZN']))
# print("STD",np.std(houses['Price_AZN']))

# Standard deviation (std) — sadəcə məlumatların orta dəyərdən orta səviyyədə uzaqlığını ölçən göstəricidir.

# prices1=numpy.array([2, 4, 4, 4, 5, 5, 7, 9])
# print("Variance",variance(prices1))


# 1️⃣ Məqsəd
#
# Variance göstərir ki, məlumatlar orta dəyərdən nə qədər uzaqlaşıb.
#
# Kiçik variance → dəyərlər orta ətrafında sıxlaşıb.
#
# Böyük variance → dəyərlər daha çox yayılıb.

# variance=(x-mean)ustu2/n
# x-hər bir dəyər
# n-elementlərin sayı
#mean-ededi orta


# Outlier → bir dataset-dəki digər dəyərlərdən xeyli fərqlənən məlumat nöqtəsidir.
#
# Yəni çox yüksək və ya çox aşağı dəyərlər.
#
# Bu dəyərlər məlumatın ümumi tendensiyasını pozur və analizdə diqqətə alınmalıdır.


# Məlumat: [1, 3, 5, 7, 9]
#
# Q1 (1-ci kvartil, 25%) → məlumatın aşağı 25%-i
#
# Q1 = 3
#
# Q2 (2-ci kvartil, 50%) → median / orta nöqtə
#
# Q2 = 5
#
# Q3 (3-cü kvartil, 75%) → məlumatın yuxarı 25%-i
#
# Q3 = 7

# q1=houses['Price_AZN'].quantile(0.25)
# q3=houses['Price_AZN'].quantile(0.75)
# print(q1)
# print(q3)
# iqr=q3-q1
# lower,upper=q1-1.5*iqr,q3+1.5*iqr
# iqr_outliers=houses[(houses['Price_AZN']<lower) | (houses['Price_AZN']>upper)]
# print(iqr_outliers)



# | Xüsusiyyət    | NumPy Array                | Pandas DataFrame                |
# | ------------- | -------------------------- | ------------------------------- |
# | Tip           | Homojen                    | Heterojen (sütunlar fərqli tip) |
# | Struktura     | Matris / array             | Sətir + sütun (etiketli)        |
# | İndeks        | Sıralı (0,1,2,…)           | İstəyə görə etiketli            |
# | Əsas məqsəd   | Sürətli ədədi əməliyyatlar | Məlumat analizi və təhlili      |
# | Funksionallıq | Vektor əməliyyatları       | Sütun/sətir seçim, filtr, merge |






# Lesson1 QISA QAYDA



# region 📘 1. Əsas kitabxanalar:
# import numpy as np
# import pandas as pd
# from statistics import mode, variance
#
#
# numpy (np) → ədədlərlə və massivlərlə (array) hesablama.
#
# pandas (pd) → cədvəl formalı məlumat (DataFrame) ilə işləmək.
#
# statistics → sadə statistik hesablamalar üçün daxili Python kitabxanası.

#endregion



#  region📗 2. DataFrame və List fərqi:
# Məlumat növü	İstifadə yeri	Üstünlükləri
# List	Kiçik və sadə məlumatlar üçün	Sadə, amma dövrlərlə (for) işləmək yavaşdır
# DataFrame	Böyük datasetlər üçün	Cədvəl tipli, çoxlu funksiyalarla daha sürətli və rahat

#endregion



#📊 #region 3. Pandas əməliyyatları:
# a) DataFrame yaratmaq:
# df = pd.DataFrame({
#    'City': ['Baku', 'Ganja', 'Sumqayit'],
#    'Population': [2300000, 330000, 340000]
# })

# b) Baxış əmrləri:
# print(df)           # bütün cədvəli göstərir
# print(df.head(2))   # ilk 2 sətri göstərir
# print(df.tail(1))   # sonuncu sətri göstərir
# print(df.sample())  # təsadüfi bir sətri göstərir

# c) Ümumi məlumat:
# df.info()       # sütunların tipi, boş dəyərlər və s.
# df.describe()   # statistik xülasə (mean, std, min, max, və s.)



#endregion



#region  📈 4. Statistik göstəricilər:
# a) Mean (Ədədi orta)
# np.mean(prices)
#
#
# 🔹 Bütün dəyərlərin cəmini onların sayına bölür.
# Məs: [2,4,6] → (2+4+6)/3 = 4
#
# b) Median (Ortada olan dəyər)
# np.median(prices)
#
#
# 🔹 Dəyərləri sırala, ortadakı dəyəri götür.
# Məs: [1,3,5,7,9] → 5
#
# c) Mode (Ən çox təkrarlanan dəyər)
# mode(houses['Rooms'])
#
#
# 🔹 Ən çox rast gəlinən dəyəri qaytarır.
# Məs: [1,2,2,3] → 2

# d) Variance (Dispersiya)
# Məlumatların ədədi ortadan nə qədər uzaqlaşdığını göstərir.
# variance=(x-mean)ustu2/n
# x-hər bir dəyər
# n-elementlərin sayı
#mean-ededi orta

# e) Standard Deviation (Standart sapma)
# Variance-in kvadrat kökü.
# Sadəcə “orta dəyərlə real dəyərlər arasında ortalama məsafə” deməkdir.
#
# Məsələn:
#
# Mean = 100
#
# Std = 10 → deməli, əksər dəyərlər [90, 110] aralığında olur.


#endregion



# region 📉 5. Outlier (Sıradan çıxan dəyərlər)
#
# 🔹 Digər dəyərlərdən çox fərqli olan nöqtələrdir (çox yüksək və ya çox aşağı).
#
# Qayda (IQR metodu):
#
# Q1 = df['Price_AZN'].quantile(0.25)
# Q3 = df['Price_AZN'].quantile(0.75)
# IQR = Q3 - Q1
# lower_limit = Q1 - 1.5 * IQR
# upper_limit = Q3 + 1.5 * IQR
#
# outliers = df[(df['Price_AZN'] < lower_limit) | (df['Price_AZN'] > upper_limit)]
#
#
# Beləliklə, bu dəyərlər “qeyri-adi” sayılır.


# import pandas as pd
#Quantile — verilən məlumat dəstini faizlərə (hissələrə) bölən dəyərdir.
# data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# df = pd.DataFrame({'Price_AZN': data})
# 🔹 1️⃣ Q1 = 0.25 quantile
# python
# Copy code
# Q1 = df['Price_AZN'].quantile(0.25)
# print(Q1)
# Nəticə:
#
# Copy code
# 32.5
# 👉 Bu o deməkdir ki, məlumatların 25%-i 32.5-dən kiçikdir,
# yəni təxminən ilk 2–3 ədəd (10, 20, 30) bu hissəyə düşür

#endregion





#
# | Funksiya      | İzah                                              |
# | ------------- | ------------------------------------------------- |
# | `np.mean()`   | Ədədi orta                                        |
# | `np.median()` | Ortadakı dəyər                                    |
# | `mode()`      | Ən çox təkrarlanan dəyər                          |
# | `variance()`  | Yayılma səviyyəsi                                 |
# | `np.std()`    | Orta uzaqlıq (standart sapma)                     |
# | `.describe()` | Əsas statistik göstəriciləri bir baxışda göstərir |
# | `.quantile()` | Quartilləri (Q1, Q2, Q3) hesablamaq üçün          |
#
#
#


# | Nüans                    | NumPy (`np`)                                                       | Pandas (`df`)                                                                                         |
# | ------------------------ | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
# | **Homogen / Eyni tipli** | Bütün elementlər eyni tipdə olmalıdır (hamısı `int` və ya `float`) | Sütunlar **fərqli tiplərdə** ola bilər (`int`, `float`, `str`, `datetime` və s.)                      |
# | **Həqiqi nümunə**        | `[1, 2, 3, 4]` → hamısı `int`                                      | `{"City": ["Baku", "Ganja"], "Population": [2300000, 330000]}` → `City` = `str`, `Population` = `int` |








# //variance() outlier() std() feqi

# Variance	Orta dəyərdən kvadrat fərq	∑(x-mean)²/n	Kvadrat
# Std	Orta dəyərdən ortalama sapma	√Variance	Orijinal
# Outlier	Digər dəyərlərdən çox fərqli	Mean ± k*Std və ya IQR	Orijinal

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