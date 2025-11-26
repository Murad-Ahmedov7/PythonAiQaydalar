
#region PythonAi1
from statistics import variance

import numpy
import pandas as pd


# AI-nin (Süni İntellektin) əsas məqsədi insanın intellektual fəaliyyətlərini maşınlara öyrətmək və avtomatlaşdırmaqdır.


#region #AI vs ML vs DL 
 
#Ai

# Tərif:
# İnsanın düşünmə, qərarvermə və problem həll etmə qabiliyyətini kompüterdə təqlid edən sistemlərin ümumi adı.

# Başqa sözlə:
# Ağıllı davranış göstərən hər bir proqram və ya maşın.


# 2. ML — Machine Learning (Maşın Öyrənməsi)
# AI-nin alt sahəsidir

# Tərif:
# Kompüterin açıq şəkildə proqramlaşdırılmadan, datalardan nümunə və qaydalar öyrənərək qərar verməsini təmin edən süni intellekt üsulu.

# Başqa sözlə:
# Model dataya baxıb özü öyrənir və nəticə verir.



# 🔴 3. DL — Deep Learning (Dərin Öyrənmə)

# ML-in alt sahəsidir

# Tərif:
# Çoxqatlı neyron şəbəkələrdən istifadə edərək, böyük və kompleks datalar üzərində avtomatik şəkildə xüsusiyyət çıxaran və öyrənən maşın öyrənməsi metodu.

# Başqa sözlə:
# Beyin kimi çalışan neyron şəbəkələri ilə öyrənmə.

#endregion





# region ⭐ 1. AI (Süni İntellekt) əsas anlayışları

# AI-də 5 əsas qabiliyyət var:

# 1) Perception — Qavrama

# Maşının ətrafı dərk etməsi
# Məs: kamera ilə şəkli görmək, səsi anlamaq

# 2) Reasoning — Məntiq

# Təhlil edib qərar vermək
# Məs: “əgər yağış yağırsa, çətir götürməliyəm”.

# 3) Learning — Öyrənmək

# Datalardan qaydalar çıxarmaq
# Məs: ML modellərinin öyrənməsi

# 4) Planning — Planlaşdırmaq

# Hədəfə çatmaq üçün addımlar seçmək
# Məs: naviqasiya xəritəsi ən qısa yolu tapır

# 5) Action — Əməl etmək

# Robotun hərəkət etməsi, səsli asssistantın cavab verməsi

#endregion


# region ⭐ 2. AI-nin növləri
# 1) Narrow AI (Zəif AI)

# Yalnız bir işi yerinə yetirir.
# Məs:

# Siri

# Google Translate

# ChatGPT

# Face ID
# ➡️ Bir sahədə güclüdür, amma ümumi zəkası yoxdur.

# 2) General AI (Ümumi AI)

# İnsan kimi hər mövzuda düşünə bilən, öyrənə bilən AI.
# Bu hələ mövcud deyil.

# 3) Super AI (Süper Zəka)

# İnsanı bütün sahələrdə keçən AI.
# Bu da mövcud deyil — nəzəri anlayışdır.

#endregion



# region ⭐ 3. AI-nin əsas sahələri

# AI bir neçə böyük sahəyə bölünür:

# Machine Learning (ML)

# Deep Learning (DL)

# Computer Vision (CV) — görüntü işləmə

# NLP — Natural Language Processing

# Robotics

# Expert Systems

# Speech Recognition (səs tanıma)




# 1️⃣ Əsas kitabxanalar

# NumPy (np) → ədədi hesablamalar, massivlər

# Pandas (pd) → DataFrame ilə işləmək

# statistics → sadə statistik funksiyalar (mode, variance)


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



#endregion

