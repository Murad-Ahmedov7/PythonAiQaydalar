
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


#endregion


# region ⭐ 4. Adi proqramlaşdırma və ML fərqi
# Normal proqramlaşdırma

# Kompüterə qaydaları sən yazırsan.

# Rules + Data → Output


# Məs:
# Əgər temperatura > 30 → kondisioneri işə sal

# Machine Learning

# Kompüter qaydaları öz çıxarır.

# Data + Output → Rules/model


# Məs:
# Tələbənin neçə saat oxumasına baxıb “neçə bal alacağını” öyrənir.

#endregion


# region ⭐ 5. ML-in 3 növü
# 🔵 1. Supervised Learning (Nəzarətli öyrənmə)

# Data + Target (doğru cavab) var.

# Məs:

# Studied	Sleep	Marks
# 5	7	80
# 2	5	50

# Model öyrənir: çox oxuyan → çox bal alır.

# 🟢 2. Unsupervised Learning (Nəzarətsiz)

# Yalnız data var, target yoxdur.
# Model qruplaşdırır və ya nümunə tapır.

# Məs:

# 20 yaşlılar çox xərcləyir

# 45 yaşlılar az xərcləyir

# Model özü “klaster” yaradır.

# 🔴 3. Reinforcement Learning (Gücləndirici öyrənmə)

# Agent çevrə ilə qarşılıqlı əlaqədə öyrənir.
# Düz etsə — mükafat
# Səhv etsə — cəza

# Məs:

# Şahmat oynayan AI

# Öz-özünə sürən maşın

#endregion


# region ⭐ 6. Açar anlayışlar
# Feature — giriş parametri

# Məs: yaş, saat, gəlir

# Label — target (doğru cavab)

# Məs: bal, xəstəlik var/yox

# Model

# Öyrədilmiş nümunə.
# Məs: tələbənin balını proqnoz edən model.

# Prediction

# Modelin verdiyi cavab.
# Məs: “70 bal alacaq”.

# Accuracy

# Modelin dəqiqliyi (faizlə)


#endregion


#region ⭐ 7. ANN — Artificial Neural Network

# İnsanın beynindən ilhamlanan model:(insan beyni neyronlarindan)

# neyronlardan ibarətdir

# bağlantılarla məlumat ötürür

# DL-in əsasını təşkil edir

#endregion


#region⭐ 8. DL — Dərin öyrənmə

# Güclü tərəfləri:

# Avtomatik feature extraction (öz-özünə düzgün xüsusiyyətləri tapır)

# Kompleks datanı başa düşür (şəkil, səs, mətn)

# Çox böyük datalarda çox dəqiq nəticə verir

# Zəif tərəfləri:

# Çox güclü hesablama tələb edir (GPU)

# Çox data lazımdır

# “Qara qutu” problem — model niyə belə qərar verdi, aydın olmur

#endregion




# region 1️⃣ Əsas kitabxanalar

# NumPy (np) → ədədi hesablamalar, massivlər

# Pandas (pd) → DataFrame ilə işləmək

# statistics → sadə statistik funksiyalar (mode, variance)

#endregion


# region 2️⃣ List vs DataFrame

# Xüsusiyyət	List	DataFrame
# Tip	Eyni strukturlu sadə data	Cədvəl forması (sətir+sütun)
# İstifadə	Kiçik dataset	Böyük dataset və analiz
# Üstünlük	Sadə	Sürətli, etiketli, çox funksiyalı

#endregion



# region 3️⃣ Pandas – əsas əmrlər

# df.head() → ilk sətirlər

# df.tail() → son sətirlər

# df.sample() → təsadüfi sətir

# df.info() → struktur, tiplər

# df.describe() → statistik xülasə

# Sütun seçimi → df['Col'], df[['A','B']]

# Filtr → df[df['Rooms'] >= 3]


#endregion

#region  4️⃣ Əsas statistik göstəricilər


# Mean → np.mean() → ədədi orta

# Median → np.median() → ortadakı dəyər

# Mode → mode() → ən çox təkrarlanan

# Variance (Dispersiya) – məlumatların orta dəyərdən(mean-den) nə qədər uzaqlaşdığını ölçən statistik göstəricidir.

# Std (Standart sapma) → variance-in kvadrat kökü

# Kiçik std → dəyərlər sıx

# Böyük std → dəyərlər yayılmış


# Kiçik std → məlumat sıx, oxşar

# Böyük std → məlumat yayılmış, fərqli

#endregion



# region 5️⃣ Outlier (Sıradan çıxan dəyərlər)

# Outlier (Sıradan çıxan dəyər) – digər məlumatlardan xeyli fərqlənən, çox yüksək və ya çox aşağı olan dəyərdir.



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


# İzah:
# Digər dəyərlərdən çox yüksək və ya aşağı olan nöqtələr.

#endregion



#region 6️⃣  Pandas əməliyyatları:
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



#region kod numunesi

# df=pd.DataFrame({
#    'City':['Baku','Ganja','Sumqayit'],
#    'Population':[2300000,330000,340000]
# })


# df["Population"]=df["Population"].astype('int64')
# df['Density_guess']=df['Population']/100

#endregion






#endregion


#region PythonAi2

import numpy as np
import pandas as pd

# Matrix əməliyyatları (ikiölçülü array) aparmaq istəyirsən,
#
# Element-wise riyazi əməliyyatları rahat etmək istəyirsən,
#
# Daha sürətli və səliqəli kod yazmaq istəyirsən,
#
# onda NumPy array (np.array) istifadə etmək vacibdir.
#
# Məsələn:


# NumPy istifadə edəndə ikiölçülü array matris kimi işləyir və element-wise əməliyyatlar çox rahat olur:
#
# import numpy as np
#
# matrix = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])
#
# print(matrix * 2)  # Hər elementi 2 ilə vurur



df=pd.DataFrame({
   "City":["Baku","Ganja","Sumqayit"],
   "Population":[2300000,330000,340000],
})


# print(df)
# print(df.head(2))
# print(df.tail(1))
# print(df.sample())


# print(df.info())
# print(df.describe())
# df["Population"]=df["Population"].astype("int64")
# df["Density_guess"]=df["Population"]/100
# print(df)


data={
   "Area_m2":[50,60,80,100,120,200],
   "Rooms":[1,2,2,3,3,5],
   "District":["Yasamal","Nizami","Nizami","Sebayil","Nerimanov","Sebayil"],
   "Price_AZN":[60000,75000,95000,120000,150000,500000]
}


# houses=pd.DataFrame(data)
#print(houses)
# print(houses[['Area_m2','Price_AZN']])
# print(houses[houses['Rooms']>=3])
# print(houses.sort_values(by="Price_AZN",ascending=False))
# print(houses["District"].value_counts())


# prices=np.array([60000,75000,95000,120000,150000,500000])
# print("Mean : ",np.mean(prices)) #ortalama tapir
# print("Median : ",np.median(prices)) # azdan choxa siralayir ve ortadaki element tapir
#
# from statistics import mode,variance
#
# print("Mode : ",mode(houses["Rooms"])) # en chox olan elementi tapir
# print("Variance : ",variance(houses["Price_AZN"]))
# print("STD : ",np.std(houses["Price_AZN"]))




a=np.random.randint(1,10,size=[3,4])
# print(a)
print("----------")
# print(a.shape) matrix-in olcusune gosterir.
# print(a.T) matrixi tersine cevirir.
# print(a[0,1])
# print(a[:,2])
# print(a[1:3,1:3])


# b=np.array([1,2,3,4])
# print(b+5)
# print(b*5)
# print(b**2)


# normal=np.random.normal(0,5,20)
# uniform=np.random.uniform(0,10,20)
# print(normal)
# print(uniform)


# 1️⃣ np.random.normal(0,5,20)
#
# np.random.normal(mean, std, size) funksiyası normal paylanmış (Gauss dağılımı) təsadüfi ədədlər yaradır.
#
# Parametrlər:
#
# 0 → orta qiymət (mean)
#
# 5 → standart sapma (standard deviation)
#
# 20 → neçə ədəd yaratmaq istədiyin (size)
#
# Nəticə: 20 ədəd təsadüfi ədədlər, əsasən 0 ətrafında, çox vaxt -15 ilə 15 arasında olacaq (çox uzaq dəyərlər nadirdir).
#
# Məsələn:
#
# [ 3.1, -2.7, 0.5, 7.8, ... ]
#
# 2️⃣ np.random.uniform(0,10,20)
#
# np.random.uniform(low, high, size) funksiyası bərabər paylanmış (uniform) təsadüfi ədədlər yaradır.
#
# Parametrlər:
#
# 0 → minimum dəyər
#
# 10 → maksimum dəyər
#
# 20 → neçə ədəd yaratmaq istədiyin
#
# Nəticə: 20 ədəd təsadüfi ədəd, hər birinin 0 ilə 10 arasında ehtimalı bərabərdir.
#
# Məsələn:
#
# [1.2, 9.8, 4.5, 0.3, ... ]
# #
# | Xüsusiyyət     | Normal (Gaussian)                      | Uniform (Bərabər)               |
# | -------------- | -------------------------------------- | ------------------------------- |
# | Forma          | Zəng (bell-shaped)                     | Düzbucaqlı (flat)               |
# | Orta dəyər     | Ədədlərin çoxu ortada                  | Ehtimal hər yerdə eynidir       |
# | Kənar dəyərlər | Nadirdir                               | Eyni ehtimalla ola bilər        |
# | İstifadəsi     | Statistik analiz, real həyat modelləri | Sadə təsadüfi seçim, simulasiya |
# | Misal          | Nümunələr: boy, ağırlıq, səhvlər       | Nümunələr: rulet, random seçim  |

import pandas as pd
# s=pd.Series([5,10,15,20],index=['A','B','C','D'])
# print(s)
# print(s.mean(),s.median())


# houses=pd.read_excel("houses_day1.xlsx")
# print(houses)
# print(houses.head(5))
# print(houses.shape)
# print(houses.columns)


# houses["Price_per_m2"]=houses["Price_AZN"].astype(float)/houses["Area_m2"]
# houses.to_excel("houses_day1.xlsx",index=False)


# houses['Price_AZN'].fillna(houses['Price_AZN'].median(),inplace=True) #fill not available
# houses.to_excel("houses_day1.xlsx",index=False)


# print("Mean : ",houses['Price_AZN'].mean())
# print("Median : ",houses['Price_AZN'].median())
# print("Mode : ",houses["Rooms"].mode()[0])


# print(houses[["Area_m2","Price_AZN"]].cov())

# Bu iki sətir Python-da pandas kitabxanası ilə DataFrame-də statistik əlaqəni öyrənmək üçün istifadə olunur. Gəlin addım-addım izah edim:
#
# 1️⃣ cov() — Kovariasiya
# houses[["Area_m2","Price_AZN"]].cov()
#
#
# cov() funksiyası iki dəyişənin kovariasiyasını hesablayır.
#
# Kovariasiya göstərir ki, iki dəyişən birlikdə necə hərəkət edir:
#
# Müsbət dəyər → bir dəyişən artanda digəri də artır.
#
# Mənfi dəyər → bir dəyişən artanda digəri azalır.
#
# Dəyərin ölçüsü dəyişənlərin ölçülərinə bağlıdır, yəni müqayisə etmək çətindir.
#
# Nümunə çıxış:
#
# 	Area_m2	Price_AZN
# Area_m2	25.0	1200.0
# Price_AZN	1200.0	80000.0
#
# Burada 1200.0 Area və Price arasındakı kovariasiyadır.


# | Xüsusiyyət        | `cov()` (Covariance)                                       | `corr()` (Correlation)                                                 |
# | ----------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------- |
# | Nə ölçür          | İki dəyişənin birlikdə necə dəyişdiyini                    | İki dəyişənin xətti əlaqəsinin gücünü və istiqamətini                  |
# | Dəyər aralığı     | -∞ … +∞ (heç bir standart ölçü yoxdur)                     | -1 … +1 (standart ölçüdə)                                              |
# | Müsbət/ mənfi     | Müsbət → birlikdə artır, Mənfi → bir artanda digəri azalır | Müsbət → birlikdə artır, Mənfi → bir artanda digəri azalır             |
# | Ölçülərə bağlılıq | Bəli, dəyişənlərin vahidinə bağlıdır                       | Xeyr, vahiddən asılı deyil                                             |
# | İzahat            | Sadəcə birlikdə necə hərəkət etdiklərini göstərir          | Hərəkətin gücünü və istiqamətini göstərir, müqayisə etmək daha asandır |




# print(houses[["Area_m2","Price_AZN"]].corr())




# print(houses[["Area_m2","Price_AZN","Rooms"]].corr())


# # import matplotlib.pyplot as plt
# # plt.hist(houses['Price_AZN'],bins=20)
# # plt.title("Price Distribution")
# # plt.xlabel("Price")
# # plt.ylabel("Count")
# # plt.show()
# #
# #
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # sns.heatmap(houses.corr(numeric_only=True),annot=True,cmap="coolwarm")
# # plt.title("Correlation Heatmap")
# # plt.show()
# #
# #
# #
# #
# # by_district=houses.groupby("District")['Price_AZN'].mean().sort_values(ascending=False)
# # print(by_district)
# #
# #
# # q1=houses['Price_AZN'].quantile(0.25)
# # q3=houses['Price_AZN'].quantile(0.75)
# #
# # iqr=q3-q1
# # lower,upper=q1-1.5*iqr,q3+1.5*iqr
# # iqr_outliers=houses[(houses['Price_AZN']<lower) | (houses['Price_AZN']>upper)]
# # print(iqr_outliers[['District','Area_m2','Price_AZN']])
# #
# #
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# #
# #
# # sns.lmplot(data=houses,x="Area_m2",y="Price_AZN",line_kws={"color":"red"})
# # plt.title("Area and Price trend line")
# # plt.show()
#
#
#
#
#
# 1️⃣ Qiymətlərin paylanması (Histogram)
# import matplotlib.pyplot as plt
# plt.hist(houses['Price_AZN'], bins=20)
# plt.title("Price Distribution")
# plt.xlabel("Price")
# plt.ylabel("Count")
# plt.show()
#
#
# plt.hist() → verilən sütunun paylanmasını (histogram) göstərir.
#
# houses['Price_AZN'] → qiymət sütunu (AZN ilə).
#
# bins=20 → qiymətləri 20 intervala bölür.
#
# plt.title, plt.xlabel, plt.ylabel → qrafikə başlıq və ox adları əlavə edir.
#
# plt.show() → qrafiki ekranda göstərir.
#
# Nəticə: Qiymətlərin neçə dəfə təkrarlanmasını vizual görəcəksən.
#
# 2️⃣ Korrelyasiya istilik xəritəsi (Heatmap)
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.heatmap(houses.corr(numeric_only=True), annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()
#
#
# houses.corr(numeric_only=True) → bütün ədədi sütunlar arasındakı korrelyasiyanı hesablayır.
#
# sns.heatmap(..., annot=True) → nəticələri rəngli cədvəl (heatmap) şəklində göstərir və rəqəmləri annotasiya edir.
#
# cmap="coolwarm" → rəng sxemi (mavi-qırmızı) istifadə olunur.
#
# Nəticə: Hansı dəyişənlərin bir-biri ilə güclü əlaqəsi olduğunu vizual olaraq görəcəksən.
#
# 3️⃣ Rayon üzrə orta qiymətlər
# by_district = houses.groupby("District")['Price_AZN'].mean().sort_values(ascending=False)
# print(by_district)
#
#
# groupby("District") → məlumatları rayonlara görə qruplaşdırır.
#
# ['Price_AZN'].mean() → hər rayondakı orta qiyməti hesablayır.
#
# sort_values(ascending=False) → nəticəni böyükdən kiçiyə sıralayır.
#
# Nəticə: Hansı rayonlarda evlərin daha bahalı olduğunu görə bilərsən.
#
# 4️⃣ Outlier-lərin tapılması (IQR metodu)
# q1 = houses['Price_AZN'].quantile(0.25)
# q3 = houses['Price_AZN'].quantile(0.75)
# iqr = q3 - q1
# lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
# iqr_outliers = houses[(houses['Price_AZN'] < lower) | (houses['Price_AZN'] > upper)]
# print(iqr_outliers[['District','Area_m2','Price_AZN']])
#
#
# quantile(0.25) və quantile(0.75) → qiymətlərin 1-ci və 3-cü kvartillərini tapır.
#
# iqr = q3 - q1 → kvartillər arasındakı fərq (Interquartile Range).
#
# lower, upper → normal qiymətlərin aşağı və yuxarı sərhədi.
#
# iqr_outliers → sərhəd xaricində olan outlier qiymətləri seçir.
#
# Nəticə: Hansı evlərin qiymətinin digərlərindən kənarda olduğunu görə bilərsən.
#
# 5️⃣ Area və Price əlaqəsi (Trend line ilə scatter plot)
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# sns.lmplot(data=houses, x="Area_m2", y="Price_AZN", line_kws={"color":"red"})
# plt.title("Area and Price trend line")
# plt.show()
#
#
# sns.lmplot() → scatter plot + xətti trend (linear regression) göstərir.
#
# x="Area_m2", y="Price_AZN" → horizontal oxda sahə, vertikal oxda qiymət.
#
# line_kws={"color":"red"} → trend xəttinin rəngini qırmızı edir.
#
# Nəticə: Ev sahəsi ilə qiymət arasındakı əlaqəni vizual və trend xətti ilə görəcəksən.



#Lesson 2 Qaydalariin qisa sekilde yaz.


students = pd.read_excel('students_performance.xlsx')

# 1️⃣ Sürətli baxış:
# head(5), tail(5) və sample(3) ilə datasetə bax.
#  Sual: Tələbələrin hansı qeydləri maraqlı görünür (çox yüksək / çox aşağı GPA və ya fərqli department)?

# print(students.head(5))
# print(students.tail(5))
# print(students.sample((3)))

#
# 2️⃣ Struktur yoxlaması:
# info() nəticəsinə bax və hansı sütunlarda boş dəyər varsa qeyd et.
#  Hər sütunun dtype-ını yaz.
#  Sual: hansı sütunlar ədədi, hansılar kateqorikdir

# print(students.info())


#
# 3️⃣ Statistik icmal:
# describe() nəticəsinə bax, GPA və MathScore üçün mean, median, std müqayisə et.
#  Yekun: GPA paylanması simmetrikdirmi?

# print(students.describe())
#
# 4️⃣ Tip düzəlişi:
# Əgər HasScholarship və AttendanceRate sütunları “object” kimi oxunubsa,
#  onları uyğun tiplərə çevir (bool, float).
#  Sual: Niyə tip uyğunluğu statistik analizdə vacibdir?

# students['HasScholarship'] = students['HasScholarship'].astype(bool)
# students['AttendanceRate']=students['AttendanceRate']
#
# print(students['HasScholarship'])


# 5️⃣ Boş dəyərləri analiz et:
# isnull().sum() ilə yoxla.
#  Boş dəyərlər varsa — GPA üçün median, Department üçün ən çox rast gəlinəni ilə doldur.
#  Sual: niyə median mean-dən daha sabit seçimdir?

# print("Boş dəyərlərin sayı:")
# print(students.isnull().sum())
#
# students['GPA'] = students['GPA'].fillna(students['GPA'].median())
#
# students['Department'] = students['Department'].fillna(students['Department'].mode()[0])
#
# print("\nDoldurulduqdan sonra boş dəyərlərin sayı:")
# print(students.isnull().sum())


#
# 6️⃣ Departament üzrə tələbə sayı:
# Department.value_counts() çıxar.
#  Sual: hansı fakültə daha çox tələbəyə malikdir və bu balanssızlıq nəyə səbəb ola bilər?

# print("\nDepartament üzrə tələbə sayı:")
# print(students['Department'].value_counts())
#
# 7️⃣ Mean vs Median (MathScore):
# MathScore üçün mean və median müqayisə et.
#  Sual: fərq varsa, outlier təsirindən qaynaqlanırmı?
# mean_math = students['MathScore'].mean()
# median_math = students['MathScore'].median()
#
# print(f"\nMathScore mean: {mean_math:.2f}")
# print(f"MathScore median: {median_math:.2f}")
# #
# 8️⃣ Scholarship təsiri:
# Təqaüd alan və almayan tələbələrin orta GPA-sını müqayisə et.
#  Sual: Təqaüdün tədris nəticəsinə real təsiri varmı?
# gpa_scholar = students.groupby('HasScholarship')['GPA'].mean()
# print("\nTəqaüdə görə GPA ortalaması:")
# print(gpa_scholar)

#
# 9️⃣ Korelyasiya:
# GPA, MathScore, ReadingScore, WritingScore, AttendanceRate arasında korelyasiya matrisini çıxar.
#  Sual: hansılar bir-biri ilə daha güclü əlaqədədir?

corr = students[['GPA','MathScore','ReadingScore','WritingScore','AttendanceRate']].corr()
print("\nKorelyasiya matris:")
print(corr)

#
# 🔟 Outlier (IQR metodu):
# MathScore üçün Q1, Q3, IQR hesabla.
#  Ən aşağı və ən yüksək outlier-ləri tap.
#  Sual: bu dəyərlər hansı tələbələrdədir və niyə fərqlənirlər?


# Q1 = students['MathScore'].quantile(0.25)
# Q3 = students['MathScore'].quantile(0.75)
# IQR = Q3 - Q1
#
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
#
# outliers_iqr = students[(students['MathScore'] < lower_bound) | (students['MathScore'] > upper_bound)]
# print("\nIQR outlier-lər:")
# print(outliers_iqr[['StudentID', 'MathScore']])
#
# 1️⃣1️⃣ Outlier (Z-score metodu):
# MathScore və GPA üçün z-score hesabla.
#  |z| > 3 dəyərləri tap.
#  Sual: IQR və Z-score fərqli nəticə verir? Niyə?


# from scipy import stats
#
# students['z_math'] = stats.zscore(students['MathScore'])
# students['z_gpa'] = stats.zscore(students['GPA'])
#
# outliers_z = students[(abs(students['z_math']) > 3) | (abs(students['z_gpa']) > 3)]
# print("\nZ-score outlier-lər:")
# print(outliers_z[['StudentID','GPA','MathScore','z_math','z_gpa']])



#
# 1️⃣2️⃣ Departament üzrə GPA müqayisəsi:
# groupby("Department")["GPA"].agg(["mean","median","count"]) çıxar.
#  Sual: Hər fakültənin GPA səviyyəsi bir-birindən fərqlidirmi?
# dept_gpa = students.groupby('Department')['GPA'].agg(['mean','median','count'])
# print("\nDepartament üzrə GPA müqayisəsi:")
# print(dept_gpa)

# 1️⃣3️⃣ Gender fərqləri:
# Gender üzrə GPA və MathScore medianlarını müqayisə et.
#  Sual: fərq böyükdürmü? Əgər varsa, səbəb nə ola bilər?

# gender_stats = students.groupby('Gender')[['GPA','MathScore']].median()
# print("\nGender üzrə GPA və MathScore medianları:")
# print(gender_stats)
#
# 1️⃣4️⃣ Vizual analiz:--------------------------------------------
# Histogram: GPA və MathScore üçün paylanma
#
#
# Boxplot: fakültələr üzrə GPA
#
#
# Scatterplot: AttendanceRate vs GPA (rəngləndirmə HasScholarship-a görə)
#  Sual: hansı vizualizasiyadan ən çox nəticə çıxarmaq olur?
#
#
#
# 1️⃣5️⃣ Mini nəticə hesabatı:
# Aşağıdakıları yaz:
# Dataset ölçüsü və struktur xülasəsi
#
#
# Əsas statistik müşahidələr (mean, median, std)
#
#
# Outlier-lər və onların təsiri
#
#
# Korelyasiya və paylanma nəticəsi
#
#
# “Nəticə”: hansı amillər tələbənin performansını ən çox təsir edir?
print("\n--- Mini Hesabat ---")

# Dataset ölçüsü və struktur
# print(f"Dataset ölçüsü: {students.shape}")
# print("\nÜmumi təsvir:")
# print(students.describe())
#
# # Əsas müşahidələr
# print("\nƏsas statistik müşahidələr:")
# print("Mean GPA:", students['GPA'].mean())
# print("Median GPA:", students['GPA'].median())
#
# # Korelyasiya nəticələri
# print("\nKorelyasiya:")
#
# corr = students[['GPA','MathScore','ReadingScore','WritingScore','AttendanceRate']].corr()
#
# # Sonra çap et
# print(corr)

#endregion

