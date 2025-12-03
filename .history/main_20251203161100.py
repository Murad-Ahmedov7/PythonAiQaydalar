


# 1) HER BIRINE BIR NUMUNE YAZ NEZERI OLARAQ 
# 2) HER DERSIN SUAL-CAVABINA BAX.
# 3) 9-CU DERSIN QUIZINE YENIDEN BAX. 



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


# region ⭐ 5. ML-in 3 növü(oyrenme novleri)
# 🔵 1. Supervised Learning (Nəzarətli öyrənmə)

#feature-giris melumatlari

#target-cixis melumatlari

# Data + Target (doğru cavab) var.

# Məs:

# Studied	   Sleep	  Marks
# 5	          7	     80
# 2	          5	     50

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


#  region 1️⃣ NumPy

# İstifadə: Matris əməliyyatları, element-wise riyazi hesablamalar, sürətli kod

# import numpy as np
# matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(matrix*2)  # hər elementi 2 ilə vurur


# Random:

# np.random.normal(mean, std, size)    # Gaussian paylanma
# np.random.uniform(low, high, size)   # Uniform paylanma



# 1️⃣ Normal (Gaussian) paylanma

# Tərif: Dəyərlərin çoxu orta ətrafında cəmləşən, kənarlarda isə nadir hallarda olan paylanmadır.

# Forma: Bell-shaped (zəng şəkilli)

# NumPy: np.random.normal(mean, std, size)

# İstifadə: Statistik analiz, real həyat modelləri, ölçümlərin və səhvlərin paylanması

# 2️⃣ Uniform (Bərabər) paylanma

# Tərif: Verilən aralıqdakı bütün ədədlərin eyni ehtimalla meydana gəldiyi paylanmadır.

# Forma: Flat (düzbucaqlı)

# NumPy: np.random.uniform(low, high, size)

# İstifadə: Təsadüfi seçim, simulasiya, oyunlar

#endregion



# region 2️⃣ Korrelyasiya & Kovariasiya



# 1️⃣ Covariance (cov()) – Kovariasiya

# Tərif: İki dəyişənin birlikdə necə dəyişdiyini göstərən statistik ölçüdür.

# Müsbət dəyər: Bir dəyişən artanda digəri də artır.

# Mənfi dəyər: Bir dəyişən artanda digəri azalır.

# Qeyd: Ölçülərə bağlıdır, müqayisə etmək çətindir.

# Numunə:

# import pandas as pd

# data = {'X': [1, 2, 3], 'Y': [2, 4, 6]}
# df = pd.DataFrame(data)

# print(df.cov())


# Nəticə:

#       X    Y
# X   1.0  2.0
# Y   2.0  4.0


# X və Y → müsbət kovariasiya → birlikdə artır.

# 2️⃣ Correlation (corr()) – Korrelyasiya

# Tərif: İki dəyişənin xətti əlaqəsinin gücünü və istiqamətini ölçən statistik göstəricidir.

# Dəyər aralığı: -1 … +1

# Müsbət dəyər: Bir dəyişən artanda digəri də artır.

# Mənfi dəyər: Bir dəyişən artanda digəri azalır.

# 0: Heç bir xətti əlaqə yoxdur.

# Qeyd: Ölçülərdən asılı deyil, müqayisə etmək rahatdır.

# Numunə:

# print(df.corr())


# Nəticə:

#      X    Y
# X  1.0  1.0
# Y  1.0  1.0


# X və Y → +1 → mükəmməl müsbət xətti əlaqə

# 💡 Qısa fərq:

# cov() → birlikdə necə dəyişir (ölçülərə bağlı)

# corr() → əlaqənin gücü və istiqaməti (-1 … +1, ölçüdən asılı deyil)


#endregion



# region 3️⃣Vizualizasiya
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Histogram
# plt.hist(df['Price'], bins=20); plt.show()

# # Heatmap (Correlation)
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm'); plt.show()

# # Scatter + Trend line
# sns.lmplot(data=df, x="Area", y="Price", line_kws={"color":"red"}); plt.show()


# Histogram: qiymət paylanması

# Heatmap: dəyişənlərin qarşılıqlı əlaqəsi

# Scatter + Trend line: iki dəyişən arasındakı əlaqə

#endregion



#endregion


#region PythonAi3

#region 📘 Data Science - Məqsəd və Mərhələlər

# Data Science (Məlumat Elmi) — müxtəlif mənbələrdən toplanan böyük həcmli məlumatları 
# təhlil edib, onları təmizləyib, modelləşdirərək nümunələr, tendensiyalar və faydalı nəticələr çıxarmağa yönəlmiş elmi və praktik sahədir.
# Məqsəd, məlumatlara əsaslanaraq qərar vermək, proqnozlaşdırmaq və problemləri həll etməkdir.

# Mərhələləri qısa olaraq:
# 1️⃣ Toplama (Collect) – məlumatları yığmaq.
# 2️⃣ Təmizləmə və hazırlama (Clean & Preprocess) – məlumatı işlək və düzgün hala gətirmək.
# 3️⃣ Modelləşdirmə (Model / Analyze) – analiz və proqnoz üçün modellər qurmaq.
# 4️⃣ Test etmək və qiymətləndirmək (Evaluate) – modellərin düzgünlüyünü yoxlamaq.
# 5️⃣ İstifadəyə vermək (Deploy / Operationalize) – real vəziyyətdə tətbiq etmək, qərar dəstəyi üçün istifadə etmək.

#endregion




#region 📗 Profiling - Avtomatik Dataset Analizi

# import pandas as pd
# import numpy as np
# from ydata_profiling import ProfileReport

# # Excel faylını oxu
# houses = pd.read_excel("houses_day.xlsx")

# # Boş dəyərlərin yoxlanması
# print("Floor null count:", houses['Floor'].isnull().sum())

# # Median ilə doldurmaq
# houses['DistanceToMetro_km'].fillna(houses['DistanceToMetro_km'].median(), inplace=True)

# # Sətirlərdə boş District olanları sil
# houses.dropna(subset=['District'], inplace=True)

# # Floor sütununu rəqəmə çevir
# houses['Floor'] = pd.to_numeric(houses['Floor'], errors='coerce')

# # ? işarələrini NaN ilə əvəzlə
# houses.replace("?", np.nan, inplace=True)

# # Dublikatları sil
# houses.drop_duplicates(inplace=True)

# # Rayon üzrə orta qiymət hesabla
# avg_by_district = houses.groupby('District')['Price_AZN'].mean().round(2)

# # Hər evin qiymətinin orta qiymətə nisbət fərqini faizlə əlavə et
# houses['price_vs_mean'] = ((houses['Price_AZN'] / houses['District'].map(avg_by_district)) - 1) * 100

# # Profiling hesabatı
# report = ProfileReport(houses, title="Houses Day Report", explorative=True)
# report.to_file("houses_day.html")

# # 🔹 Profiling (məlumat profilləşdirmə) – datasetin avtomatik analiz edilməsi və 
# # xülasə hesabatının hazırlanması prosesidir.


#endregion




#endregion


#region PythonAi4 Notbukdadi

#endregion

#region PythonAi5 Notbukdadi

#endregion


#region PythonAi6


# # Distribution (Paylanma) — verilənlərin və ya ehtimalların hansı dəyərlər arasında və hansı tezliklə yayıldığını göstərən statistik anlayışdır.



# # | Paylanma növü    | Şəkil           | Əsas xüsusiyyət              |
# # | ---------------- | --------------- | ---------------------------- |
# # | **Normal**       | 🔔 Zəng formalı | Simmetrik, orta dəyərlər çox |
# # | **Uniform**      | ▭ Düz           | Hər dəyər bərabər ehtimallı  |
# # | **Poisson**      | 📉 Sağ əyilmiş  | Nadir hadisələrin sayı       |
# # | **Right-skewed** | ↘ Sağ quyruqlu  | Çox kiçik, az böyük dəyərlər |
# # | **Left-skewed**  | ↙ Sol quyruqlu  | Çox böyük, az kiçik dəyərlər |



# # 📊 1. Normal Distribution (Normal paylanma)

# # 📈 Şəkli: Zəng (bell) formasında, simmetrik.

# # Ortada ən çox dəyərlər var.

# # Uclarda (az və çox) az dəyərlər olur.

# # ✨ Xüsusiyyətlər:

# # Mean = Median = Mode

# # Data “orta” ətrafında yığılır.

# # Statistikada və Machine Learning-də ən çox istifadə olunan paylanmadır.



# # 📊 2. Uniform Distribution (Bərabər paylanma)

# # 📈 Şəkli: Düz xətt — bütün dəyərlərin eyni ehtimalı var.

# # ✨ Xüsusiyyətlər:

# # Hər nəticə eyni şansla baş verir.

# # “Tam ədalətli” təsadüf hadisəsi.


# # 📊 3. Poisson Distribution (Puasson paylanması)

# # 📈 Şəkli: Sağ tərəfə əyilmiş (right-skewed).

# # Nadir, amma baş verə bilən hadisələrin paylanması üçün istifadə olunur.

# # Diskret (tam ədədlərlə işləyir).

# # ✨ Xüsusiyyətlər:

# # “Hadisələrin sayı”na baxır (vaxt və ya məkan daxilində).

# # Nəticələr 0, 1, 2, 3 kimi olur (say).


# # 📊 4. Right-Skewed Distribution (Sağa əyilmiş paylanma)

# # 📈 Şəkli: Qrafikin quyruğu sağ tərəfə uzanır.
# # Yəni çox dəyərlər kiçik, amma bir neçə böyük dəyər var.

# # ✨ Xüsusiyyətlər:

# # Mean > Median > Mode

# # Outlier-lar (böyük dəyərlər) sağdadır.



# # 📊 5. Left-Skewed Distribution (Sola əyilmiş paylanma)

# # 📈 Şəkli: Quyruq sol tərəfə uzanır.
# # Yəni çox dəyərlər böyük, amma bir neçə kiçik dəyər var.

# # ✨ Xüsusiyyətlər:

# # Mean < Median < Mode

# # Outlier-lar (kiçik dəyərlər) soldadır.


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd


# # house_room_count=np.random.normal(loc=4,scale=1,size=1000)

# # # print(house_room_count)

# # plt.hist(house_room_count,bins=20,color='skyblue',edgecolor='black')

# # plt.title('Normal distribution typical house rooms')
# # plt.xlabel('Room Count')
# # plt.ylabel('House Count')
# # plt.show()



# # house_room_count=np.random.uniform(low=1, high=6, size=1000)

# # # print(house_room_count)
# # #
# # plt.hist(house_room_count,bins=10,color='skyblue',edgecolor='black')
# # plt.xlabel('Room Count')
# # plt.ylabel('House Count')
# # plt.show()

# # # bins=20
# # # Histogramın sütun sayını göstərir (20 sütun)




# house_room_count=np.random.poisson(2,size=1000)

# print(house_room_count)

# plt.hist(house_room_count,bins=range(0,10),color='salmon',edgecolor='black')
# plt.title("Poisson distribution")
# plt.xlabel("House Count")
# plt.ylabel("Room Count")
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.stats import normaltest, stats, poisson, chisquare

# # ------------------------------
# #        POISSON P AYLANMA
# # ------------------------------

# house_room_count = np.random.poisson(2, size=1000)

# plt.hist(house_room_count, bins=range(0, 10), edgecolor='black')
# plt.title("Poisson distribution")
# plt.xlabel("House Count")
# plt.ylabel("Room Count")
# plt.show()

# # λ — Poisson paylanmasının orta dəyəri (mean)

# # ------------------------------
# #        NORMAL TEST NÜMUNƏSİ
# # ------------------------------

# normal_data = np.random.normal(loc=4, scale=1, size=1000)

# # stat, p_value = normaltest(normal_data)
# # print(stat, p_value)
# # if p_value > 0.05:
# #     print("Normal Distribution")
# # else:
# #     print("Not Normal Distribution")

# # ------------------------------
# #        POISSON TEST NÜMUNƏSİ
# # ------------------------------

# poisson_data = np.random.poisson(2, size=1000)

# # observed_count = np.bincount(poisson_data)
# # expected_count = [poisson.pmf(i, 2) * len(poisson_data) for i in range(len(observed_count))]
# # expected_count[-1] += len(poisson_data) - sum(expected_count)

# # chi_stat, p_value = chisquare(observed_count, expected_count)
# # print(chi_stat, p_value)

# # ------------------------------
# #        SKEWNESS
# # ------------------------------


# # 🎯 Skew (Skewness) nədir?

# # Skewness — paylanmanın simmetrik olub-olmamasını göstərən statistik ölçüdür.

# # Sadə dildə:

# # Paylanma sağa əyilirsə → çox kiçik dəyərlər var, az böyük → skew > 0

# # Paylanma sola əyilirsə → çox böyük dəyərlər var, az kiçik → skew < 0

# # Paylanma tam simmetrikdirsə → skew ≈ 0

# print("Skewness with Normal data:", stats.skew(normal_data))
# print("Skewness with Poisson data:", stats.skew(poisson_data))
# print("Skewness with Excel houses data:", stats.skew(house_room_count))




# # print("Skewness with Possion data")
# # print(stats.skew(possion_data))




# # print("Skewness with Excel houses data")
# # print(stats.skew(house_room_count))

#endregion


#region PythonAi7

# # 1️⃣ Regression nə üçün istifadə olunur?

# # Məqsəd: Bir və ya bir neçə müstəqil dəyişən (X) əsasında bir asılı dəyişən (y) proqnoz etmək.

# # Misal:

# # Ev ölçüsü və otaq sayı → evin qiyməti

# # Reklam xərcləri → satış sayı

# # Temperatur və rütubət → enerji sərfiyyatı

# # 2️⃣ Regression növləri

# # Linear Regression (Xətti Regression): y = a*X + b
# # Ən sadə formadır, nəticə müstəqil dəyişənlərlə xətti əlaqədədir.

# # Polynomial Regression (Polinomial Regression): y = a*X^2 + b*X + c
# # X və y arasında xətti olmayan əlaqələr üçün.

# # Multiple Regression (Çoxlu Regression): Bir neçə X istifadə olunur: y = a1*X1 + a2*X2 + ... + b

# # Digər növlər: Ridge, Lasso, Decision Tree Regression, Random Forest Regression və s.


# # X → müstəqil dəyişən (input, predictor)

# # y → asılı dəyişən (output, target)

# # a → meyl (slope) – X dəyişdikcə y nə qədər dəyişir

# # b → intercept (kəsilmə nöqtəsi) – X=0 olanda y-nin qiyməti



# #MAE- Mean Absolute Error-ortalama sehv
# #y=[100,110,120]
# #y^=[103,113,124]
# #MAE=(|103-100|+|113-110|+|124-100|)/3=3.3


# # MSE — Mean Squared Error, yəni Ortalama Kvadrat Səhv demək

# #y=[100,110,120]
# #y^=[103,113,124]
# #MSE=(|103-100|^2+|113-110|^2+|124-100|^2)/3=3.3/34......

# #R2=1-(3.3/34)=90%  R² Score Modelin nə qədər düzgün proqnoz etdiyini göstərir (0–1 arası)

# # Regression Metrics




# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd





# # “plt” — Python-da matplotlib.pyplot modulunun qısaldılmış adıdır.

# # Sən kodda bunu görmüsən:

# # import matplotlib.pyplot as plt


# # Bu sətr matplotlib.pyplot kitabxanasını plt adı ilə çağırmağa imkan verir.

# # 🎯 plt nə üçündür?

# # plt istifadə olunur:

# # qrafik çəkmək

# # histogram yaratmaq

# # scatter plot çəkmək

# # x/y oxlarını yazmaq

# # başlıq əlavə etmək

# # qrafiki göstərmək

# # yəni bütün vizualizasiya (qrafik) əməliyyatlarında.



# from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# # mae=mean_absolute_error(price,pred)
# # mse=mean_squared_error(price,pred)

# # print(mae)
# # print(mse)
# # r2=r2_score(price,pred)



# # while True:
# #     area=int(input("Enter the area of interest: "))
# #     Pred=a*area+b
# #     print(Pred)




# # Əlbəttə! Gəlin kovariyant (covariance) anlayışını tam sadə şəkildə izah edək.

# # 1️⃣ Kovariyant nədir?

# # Kovariyant iki dəyişənin birlikdə necə dəyişdiyini göstərən ölçüdür.

# # İki dəyişən eyni istiqamətdə dəyişirsə → kovariyant müsbət olur.

# # İki dəyişən əks istiqamətdə dəyişirsə → kovariyant mənfi olur.

# # Heç bir əlaqəsi yoxdursa → kovariyant 0-a yaxın olur.


# # 2) Dispersiya(variance) nədir?

# # Dispersiya bir dəyişənin orta dəyərdən nə qədər uzaqlaşdığını ölçür.
# # Sadə desək, bir sıra dəyərlərin nə qədər “yayılmış” olduğunu göstərir.


# 🔹 Dispersiya (Variance) nədir?
# Dəyərlərin orta qiymətdən neçə vahid² uzaqlaşdığını ölçən göstəricidir.
# Yəni orta kvadratik kənarlaşmadır.


# Std (Standard Deviation) nədir?

# Dispersiyanın kvadrat kökü deməkdir.


# #Gradient Descent


# # Gradient Descent — Machine Learning modelinin “daha yaxşı cavab” tapmaq üçün təkrarlayaraq özünü düzəltmə metodudur.

# # Model səhv edir → səhvin ölçüsü hesablanır → model səhvi azaltmaq üçün kiçik addım dəyişiklik edir → yenidən yoxlanır.


# # Əla, sən sadə bir Linear Regression (Xətti reqressiya) modelini gradient descent ilə sıfırdan yazmısan

# # X=area
# # Y=price


# # m=1300
# # b=80000
# # Y_pred=m*X+b
# # print(Y_pred)
# # L=0.0001    #learning rate
# # epochs=40000


# # n=len(X)
# # for i in range(epochs):
# #    Y_pred=m*X+b
# #    D_m=(-2/n)*sum(X*(Y-Y_pred))
# #    D_b=(-2/n)*sum(Y-Y_pred)
# #    m=m-L*D_m
# #    b=b-L*D_b


# # print(m)
# # print(b)


# # mae=mean_absolute_error(Y,Y_pred)
# # mse=mean_squared_error(Y,Y_pred)
# # print("================")
# # print(mae)
# # print(mse)
# # r2=r2_score(Y,Y_pred)
# # print(r2)





# # Verilənlər (X = sahə, Y = qiymət)
# area = np.array([50, 55, 60, 65, 70, 80, 90, 100, 120])
# price = np.array([150000, 165000, 180000, 195000, 210000, 220000, 230000, 240000, 280000])

# # Başlanğıc dəyərlər (təsadüfi seçilmiş əmsallar)
# m = 1300     # xəttin meyli (slope)
# b = 80000    # y-kəsişmə nöqtəsi (intercept)

# # Proqnoz (Y_pred = təxmin edilən qiymətlər)
# Y_pred = m * area + b
# print(Y_pred)   # ilkin proqnozlar

# # Hyperparametrlər
# L = 0.0001      # learning rate (öyrənmə sürəti)
# epochs = 40000  # təkrarlama sayı
# n = len(area)   # nümunələrin sayı

# # Gradient descent dövrü
# for i in range(epochs):
#     # Mövcud modelə görə proqnoz
#     Y_pred = m * area + b

#     # Gradientlərin hesablanması
#     # D_m və D_b - xəta funksiyasının törəmələri (slope və intercept üçün)
#     D_m = (-2/n) * sum(area * (price - Y_pred))  # m üzrə dəyişmə
#     D_b = (-2/n) * sum(price - Y_pred)           # b üzrə dəyişmə

#     # Əmsalların yenilənməsi
#     m = m - L * D_m
#     b = b - L * D_b

# # Nəticə əmsallar (m və b)
# print("Öyrənilmiş m:", m)
# print("Öyrənilmiş b:", b)

# # Modelin keyfiyyət göstəriciləri
# mae = mean_absolute_error(price, Y_pred)  # orta mütləq xəta
# mse = mean_squared_error(price, Y_pred)   # orta kvadrat xəta
# r2 = r2_score(price, Y_pred)              # R^2 skor (modelin uyğunluğu)

# print("================")
# print("MAE:", mae)
# print("MSE:", mse)
# print("R2 Score:", r2)



# Sadə Linear Regression üçün formulla (covariance ilə) hesablamaq daha rahatdır.
# Amma öyrənmə və machine learning üçün — Gradient Descent daha vacibdir və daha güclü üsuldur.

#endregion

#region PythonAi8

# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler,OneHotEncoder


#region Uzun versiya 
# # ------------------------------
# # 1) MISSING VALUES (NaN) İMPUTATION
# # ------------------------------

# # DataFrame yaradırıq (bəzilərində NaN boş dəyərlər var)
# data = {
#     'Rooms': [2, 3, np.nan, 4, 3],
#     'Area_m2': [60, 80, 100, np.nan, 120],
#     'Price_AZN': [90000, 120000, 150000, 200000, np.nan]
# }

# # Əsas DataFrame
# df = pd.DataFrame(data)

# # NaN dəyərlər median ilə doldurulacaq
# # strategy='median' → boş yerləri median ilə əvəz edir
# imputer = SimpleImputer(strategy='median')

# print("---- Əvvəlki DataFrame ----")
# print(df)

# # fit_transform() → həm öyrənir, həm doldurur
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# print("\n---- NaN-lar doldurulmuş DataFrame ----")
# print(df_imputed)



# # ------------------------------
# # 2) ONE-HOT ENCODING (Kategoriya -> Sayı formatı)
# # ------------------------------

# # Rayon adları (kategoriya məlumatı)
# df = pd.DataFrame({
#     'District': ['Yasamal', 'Nizami', 'Sebayil', 'Yasamal', 'Sebayil']
# })

# # OneHotEncoder → hər rayon üçün ayrıca sütun yaradır (0 və 1)
# encoder = OneHotEncoder(sparse_output=False)

# # fit_transform() → həm öyrənir, həm çevrilir
# encoded = encoder.fit_transform(df[['District']])

# # Yeni sütun adlarını alırıq
# encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['District']))

# print("\n---- One-Hot Encoding nəticəsi ----")
# print(encoded_df)


# from sklearn.preprocessing import StandardScaler

# X=pd.DataFrame({
#     'Area_m2':[50,70,100,150,200],
#     'Rooms':[1,2,3,4,5]
# })

# scaler = StandardScaler()
# scaled=scaler.fit_transform(X)
# scaled_df = pd.DataFrame(scaled,columns=X.columns)
# print(scaled_df)



#scaling prfonrmansin artmasi ve hesablamanin balansi olmasi ucundue?


# Scaling modelin rəqəmləri daha yaxşı başa düşməsi, daha tez öyrənməsi və daha düzgün nəticə verməsi üçündür.




# 🔵 Multiple Linear Regression nədir?

# Bu, birdən çox dəyişən istifadə edərək bir nəticəni proqnoz edən modeldir.

# y^​=b+a1​x1​+a2​x2​+...+an​xn​


# 🔵 Multiple Linear Regression ML-in hansı hissəsinə daxildir?
# ✔ Machine Learning → Supervised Learning → Regression

# Bu ardıcıllıqla gedir:

# Machine Learning (Ümumi sahə)

# Supervised Learning (Nəzarət olunan öyrənmə — modelə həm input, həm də cavab verilir)

# Regression (Nəticə rəqəm olanda)

# Linear Regression

# Multiple Linear Regression

# Yəni struktur belədir:

# Machine Learning
#  └── Supervised Learning
#       └── Regression
#            └── Linear Regression
#                 └── Multiple Linear Regression

# 🔵 Niyə ML sayılır?

# Çünki:

# Model məlumatdan öyrənir

# Öyrənilən əmsallarla (a₁, a₂, a₃...) proqnoz edir

# Xəta azaldılır, model optimallaşdırılır

# Yeni məlumat verəndə cavab tapır

# Bu klassik ML davranışıdır.


# import random

# np.random.seed(42)
# districts = ["Yasamal", "Nizami", "Sabayil", "Khatai", "Binagadi", "Narimanov"]
# building_types = ["New", "Old", "Premium", "Economy"]

# data = {
#     "Rooms": np.random.randint(1, 6, 100),
#     "Area_m2": np.random.randint(40, 250, 100),
#     "District": [random.choice(districts) for _ in range(100)],
#     "BuildingType": [random.choice(building_types) for _ in range(100)],
#     "Floor": np.random.choice([1, 2, 3, 4, 5, np.nan], 100, p=[0.15,0.15,0.2,0.2,0.2,0.1]),
#     "YearBuilt": np.random.choice([2000, 2005, 2010, 2015, 2020, np.nan], 100, p=[0.15,0.15,0.2,0.2,0.2,0.1]),
# }

# price = (
#     data["Area_m2"] * 1000
#     + data["Rooms"] * 8000
#     + np.random.randint(-30000, 30000, 100)
# )
# data["Price_AZN"] = price

# df = pd.DataFrame(data)
# df.to_excel("houses_extended.xlsx", index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.impute import SimpleImputer



df=pd.read_excel("houses_extended.xlsx")
#print(df.head())
#print(df.info())



# X → modelin istifadə edəcəyi input məlumatlar (Price_AZN sütunu çıxılıb)
X = df.drop("Price_AZN", axis=1)

# y → modelin proqnoz etməli olduğu nəticə (Price_AZN)
y = df["Price_AZN"]

# Rəqəmli (numeric) sütunların siyahısı
num_cols = ["Rooms", "Area_m2", "Floor", "YearBuilt"]

# Kategorik (categorical) sütunların siyahısı
cat_cols = ["District", "BuildingType"]

# ==========================================
# Rəqəmli məlumatlar üçün pipeline
# 1) NaN-ları median ilə doldurur
# 2) Rəqəmləri StandardScaler ilə standartlaşdırır
numeric_transformer = Pipeline(steps=[
   ('imputer', SimpleImputer(strategy='median')),
   ('scaler', StandardScaler())
])

# ==========================================
# Kategorik məlumatlar üçün pipeline
# 1) NaN-ları ən çox təkrarlanan dəyərlə doldurur
# 2) One-hot encoding ilə hər kateqoriyanı sütuna çevirir
#    handle_unknown='ignore' → train-də olmayan dəyərlər gəlsə xətaya düşməsin
categorical_transformer = Pipeline(steps=[
   ('imputer', SimpleImputer(strategy='most_frequent')),
   ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# ==========================================
# ColumnTransformer ilə bütün məlumatları birləşdiririk
# - Rəqəmli sütunlara numeric_transformer tətbiq olunur
# - Kategorik sütunlara categorical_transformer tətbiq olunur
preprocessor = ColumnTransformer(
   transformers=[
       ('num', numeric_transformer, num_cols),
       ('cat', categorical_transformer, cat_cols),
   ]
)

# ==========================================
# 1️⃣ Pipeline ilə model yaratmaq
# 'preprocessor' → ColumnTransformer-i tətbiq edir (numeric + categorical preprocessing)
# 'regressor'   → Linear Regression modelini əlavə edir
model = Pipeline(steps=[
   ('preprocessor', preprocessor),
   ('regressor', LinearRegression())
])

# ==========================================
# 2️⃣ Train və test set-lərə bölmək
# X_train, y_train → modelin öyrənəcəyi məlumatlar (training data)
# X_test, y_test   → modelin performansını yoxlayacağı məlumatlar (test data)
# test_size=0.2    → verilənlərin 20%-i test üçün, 80%-i train üçün ayrılır
# random_state=42  → nəticələrin təkrar eyni olması üçün seed təyin olunur
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)


pred=model.predict(X_test)  #random secilmis 20 % gore hesablanir.


mae=mean_absolute_error(y_test,pred)
mse=mean_squared_error(y_test,pred)
r2=r2_score(y_test,pred)
print(mae)
print(mse)
print(r2)



# encoder = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
# encoded_feature_names = encoder.get_feature_names_out(cat_cols)
#
# feature_names = num_cols + list(encoded_feature_names)
#
# coef = model.named_steps["regressor"].coef_
# importance = pd.Series(coef, index=feature_names).sort_values(ascending=False)
#
# print("\nƏn çox təsir edən sütunlar:\n")
# print(importance.head(10))
#endregion



#region Qisa versiya

# 1️⃣ Dataset və target

# Input features (X) → Rooms, Area_m2, Floor, YearBuilt, District, BuildingType

# Target (y) → Price_AZN (proqnoz etmək istədiyimiz qiymət)

# 2️⃣ Feature növləri

# Numeric (rəqəmli) → Rooms, Area_m2, Floor, YearBuilt

# Categorical (kateqorik) → District, BuildingType

# 3️⃣ Data preprocessing (ön emal)

# Numeric pipeline:

# SimpleImputer(strategy='median') → NaN-ları median ilə doldurur

# StandardScaler() → bütün rəqəmləri standartlaşdırır (mean=0, std=1)

# Categorical pipeline:

# SimpleImputer(strategy='most_frequent') → NaN-ları ən çox təkrarlanan dəyərlə doldurur

# OneHotEncoder(handle_unknown='ignore') → hər kateqoriyanı 0/1 sütunlarına çevirir

# Bütün sütunları birləşdirir: ColumnTransformer

# 4️⃣ Model

# Linear Regression → bir neçə input feature-dan price-i proqnoz edir

# Pipeline-da həm preprocessing, həm model bir yerdədir

# 5️⃣ Train/Test split

# train_test_split(test_size=0.2) → 80% train, 20% test

# Random state 42 → nəticə təkrar olunur

# 6️⃣ Model öyrədilməsi
# model.fit(X_train, y_train)


# Pipeline avtomatik olaraq:

# Numeric və categorical preprocessing edir

# Linear Regression-i öyrədir

# 7️⃣ Performance ölçüləri

# MAE → orta abs(xəta)

# MSE → orta kvadrat xətası

# R² → modelin izahat gücü (1.0 yaxşı, 0.0 pis)

# mae=mean_absolute_error(y_test,pred)
# mse=mean_squared_error(y_test,pred)
# r2=r2_score(y_test,pred)

# 8️⃣ Yeni məlumatdan proqnoz

# İstifadəçi input verir: Rooms, Area_m2, District, BuildingType, Floor, YearBuilt

# Yeni DataFrame yaradılır → model.predict(new_df) ilə price təxmin olunur

# 9️⃣ Nəticə

# Pipeline + Linear Regression → tam ML workflow

# Kod bütün preprocessing-i avtomatik edir → NaN-ları doldurur, scale edir, one-hot encoding tətbiq edir

# Model təlim olunub → yeni input üçün qiymət təxmin edir


# StandartScaler rəqəmli sütunları 0 ortalama, 1 standart sapma ilə normalizə edir ki, model tez, stabil və balanslı öyrənsin.

 

# | Funksiya          | Nə edir                                   |
# | ----------------- | ----------------------------------------- |
# | `fit()`           | Parametrləri öyrənir                      |
# | `transform()`     | Məlumatı öyrənilmiş parametrlərlə çevirir |
# | `fit_transform()` | Həm öyrənir, həm çevirir                  |



#endregion




#endregion 



#region PythonAi9
# 1)B+
# 2)B+
# 3)C+
# 4)B+
# 5)A+
# 6)A+
# 7)B+
# 8)B+
# 9)B+
# 10)A+
# 11)D- CAVAB B-DIR.
# 12)B+
# 13)A+
# 14)A+
# 15)B+
# 16)A+
# 17)A+
# 18)A+
# 19)C+
# 20)A+


# Lesson 9 cavablari ve imtahan ucun bir daha bax.Ve hemcinin aciq suallara da bax.

#endregion



#region PythonAi10

#  Polynominal and L1 L2 



# Overfitting — Machine Learning modelinin təlim (training) məlumatını həddindən artıq əzbərləməsi deməkdir.
#  Model verilənləri real nümunə kimi yox, sanki yaddaş kimi saxlayır

# Nəticədə:

# Training-də çox yaxşı nəticə verir

# Test (real) məlumatlarda isə pis işləyir

# Yəni model ümumiləşdirə bilmir, sadəcə yadda saxlayır.



# Underfitting — modelin həm training, həm də test məlumatlarında pis nəticə verməsidir.

# Yəni model çox sadədir, məlumatın içindəki əlaqələri öyrənə bilmir.





# Burada sklearn 5 testdən 5 nəticə çıxarır.

# scores = cross_val_score(...) → bu cross validation scores deməkdir.

# Yəni:

# ✔️ Cross-validation = modeli bir neçə dəfə (məs: 5 dəfə) fərqli hissələrdə test etmək





# Polynomial Regression

# Polynomial Regression — linear regression-in bir variantıdır, amma düz xətt əvəzinə əyri xətt çəkməyə imkan verən regresiya üsuludur.

# y=a+bx+cx^2

# y → Modelin proqnoz etdiyi nəticə (dependent variable).

# a → Intercept (sabit termin). Yəni x=0 olarkən y-nin dəyəri.

# b → x-in çəkisi (weight). Bu xətti termin üçün əhəmiyyətini göstərir.

# c → x²-in çəkisi (weight). Bu kvadratik termin üçün əhəmiyyətini göstərir.

# x → Müstəqil dəyişən (feature).



#L1-Lasso
#L2-Ridge

#Regualization-

# Lasso (L1): Az təsir göstərən (önəmsiz) feature-ləri tam sıfıra çevirir, yəni onları modeldən çıxarır.

# Ridge (L2): Çox təsir göstərən (önəmli) feature-lərin çəkilərini azaldır, amma heç birini sıfıra çevirmir.



#endregion



#region PythonAi11

#Decision Tree and Rf

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split


# https://medium.com/@shrutimisra/interpretable-ai-decision-trees-f9698e94ef9b (decision treenin sekli)



# # Decision Tree Terminləri

# # 1)Root Node (Kök Düyün)

# # Ağacın başlanğıc nöqtəsi

# # Bütün məlumatlar buradan bölünməyə başlayır

# # Məsələn: “Rəngi qırmızıdır?” sualı root node ola bilər



# # 2)Decision Node (Qərar Düyünü / Daxili Düyün)

# # Kökdən sonra gələn və məlumatı bölən düyünlər

# # Hər bir düyün müəyyən xüsusiyyətə görə qruplar yaradır

# # Məsələn: “Yumşaqdır?” sualı decision node ola bilər

# # Leaf Node (Yarpaqlar / Son Düyün)

# # Ağacın nəticə verdiyi düyünlər




# # 3) Leaf node-da artıq proqnoz və ya nəticə var, yeni qərar verilmir

# # Məsələn: “Alma”, “Banan”, “Kivi” leaf node-dur



# # 4)Subtree (Alt Ağac)

# # Decision node-dan başlayan və leaf node ilə bitən ağacın kiçik hissəsi

# # Hər decision node öz subtree-inə malikdir

# # Başqa sözlə, subtree ağacın bir kiçik hissəsi, özü də kiçik bir ağacdır

#yəni decison node+leaf node=subtree

# # 5)Entropy (Entropiya)

# # Dataset-dəki qarışıqlıq və qeyri-müəyyənlik səviyyəsini ölçən göstərici

# # Dataset tam qarışıqdırsa → entropy yüksək

# # Dataset tam təmizdirsə → entropy = 0


   # Sadə dillə desək, qarışıqlıq dedikdə “datasetdəki nümunələrin müxtəlif siniflərə (labels) necə paylandığı” nəzərdə tutulur.

   # Əgər bütün nümunələr eyni sinifdədirsə → qarışıqlıq yoxdur.

   # Əgər nümunələr fərqli siniflər üzrə bərabər paylanıbsa → qarışıqlıq yüksəkdir.


# # 6)Information Gain (Məlumat Qazancı / IG)


# Information Gain = bir feature istifadə edərək məlumatdakı qeyri-müəyyənliyi nə qədər azalda bilərik.

# # IG=Entropy(S)−Weighted Entropy of subgroup

# # IG=0.881−0.583≈0.29


# # ✅ Nəticə: Decision Tree tətbiqindən sonra qarışıqlıq azaldı

# # Başlanğıc qarışıqlıq = 0.881

# # Bölmədən sonra = 0.583

# # Fərq = 0.298 → bu bölmə ilə məlumat daha “təmiz” oldu

# # 4️⃣ Sadə desək

# # Başlanğıc qarışıqlıq: dataset qarışıqdır, proqnoz qeyri-müəyyəndir

# # Decision Tree tətbiq etdikdən sonra: məlumat xüsusiyyətlərə görə qruplara ayrılır, qarışıqlıq azalır, nəticələr daha dəqiq olur



# # Decision Tree — verilənləri xüsusiyyətlərinə görə ardıcıl olaraq bölən və nəticədə qərar verən ağac strukturu olan bir məşhur nəzarətli öyrənmə (supervised learning) üsuludur.



# #Random Forest Tree

# # Random Forest — çoxlu Decision Tree-lərin (Qərar Ağacları) birləşməsidir.

# # Tək ağac = Decision Tree

# # Bir neçə ağacın birlikdə işləməsi = Random Forest




# rf=RandomForestRegressor(
#     n_estimators=400,
#     max_depth=4,
#     min_samples_split=4,
#     n_jobs=-1,
#     random_state=42
# )

# # 🌲 RandomForestRegressor Parametrlərinin TƏRİFLƏRİ
# # 1️⃣ n_estimators

# # Tərif:
# # ➡ Random Forest-in içində qurulacaq decision tree-lərin sayı.

# # Sənin dəyərin: 400
# # Yəni model 400 ağac yaradacaq.

# # 2️⃣ max_depth

# # Tərif:
# # ➡ Hər decision tree-nin icazə verilən maksimum dərinliyi (neçə səviyyə enə biləcəyi).

# # Sənin dəyərin: 4
# # Yəni hər ağac maksimum 4 səviyyə olacaq.

# # 3️⃣ min_samples_split

# # Tərif:
# # ➡ Bir node-un iki yerə bölünməsi üçün minimum lazım olan sample sayı.

# # Sənin dəyərin: 4
# # Node içində 4-dən az sample varsa, bölünməyəcək.

# # 4️⃣ n_jobs

# # Tərif:
# # ➡ Modelin train zamanı istifadə edəcəyi CPU nüvələrinin sayı.

# # Sənin dəyərin: -1
# # Bu deməkdir: bütün CPU nüvələrini istifadə et → maksimum sürət.

# # 5️⃣ random_state

# # Tərif:
# # ➡ Bütün random prosesləri (data seçimi, feature seçimi, split-lər) sabitləşdirən toxum (seed).

# # Sənin dəyərin: 42
# # Yəni model hər dəfə eyni nəticəni verəcək.

# # ✨ QISA XÜLASƏ
# # Parametr	Tərif
# # n_estimators	Ağacların sayı
# # max_depth	Ağacın maksimum dərinliyi
# # min_samples_split	Split üçün lazım olan minimum sample
# # n_jobs	CPU sayı (paralelləşmə)
# # random_state	Nəticəni sabit saxlamaq üçün random toxum



# # ✔ NƏTİCƏ (super sadə)

# # Random Forest = çox decision tree → səhvləri ortalaşdırır → daha güclü model yaradır.

# # Bu səbəbdən istifadə edirik:

# # ✓ daha stabil
# # ✓ daha dəqiq
# # ✓ daha az overfitting
# # ✓ daha etibarlı
# # ✓ daha güclü nəticə











#endregion


#region PythonAi12


# GB EGB


# 🌲 1) Random Forest — paralel ağaclar

# Nədir?
# Birdən çox decision tree eyni anda (paralel) qurulur və nəticələri birləşdirilir.

# Niyə belə edir?
# Çünki çox ağac birlikdə daha stabil nəticə verir.

# Necə işləyir?

# Hər ağac dataset-in bir hissəsini görür

# Hər ağac təsadüfi feature-lər seçir

# Sonda bütün ağacların nəticələri birləşdirilir (səsvermə / orta)

# 👉 Ağaclar bir-birinin səhvini düzəltmir.
# Hamısı eyni anda işləyir (paralel).

# 🔥 2) Gradient Boosting — ardıcıl ağaclar

# Nədir?
# Decision tree-lər ardıcıl (sequence) qurulur və sonrakı ağac əvvəlki ağacın səhvlərini düzəltməyə çalışır.

# Necə işləyir?

# İlk ağac sadə proqnoz edir → səhv edir

# İkinci ağac həmin səhvləri öyrənir və düzəltməyə çalışır

# Üçüncü ağac əvvəlkilərin qalan səhvlərini düzəldir

# Belə-belə hər yeni ağac daha dəqiq olur

# 🔍 Yəni:
# təkmilləşdirilən ardıcıl ağaclar → daha dəqiq model

# ⚡ 3) XGBoost (Extreme Gradient Boosting)

# Gradient Boosting-in daha güclü, daha sürətli və daha az overfitting edən versiyasıdır.

# Üstünlükləri:

# regularization var

# daha sürətli optimizasiya

# RAM istifadə çox effektli

# ən çox Kaggle yarışmalarının qalibi → XGBoost






#endregion



#region PythonAi13



# 🚀 ANN nədir?
# ANN = Artificial Neural Network = Süni Neyron Şəbəkəsi
# Komputerin beyin kimi öyrənmə üsuludur.

# 🧠 ANN necə işləyir?
# ANN məlumatı çoxlu kiçik neyronlar içindən keçirərək nəticə çıxaran alqoritmdir.

# 🔌 ANN-in strukturu
# 1️⃣ Input Layer — Giriş (məsələn, 13 xüsusiyyət)
# 2️⃣ Hidden Layer — Gizli qatlar (hesablama və öyrənmə burada baş verir)
# 3️⃣ Output Layer — Çıxış (məsələn, 0 və ya 1)


# 🏁 ANN-in istifadə sahələri
# ✔ Üz tanıma
# ✔ Səs tanıma
# ✔ Şəkil təsnifatı
# ✔ Proqnozlamalar
# ✔ Tibbi diaqnostika
# ✔ Döyüş oyunlarında botlar
# ✔ ChatGPT və digər AI modelləri

# 📊 Regression vs Classification
# | Xüsusiyyət     | Regression                          | Classification                        |

# | Çıxış tipi     | Real rəqəm (continuous)             | Sinif (categorical)                   |
# | Sual tipi      | Nə qədər / neçə?                    | Hansı? Hə/Yox?                        |
# | Nümunə         | Ev qiyməti, maaş, temperatur        | Xəstə/sağlam, spam, pişik/it          |
# | Ehtimal        | ❌ Yox                               | ✅ Ola bilər (sigmoid/softmax)         |
# | Model nümunəsi | Linear Regression, ANN (linear)    | Logistic Regression, ANN (sigmoid)   |

# 🟢 Neyronun linear çıxışı (1 neyron)
# Düstur: y = w1*x1 + w2*x2 + ... + wn*xn + b
# x → giriş məlumatları (features), misal: yaş, boy, çəki
# w → girişlərin çəkisi (weight), böyük çəkilər → daha əhəmiyyətli
# b → bias (sabit dəyər)

# Misal:
# x1 = 2, x2 = 3
# w1 = 0.5, w2 = 1.2
# b = 0.7
# y = 0.5*2 + 1.2*3 + 0.7 = 5.3

# 🟢 Bias nədir?
# Bias = neyronun başlanğıc nöqtəsi, girişlər 0 olsa da çıxış verə bilir
# Misal:
# x1 = 0, x2 = 0, w1 = 0.5, w2 = 1.2, b = 0.7 → y = 0.7

# 🟢 Input → Weight → Sum → Activation → Output
# - Input: x1, x2, ..., xn
# - Weight: w1, w2, ..., wn
# - Sum: Σ(wx) + b
# - Activation: Step / Sigmoid / ReLU
# - Output: Neyronun proqnozu (0/1 və ya ehtimal)

# 🟢 Perceptron
# - Ən sadə neyron modeli
# - Binary classification üçün
# - Aktivasiya funksiyası: Step (0/1)

# 🟢 Multi-Layer Perceptron (MLP)
# - Çox qatlı neyron şəbəkəsi
# - Input layer → Hidden layers → Output layer
# - Gizli qatlar mürəkkəb patternləri öyrənir
# - Aktivasiya funksiyası: ReLU, Sigmoid, Softmax
# - Binary və Multi-class classification, regression üçün istifadə oluna bilər

# 🔹 Linear vs Non-linear
# - Linear neuron: y = w1*x1 + w2*x2 + ... + wn*xn + b → düz xətt
# - Non-linear neuron: y = activation(wx + b) → parabola, sigmoid, softmax
# - Non-linear olmadan mürəkkəb patternlər öyrənilə bilməz

# 🟢 Qaydalar / əsas anlayışlar
# 1. Hər giriş öz çəkisi ilə vurulur, sonra hamısı toplanır, bias əlavə olunur.
# 2. Aktivasiya funksiyası linear çıxışı ehtimala və ya 0/1 kimi sərt çıxışa çevirir.
# 3. Bias olmadan xətt həmişə orijindən keçir, model məlumatı yaxşı uyğunlaşdıra bilmir.
# 4. MLP-də hidden qatlar modelin non-linear patternləri öyrənməsini təmin edir.
# 5. ANN-in çıxışı problemi görə dəyişir:
#    - Binary classification → 0/1 və ya 0–1 ehtimal
#    - Multi-class classification → sinif indeksləri (0,1,2,...)
#    - Regression → real dəyər




# 🟢 Perceptron və Neyronun İşləmə Mexanizmi

#meselen 13 neyron inputu varsa hiddenda 32 olmalidi

# Perceptron = ən sadə neyron (Artificial Neuron) modelidir.

# x1 --- w1 \
# x2 --- w2  ---> Σ (toplama) ---> Aktivasiya → y (0/1)
# x3 --- w3 /
#           +
#           b (bias)

#Input=>Weight=>Sum=>Activation=>Output(1 neyronun isi)

#endregion

#region PythonAi14



#ilk 25 deq sual cavab

#PyTorch
# PyTorch Facebook (Meta) tərəfindən hazırlanmış, açıq-mənbə (open-source),
#  xüsusilə dərin öyrənmə (deep learning) və neyron şəbəkələri
#  qurmaq üçün istifadə olunan çox güclü bir machine learning framework-dür.





# Scalar -> 5
# Vector -> [2,3,4]
# Matrix -> [[1,2],[3,4]]
# Tensor -> [[[[1,1]],[2,2],[3,3]],[[4,4],[5,5],[6,6]],[[7,7]]]]


# | Ad     | Ölcü | Nümunə                 |
# | ------ | ---- | ---------------------- |
# | Scalar | 0D   | `5`                    |
# | Vector | 1D   | `[2,3,4]`              |
# | Matrix | 2D   | `[[1,2],[3,4]]`        |
# | Tensor | 3D+  | `[[[[1,1]],[2,2]...]]` |

import torch
#
# #Scalar
a=torch.tensor(5)
#Vector
b=torch.tensor([1,2,3])
#Matrix
c=torch.tensor([[1,2],[3,4]])






#recordingin 1-ci hissesi 01.02.00


# 1️⃣ Fully Connected Layer (nn.Linear)
#
# nn.Linear(in_features, out_features) → hər bir giriş neyronu hər çıxış neyronuna bağlıdır.
#
# Buna fully connected (tam bağlı) layer deyilir.

# fc1: 2 giriş neyronu → 4 çıxış neyronu
#
# Hər 2 giriş hər 4 çıxış neyronuna bağlıdır → fully connected
#
# fc2: 4 giriş (hidden layer) → 1 çıxış
#
# Hər 4 giriş çıxış neyronuna bağlıdır → fully connected


# Hidden layer inputdan böyük olmalıdır?
#
# Xeyr, məcbur deyil.
#
# Amma input-dan bir az daha böyük seçmək normaldır, ki model daha mürəkkəb nümunələri öyrənsin.



# 1️⃣ Activation function nədir?
#
# Activation function (aktivasiya funksiyası) → neyronun çıxışını müəyyən qaydada dəyişdirən funksiyadır.
#
# Neyron şəbəkədə non-linearlıq əlavə etmək üçün istifadə olunur.
#


# | Funksiya | İstifadə                                                              |
# | -------- | --------------------------------------------------------------------- |
# | ReLU     | Hidden layer-lərdə (0-dan böyük dəyərləri saxlayır, mənfiləri 0 edir) |
# | Sigmoid  | Çıxış layer-də, ehtimal üçün (0-1 aralığı)                            |
# | Softmax  | Multi-class classification, ehtimalların cəmi 1 olur                  |


# 1️⃣ Aktivasiya funksiyasının yeri
#
# ANN (Artificial Neural Network)-də aktivasiya funksiyası layer-lərin çıxışında yerləşir.
#
# Hər hidden layer-in sonunda
#
# Output layer-dən əvvəl (çox vaxt ehtimala çevirmək üçün)







#endregion


#region PythonAi15


#ilk 23 deq sual cavab




# Activation funksiyaları neyron şəbəkələrində neyronun çıxışını hesablamaq üçün istifadə olunur.(yeni cixisdan evvel hidden layerdan sonra )
# Onlar neyronun “aktiv olub-olmamasını” müəyyənləşdirir və modelə xətti olmayanlıq (non-linearity) əlavə edir.
# Əgər activation funksiyası olmasa, neyron şəbəkəsi yalnız xətti funksiyaları öyrənə bilər və mürəkkəb nümunələri tanıya bilməz.




#her birini nezeri numune yaz............


# Sigmoid – 0–1 arası ehtimal verir, adətən binary classification üçün.

# Softmax – 0–1 arası ehtimal verir, multi-class classification üçün (siniflər üzrə cəmi 1 olur).


# 1️⃣ Sigmoid

# Çıxış: 0 – 1 arası

# İstifadə: Binary classification (ikili təsnifat)

# Dezavantaj: Vanishing gradient problem (çox böyük və ya kiçik x dəyərlərində gradient itir)



# 2️⃣ ReLU (Rectified Linear Unit)

# Çıxış: 0 – ∞

# Mənfi dəyərləri 0 edir

# İstifadə: Hidden layer-lərdə çox istifadə olunur

# Dezavantaj: Dead neuron problem (bəzən neyron tamamilə deaktiv ola bilər)


# 3️⃣ Softmax

# Çıxış: 0 – 1 arası, cəmi 1

# İstifadə: Multi-class classification (çoxlu sinifli təsnifat)

#Dezavantaj: Softmax çoxlu siniflər üçün əla ehtimal verir, amma çox böyük və ya çoxlu logit-lərdə həssas və ağır ola bilər.




# ReLU: mənfiləri tam 0 edir

# Sigmoid: mənfiləri 0-a yaxın, amma sıfır deyil edir


# Kodun izahı



# ReLU → hidden layer-lərdə istifadə olunur (mənfiləri 0 edir, non-linearity əlavə edir)

# Sigmoid → çıxışda ehtimal verir (0–1 arası), çünki xəstəliyin olub-olmaması binary



#endregion


#region PythonAi16




# CNN nədir? (Convolutional Neural Network)

# CNN – Konvolyusion Neyron Şəbəkəsi deməkdir. Bu, şəkil, video, obyekt tanıma, təsnifat, üz tanıma kimi vizual məlumatlarla işləmək üçün yaradılmış xüsusi neyron şəbəkə növüdür.




# CNN nə iş görür?

# Şəkillərdə xüsusiyyətləri (edges, rəng keçidləri, formalar) özü avtomatik tapır.

# İnsan beyninin görmə sisteminə bənzəyir — əvvəl xırda şeyləri tapır, sonra daha böyük strukturları anlayır.






# Niyə adi neyron şəbəkədən fərqlidir?

# Adi şəbəkələr bütün piksellərə birdən baxır. CNN isə şəkli kiçik hissələrə bölüb filterlərlə (kernel) “süzür” və maraqlı nümunələri çıxarır.






# CNN-in əsas hissələri

# 1. Input Layer

# Şəkilin daxil olduğu qat (məsələn: 224×224×3).

# 2. Convolution Layer

# Şəkili filterlərlə (kernel) gəzib xüsusiyyətləri (edges, forms) çıxarır.

# 3. ReLU Layer

# Aktivasiya funksiyasıdır — mənfi dəyərləri sıfırlayır, modeli qeyri-xətti edir.

# 4. Pooling Layer (Max/Average Pooling)

# Şəkili kiçildir (downsampling)

# Vacib məlumatı saxlayır

# Qeyd: Pooling = Convolution + ReLU deyil.
# Pooling ayrıca bir qattır.

# 5. Flatten Layer

# 2D matrisi 1D vektora çevirir ki, fully connected layer istifadə edə bilsin.

# 6. Fully Connected (Dense) Layer

# Son təsnifatı edir.
# Məsələn: pişik / it, rəqəm → 0–9 və s.





# CNN harada istifadə olunur?

# Üz tanıma (Face ID)

# Obyekt tanıma (YOLO, Tesla-nın kameraları)

# Tibb (rentgen analizi)

# Kamera təsviri yaxşılaşdırma

# Çatbotlarda OCR (şəkildən mətn oxuma)

# Bir cümləlik yekun

# 👉 CNN – şəkilləri anlamaq üçün ən güclü süni intellekt modelidir.






# “ANN şəkili düz başa düşmür, çünki onu əvvəlcə parçalayır.”

# ✔️ Məna düzgün – ANN şəkili 1D edərək strukturunu itirir.

# “2D struktur, yaxınlıq əlaqələri, formalar hamısı itir.”

# ✔️ Düzgün – bu ANN-in görüntü üçün əsas problemidir.

# “Parçalayıb sonra analiz edir deyə problemdir.”

# ✔️ Bəli, problem məhz budur – 1D çevrilməsi nəticəsində məkan əlaqələri qorunmur.






# ⭐ Edge Detection nədir?

# Edge Detection — şəkildə kənar xətləri tapmaq deməkdir.

# Yəni şəkildə rəngin və ya işıqlığın kəskin dəyişdiyi yerləri aşkar edir.






# SNR=Bir məlumatda (şəkil, audio, sensor, video) “faydalı siqnal” səs-küydən güclüdür, ya yox — bunu ölçmək üçün istifadə olunur.

# Şəkil analizi (Image Processing)
# Burada SNR istifadə olunur ki:
# Şəkil nə qədər təmizdir?
# Noise çoxdur ya azdır?
# Filtr (Gaussian, Median və s.) görüntünü nə qədər təmizlədi?
# Reconstruction algoritmi (SR, Autoencoder) nəticəni yaxşılaşdırdı?


# Daha yaxşı model çıxışı üçün input keyfiyyətini ölçmək.


# MLP sinir şəbəkəsi modelidir.

# Ne ferqleri var?
# SNR-in MLP ilə heç bir birbaşa əlaqəsi yoxdur.
# SNR sadəcə məlumatın səsli-səssiz olub-olmadığını göstərir.
# MLP isə o məlumatdan öyrənən modeldir.

# Niyə görə şəkil analizində SNR istifadə edirdik?
# Çünki şəkillər də əslində siqnaldır — 2D siqnal.
#  Hər piksel = məlumat.
#  Şəkildə səs-küy (noise) varsa → modelin görməsi və öyrənməsi pisləşir.
# Bu səbəbdən şəkil emalı və Computer Vision-da SNR çox kritikdir.
# ve sekillerimiz bezen temiz olmaya biler. Bu halda SNR:
# Şəklin nə qədər “təmiz” olduğunu ölçür
# Noise-u azaltma metodlarının effektivliyini müqayisə etməyə imkan verir
# Modelə verəcəyin input-un keyfiyyətini yoxlamaq üçündür



#endregion


#region PythonAi17

# Lesson17:

# ResNet18= ResNet-18, 2015-ci ildə Microsoft Research tərəfindən yaradılmış Residual Network ailəsinə aid olan, 18 qatlı (layer) bir Convolutional Neural Network-dir (CNN).
# Buradakı “18” sadəcə — qatların sayıdır.
# neye gore ResNet18?: Modelə “layer-ləri keçib getməyə” icazə verir → bu da dərin şəbəkələrdə yaranan vanishing gradient problemini öldürür.
# ResNet-18-in üstünlükləri
# Yüngül və sürətli
# Az GPU RAM istəyir
# Training-i stabil
# Overfitting az olur
# Transfer learning üçün çox əlverişli
# Accuracy normaldır (ResNet50 qədər olmasa da)


#endregion