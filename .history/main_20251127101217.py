

#HER BIRINE BIR NMUNE YAZ NEZERI OLARAQ 



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

# Studied	Sleep	Marks
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

#endregion