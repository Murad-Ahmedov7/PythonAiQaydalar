
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

