# **House Price Prediction**

Ini adalah project yang dibuat untuk mengikuti [Kaggle Data Science Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), dan sebagai pembelajaran.

## **Keterangan Data**
Terdapat 81 kolom dalam data [test.csv](https://github.com/mwahid60/house_price_prediction/blob/main/train.csv), berikut ke 81 kolom tersebut:

| Column | Non-Null Count | Data Type |
| :----: | :----: | :----: |
|Id            | 1460 | int64   |
|MSSubClass    | 1460 | int64   |
|MSZoning      | 1460 | object  |
|LotFrontage   | 1460 | float64 |
|LotArea       | 1460 | int64   |
|Street        | 91   | object  |
|Alley         | 1460 | object  |
|LotShape      | 1460 | object  |
|LandContour   | 1460 | object  |
|Utilities     | 1460 | object  |
|LotConfig     | 1460 | object  |
|LandSlope     | 1460 | object  |
|Neighborhood  | 1460 | object  |
|Condition1    | 1460 | object  |
|Condition2    | 1460 | object  |
|BldgType      | 1460 | object  |
|HouseStyle    | 1460 | object  |
|OverallQual   | 1460 | int64   |
|OverallCond   | 1460 | int64   |
|YearBuilt     | 1460 | int64   |
|YearRemodAdd  | 1460 | int64   |
|RoofStyle     | 1460 | object  |
|RoofMatl      | 1460 | object  |
|Exterior1st   | 1460 | object  |
|Exterior2nd   | 1460 | object  |
|MasVnrType    | 588  | object  |
|MasVnrArea    | 1452 | float64 |
|ExterQual     | 1460 | object  |
|ExterCond     | 1460 | object  |
|Foundation    | 1460 | object  |
|BsmtQual      | 1423 | object  |
|BsmtCond      | 1423 | object  |
|BsmtExposure  | 1422 | object  |
|BsmtFinType1  | 1423 | object  |
|BsmtFinSF1    | 1460 | int64   |
|BsmtFinType2  | 1422 | object  |
|BsmtFinSF2    | 1460 | int64   |
|BsmtUnfSF     | 1460 | int64   |
|TotalBsmtSF   | 1460 | int64   |
|Heating       | 1460 | object  |
|HeatingQC     | 1460 | object  |
|CentralAir    | 1460 | object  |
|Electrical    | 1459 | object  |
|1stFlrSF      | 1460 | int64   |
|2ndFlrSF      | 1460 | int64   |
|LowQualFinSF  | 1460 | int64   |
|GrLivArea     | 1460 | int64   |
|BsmtFullBath  | 1460 | int64   |
|BsmtHalfBath  | 1460 | int64   |
|FullBath      | 1460 | int64   |
|HalfBath      | 1460 | int64   |
|BedroomAbvGr  | 1460 | int64   |
|KitchenAbvGr  | 1460 | int64   |
|KitchenQual   | 1460 | object  |
|TotRmsAbvGrd  | 1460 | int64   |
|Functional    | 1460 | object  |
|Fireplaces    | 1460 | int64   |
|FireplaceQu   | 770  | object  |
|GarageType    | 1379 | object  |
|GarageYrBlt   | 1379 | float64 |
|GarageFinish  | 1379 | object  |
|GarageCars    | 1460 | int64   |
|GarageArea    | 1460 | int64   |
|GarageQual    | 1379 | object  |
|GarageCond    | 1379 | object  |
|PavedDrive    | 1460 | object  |
|WoodDeckSF    | 1460 | int64   |
|OpenPorchSF   | 1460 | int64   |
|EnclosedPorch | 1460 | int64   |
|3SsnPorch     | 1460 | int64   |
|ScreenPorch   | 1460 | int64   |
|PoolArea      | 1460 | int64   |
|PoolQC        | 7    | object  |
|Fence         | 281  | object  |
|MiscFeature   | 54   | object  |
|MiscVal       | 1460 | int64   |
|MoSold        | 1460 | int64   |
|YrSold        | 1460 | int64   |
|SaleType      | 1460 | object  |
|SaleCondition | 1460 | object  |
|SalePrice     | 1460 | int64   |

Disini saya menggunakan Pandas 2.0, pada data ini nilai kategorik **NA** dianggap sebagai missing value oleh pandas 2.0 sehingga harus dinyatakan ulang sebagai value **NA**.

Setelah menyatakan ulang value **NA**, lalu kita hitung ada berapa banyak missing value pada masing-masing kolom pada data. Disini kita prioritaskan kolom yang memiliki missing value paling banyak untuk di isi.

Ada beberapa metode yang dapat digunakan, untuk data kategorik disini saya menggunakan

``` Python
df_train["Alley"] = df_train["Alley"].fillna("NA")
```

Pada data ketegorik sebenarnya tidak ada missing, hanya saja pandas 2.0 yang menganggap string **NA** sebagai missing value sehingga hanya perlu dinyatakan ulang sebagai nilai string **NA**.

Sedangkan untuk data kontinyu yang berbentuk *integer* atau *float* disini saya mengisi missing value berdasarkan proporsi distribusi datanya, sebagai berikut:

``` Python
# Fill GarageYrBlt with distribution ratio

# Menghitung distribusi GarageYrBlt
proporsi = df_train["LotFrontage"].dropna().value_counts(normalize=True)

# target missing value
target = df_train["LotFrontage"].isna()

# menghitung jumlah missing value
miss_jumlah = target.sum()

# mengambil sample non-missing value dari LotFrontage
replace_value = np.random.choice(proporsi.index,    # index dari proporsi
                                 size=miss_jumlah,  # jumlah missing value
                                 p=proporsi.values  # nilai proporsi distribusi
                                 )

df_train.loc[target, "LotFrontage"] = replace_value
```

## **Data Encoding**
Mengubah setiap kolom kategorikal menjadi angka, sehingga dapat diproses menggunakan algoritma machine learning dan mendapatkan hasil prediksi yang baik.

Pada kolom **MSSubClass** nilainya sudah berbentuk angka (20, 30, 40, ...). Sehingga disini saya menggunakan `.loc[]` untuk mengubah ulang nilai pada kolom tersebut.

``` Python
df_train.loc[df_train["MSSubClass"]==20, "MSSubClass"] =1
df_train.loc[df_train["MSSubClass"]==30, "MSSubClass"] =2
df_train.loc[df_train["MSSubClass"]==40, "MSSubClass"] =3
# ...
```

Sedangkan untuk kolom kategorikal yang berbentuk string saya menggunakan `.map()` untuk mengganti nilai string tersebut menjadi angka.

``` Python
# Mapping MSZoning 1-8 
map_MSZoning = {
    'A': 1,
    'C (all)': 2,
    'FV': 3,
    'I': 4,
    'RH': 5,
    'RM': 6,
    'RL': 7,
    'RP': 8
}

# Train Data
# Mengganti Nilai MSZoning
df_train["MSZoning"] = df_train["MSZoning"].map(map_MSZoning)

# Test Data
# Mengganti Nilai MSZoning
df_test["MSZoning"] = df_test["MSZoning"].map(map_MSZoning)
```

## **Seleksi Fitur**

Disini saya menggunakan hasil dari multivariat analisis untuk menseleksi fitur yang akan dimasukkan kedalam machine learning, sebelumnya saya juga menggunakan PCA namun ternyata hasilnya menunjukkan error yang lebih tinggi dari pada menggunakan seleksi multivariat analisis.

Fitur yang akan dibuang adalah fitur yang memiliki multivariat analisis kurang dari 1 dan lebih besar dari -1, atau bisa juga dibilang nilai yang mendekati 0.

## **Data Scaling**

Disini Saya Menggunakan Min-Max Scaler untuk menstandarisasi data. sebelumnya saya juga sudah mengetes menggunakan standard scaler namun hasilnya menunjukkan nilai error yang lebih tinggi dari pada Min-Max scaler.

Dari yang saya ketahui alasan kenapa Min-Max scaler lebih baik dalam meningkatkan akurasi adalah karena adanya cukup banyak outlier dalam beberapa kolom fitur. Kenapa tidak saya buang outlier tersebut? alasannya karena outlier tersebut masih memiliki informasi yang baik dan masuk akal, contohnya seperti kolom Lotfrontage. Pada kolom ini menunjukkan data seberapa luas halaman depan yang dimiliki pada rumah tersebut, yang mana data tersebut merupakan data kontinyu, dan tidak memiliki batasan tertentu. 