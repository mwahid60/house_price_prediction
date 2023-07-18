# House Price Prediction

Ini adalah project yang dibuat untuk mengikuti [Kaggle Data Science Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), dan sebagai pembelajaran.

## Keterangan Data
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

Ada beberapa metode yang dapat digunakan, untuk data kategorik disini aku menggunakan

``` Python
df_train["Alley"] = df_train["Alley"].fillna("NA")
```

Pada data ketegorik sebenarnya tidak ada missing, hanya saja pandas 2.0 yang menganggap string **NA** sebagai missing value sehingga hanya perlu dinyatakan ulang sebagai nilai string **NA**.

Sedangkan untuk data kontinyu yang berbentuk *integer* atau *float* disini aku mengisi missing value berdasarkan proporsi distribusi datanya, sebagai berikut:

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

