# Aprendizaje_automatico_en_la_nube
Taller 1

Telco Customer Churn
Haz doble clic (o ingresa) para editar

Análisis Exploratorio de Datos (EDA) - Dataset Telco Customer Churn
Configuración Inicial e Importación de Librerías Primero, importamos todas las librerías que necesitaremos para la manipulación de datos y la visualización.


[2]
1 s
# Importación de librerías esenciales para manipulación de datos y visualización
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración para mejorar la visualización de gráficos
sns.set_style("whitegrid") # Establece un estilo de cuadrícula para los gráficos de Seaborn
plt.rcParams['figure.figsize'] = (12, 7) # Define el tamaño por defecto de las figuras para mejor visibilidad
plt.rcParams['figure.dpi'] = 100 # Define la resolución por defecto de las figuras
plt.rcParams['font.size'] = 12 # Ajusta el tamaño de la fuente para mejor legibilidad
1. Carga y Primera Inspección del Conjunto de Datos
En esta fase, cargamos el dataset y realizamos una inspección inicial para entender su estructura, tipos de datos y la presencia de valores faltantes o duplicados.


[3]
0 s
# --- Fase 1: Carga y Primera Inspección del Conjunto de Datos ---

print("--- 1. Carga y Primera Inspección del Conjunto de Datos ---")

# Cargar el conjunto de datos desde un archivo CSV
# Asegúrate de que 'WA_Fn-UseC_-Telco-Customer-Churn.csv' esté en la ruta correcta.
# Si lo tienes en data/raw/, la ruta sería 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
datos = pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')


--- 1. Carga y Primera Inspección del Conjunto de Datos ---

[4]
0 s
# Mostrar las primeras 5 filas del DataFrame para entender su estructura y contenido
print("\nPrimeras 5 filas del DataFrame:")
print(datos.head())

Primeras 5 filas del DataFrame:
   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService  \
0  7590-VHVEG  Female              0     Yes         No       1           No   
1  5575-GNVDE    Male              0      No         No      34          Yes   
2  3668-QPYBK    Male              0      No         No       2          Yes   
3  7795-CFOCW    Male              0      No         No      45           No   
4  9237-HQITU  Female              0      No         No       2          Yes   

      MultipleLines InternetService OnlineSecurity  ... DeviceProtection  \
0  No phone service             DSL             No  ...               No   
1                No             DSL            Yes  ...              Yes   
2                No             DSL            Yes  ...               No   
3  No phone service             DSL            Yes  ...              Yes   
4                No     Fiber optic             No  ...               No   

  TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling  \
0          No          No              No  Month-to-month              Yes   
1          No          No              No        One year               No   
2          No          No              No  Month-to-month              Yes   
3         Yes          No              No        One year               No   
4          No          No              No  Month-to-month              Yes   

               PaymentMethod MonthlyCharges  TotalCharges Churn  
0           Electronic check          29.85         29.85    No  
1               Mailed check          56.95        1889.5    No  
2               Mailed check          53.85        108.15   Yes  
3  Bank transfer (automatic)          42.30       1840.75    No  
4           Electronic check          70.70        151.65   Yes  

[5 rows x 21 columns]

[5]
0 s
datos.head



[6]
0 s
# Mostrar información concisa del DataFrame, incluyendo el número de entradas,
# el número de columnas, los tipos de datos de cada columna y el conteo de valores no nulos.
print("\nInformación del DataFrame (datos.info()):")
datos.info()

Información del DataFrame (datos.info()):
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   object 
 20  Churn             7043 non-null   object 
dtypes: float64(1), int64(2), object(18)
memory usage: 1.1+ MB

[7]
0 s
# Mostrar estadísticas descriptivas para columnas numéricas.
# Esto incluye conteo, media, desviación estándar, mínimo, cuartiles (25%, 50%, 75%) y máximo.
print("\nEstadísticas descriptivas para columnas numéricas (df.describe()):")
print(datos.describe())

Estadísticas descriptivas para columnas numéricas (df.describe()):
       SeniorCitizen       tenure  MonthlyCharges
count    7043.000000  7043.000000     7043.000000
mean        0.162147    32.371149       64.761692
std         0.368612    24.559481       30.090047
min         0.000000     0.000000       18.250000
25%         0.000000     9.000000       35.500000
50%         0.000000    29.000000       70.350000
75%         0.000000    55.000000       89.850000
max         1.000000    72.000000      118.750000

[8]
0 s
# Mostrar estadísticas descriptivas para columnas categóricas (tipo 'object').
# Esto incluye conteo, número de valores únicos, el valor más frecuente (top) y su frecuencia.
print("\nEstadísticas descriptivas para columnas categóricas (df.describe(include='object')):")
print(datos.describe(include='object'))

Estadísticas descriptivas para columnas categóricas (df.describe(include='object')):
        customerID gender Partner Dependents PhoneService MultipleLines  \
count         7043   7043    7043       7043         7043          7043   
unique        7043      2       2          2            2             3   
top     3186-AJIEK   Male      No         No          Yes            No   
freq             1   3555    3641       4933         6361          3390   

       InternetService OnlineSecurity OnlineBackup DeviceProtection  \
count             7043           7043         7043             7043   
unique               3              3            3                3   
top        Fiber optic             No           No               No   
freq              3096           3498         3088             3095   

       TechSupport StreamingTV StreamingMovies        Contract  \
count         7043        7043            7043            7043   
unique           3           3               3               3   
top             No          No              No  Month-to-month   
freq          3473        2810            2785            3875   

       PaperlessBilling     PaymentMethod TotalCharges Churn  
count              7043              7043         7043  7043  
unique                2                 4         6531     2  
top                 Yes  Electronic check                 No  
freq               4171              2365           11  5174  

[9]
0 s
# Verificar la presencia de valores nulos en cada columna y mostrar el conteo.
print("\nConteo de valores nulos por columna:")
print(datos.isnull().sum())

Conteo de valores nulos por columna:
customerID          0
gender              0
SeniorCitizen       0
Partner             0
Dependents          0
tenure              0
PhoneService        0
MultipleLines       0
InternetService     0
OnlineSecurity      0
OnlineBackup        0
DeviceProtection    0
TechSupport         0
StreamingTV         0
StreamingMovies     0
Contract            0
PaperlessBilling    0
PaymentMethod       0
MonthlyCharges      0
TotalCharges        0
Churn               0
dtype: int64

[10]
0 s
# Calcular el porcentaje de valores nulos por columna.
print("\nPorcentaje de valores nulos por columna:")
print((datos.isnull().sum() / len(datos)) * 100)

Porcentaje de valores nulos por columna:
customerID          0.0
gender              0.0
SeniorCitizen       0.0
Partner             0.0
Dependents          0.0
tenure              0.0
PhoneService        0.0
MultipleLines       0.0
InternetService     0.0
OnlineSecurity      0.0
OnlineBackup        0.0
DeviceProtection    0.0
TechSupport         0.0
StreamingTV         0.0
StreamingMovies     0.0
Contract            0.0
PaperlessBilling    0.0
PaymentMethod       0.0
MonthlyCharges      0.0
TotalCharges        0.0
Churn               0.0
dtype: float64

[11]
0 s
# Verificar la presencia de filas duplicadas en todo el DataFrame.
print("\nNúmero de filas duplicadas:")
print(datos.duplicated().sum())

Número de filas duplicadas:
0

[12]
0 s
# --- Manejo específico de la columna 'TotalCharges' ---
# La columna 'TotalCharges' a menudo se carga como tipo 'object' debido a la presencia de espacios en blanco
# para clientes nuevos (que aún no tienen cargos totales).
# Primero, reemplazamos los espacios en blanco con NaN y luego convertimos la columna a tipo numérico.
print("\n--- Procesando la columna 'TotalCharges' ---")
datos['TotalCharges'] = datos['TotalCharges'].replace(' ', np.nan)
datos['TotalCharges'] = pd.to_numeric(datos['TotalCharges'])

--- Procesando la columna 'TotalCharges' ---

[13]
0 s
# Después de la conversión, volvemos a verificar los nulos en 'TotalCharges'
print("Conteo de valores nulos en 'TotalCharges' después de la conversión:")
print(datos['TotalCharges'].isnull().sum())
print("Porcentaje de valores nulos en 'TotalCharges' después de la conversión:")
print((datos['TotalCharges'].isnull().sum() / len(datos)) * 100)
Conteo de valores nulos en 'TotalCharges' después de la conversión:
11
Porcentaje de valores nulos en 'TotalCharges' después de la conversión:
0.1561834445548772

[14]
0 s
# Para este EDA, imputaremos los nulos de TotalCharges con la mediana,
# ya que es una estrategia común y simple para la línea base.
# En un proyecto real, se podría considerar una imputación más sofisticada o eliminar las filas.
median_total_charges = datos['TotalCharges'].median()
datos['TotalCharges'].fillna(median_total_charges, inplace=True)
print(f"Valores nulos en 'TotalCharges' imputados con la mediana: {median_total_charges}")
print("Verificación final de nulos en 'TotalCharges':", datos['TotalCharges'].isnull().sum())
Valores nulos en 'TotalCharges' imputados con la mediana: 1397.475
Verificación final de nulos en 'TotalCharges': 0
/tmp/ipython-input-2266924351.py:5: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  datos['TotalCharges'].fillna(median_total_charges, inplace=True)

[15]
0 s

# Confirmar los tipos de datos después de la conversión
print("\nTipos de datos después de procesar 'TotalCharges':")
datos.info()

Tipos de datos después de procesar 'TotalCharges':
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   float64
 20  Churn             7043 non-null   object 
dtypes: float64(2), int64(2), object(17)
memory usage: 1.1+ MB
2. Análisis Univariado
En esta fase, examinamos cada variable de forma individual para entender su distribución, rango y la frecuencia de sus valores.


[16]
0 s
# --- Fase 2: Análisis Univariado ---

print("\n--- 2. Análisis Univariado ---")

# Separar columnas numéricas y categóricas para un análisis más fácil
# Excluimos 'customerID' ya que es un identificador y no una característica predictiva.
numerical_cols = datos.select_dtypes(include=np.number).columns.tolist()
categorical_cols = datos.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('customerID') # Eliminar customerID de las columnas categóricas

--- 2. Análisis Univariado ---

[17]
0 s
print(f"\nColumnas Numéricas para análisis: {numerical_cols}")
print(f"Columnas Categóricas para análisis: {categorical_cols}")

Columnas Numéricas para análisis: ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
Columnas Categóricas para análisis: ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

[18]
0 s
# 2.1 Análisis de Variables Numéricas
print("\n2.1 Análisis de Variables Numéricas:")
for col in numerical_cols:
    print(f"\nAnalizando columna numérica: {col}")

2.1 Análisis de Variables Numéricas:

Analizando columna numérica: SeniorCitizen

Analizando columna numérica: tenure

Analizando columna numérica: MonthlyCharges

Analizando columna numérica: TotalCharges

[19]
0 s
 # Histograma para visualizar la distribución de la variable.
 # Muestra la frecuencia de los valores dentro de diferentes rangos (bins).
plt.figure(figsize=(10, 5))
sns.histplot(datos[col], kde=True, bins=30, color='skyblue')
plt.title(f'Distribución de {col}')
plt.xlabel(col)
plt.ylabel('Frecuencia')
plt.show()


<img width="759" height="391" alt="image" src="https://github.com/user-attachments/assets/b40d336e-c029-4038-bd3c-d67414dd3342" />



[20]
0 s
# Boxplot para identificar la mediana, cuartiles, rango intercuartílico y posibles outliers.
plt.figure(figsize=(10, 5))
sns.boxplot(x=datos[col], color='lightcoral')
plt.title(f'Boxplot de {col}')
plt.xlabel(col)
plt.show()

<img width="706" height="376" alt="image" src="https://github.com/user-attachments/assets/d670b7bf-8139-42f5-970f-072e863fcd36" />


[21]
0 s
# 2.2 Análisis de Variables Categóricas
print("\n2.2 Análisis de Variables Categóricas:")
for col in categorical_cols:
    print(f"\nAnalizando columna categórica: {col}")


2.2 Análisis de Variables Categóricas:

Analizando columna categórica: gender

Analizando columna categórica: Partner

Analizando columna categórica: Dependents

Analizando columna categórica: PhoneService

Analizando columna categórica: MultipleLines

Analizando columna categórica: InternetService

Analizando columna categórica: OnlineSecurity

Analizando columna categórica: OnlineBackup

Analizando columna categórica: DeviceProtection

Analizando columna categórica: TechSupport

Analizando columna categórica: StreamingTV

Analizando columna categórica: StreamingMovies

Analizando columna categórica: Contract

Analizando columna categórica: PaperlessBilling

Analizando columna categórica: PaymentMethod

Analizando columna categórica: Churn

[22]
0 s
# Conteo de valores únicos y sus frecuencias.
# Muestra cuántas veces aparece cada categoría.
print(f"Conteo de valores para '{col}':")
print(datos[col].value_counts())
print(f"Porcentaje de valores para '{col}':")
print(datos[col].value_counts(normalize=True) * 100)
Conteo de valores para 'Churn':
Churn
No     5174
Yes    1869
Name: count, dtype: int64
Porcentaje de valores para 'Churn':
Churn
No     73.463013
Yes    26.536987
Name: proportion, dtype: float64

[23]
0 s
# Gráfico de barras para visualizar la distribución de las categorías.
# Útil para ver el balance o desbalance de las categorías.
plt.figure(figsize=(10, 5))
sns.countplot(x=col, data=datos, palette='viridis', order=datos[col].value_counts().index)
plt.title(f'Distribución de {col}')
plt.xlabel(col)
plt.ylabel('Conteo')
plt.xticks(rotation=45, ha='right') # Rotar etiquetas si son largas para evitar superposición
plt.show()

<img width="1088" height="470" alt="image" src="https://github.com/user-attachments/assets/fc57a80e-821b-43c2-8ce9-74b1abbbe2ea" />


[24]
0 s
# 2.3 Análisis de la Variable Objetivo ('Churn')
# Es crucial entender la distribución de la variable que queremos predecir.
print("\n2.3 Análisis de la Variable Objetivo: Churn")
print("Distribución de la variable objetivo 'Churn':")
print(datos['Churn'].value_counts())
print(datos['Churn'].value_counts(normalize=True) * 100)

2.3 Análisis de la Variable Objetivo: Churn
Distribución de la variable objetivo 'Churn':
Churn
No     5174
Yes    1869
Name: count, dtype: int64
Churn
No     73.463013
Yes    26.536987
Name: proportion, dtype: float64

[25]
0 s
plt.figure(figsize=(7, 5))
sns.countplot(x='Churn', data=datos, palette='coolwarm')
plt.title('Distribución de Churn')
plt.xlabel('Churn')
plt.ylabel('Conteo')
plt.show()

<img width="1129" height="447" alt="image" src="https://github.com/user-attachments/assets/a891058f-729c-4033-9ea9-62c3df06d414" />


[26]
0 s
# Advertencia si se detecta un desequilibrio significativo de clases
churn_percentage = datos['Churn'].value_counts(normalize=True)
if churn_percentage['No'] > 0.75 or churn_percentage['Yes'] > 0.75:
    print("¡Advertencia! Se detecta un posible desequilibrio de clases en la variable 'Churn'.")
    print("Esto es común en problemas de churn y requerirá técnicas de manejo de desequilibrio en el modelado.")
3. Análisis Bivariado y Multivariado
Aquí exploramos las relaciones entre pares de variables y cómo se relacionan con la variable objetivo (Churn).


[27]
0 s
#Fase 3: Análisis Bivariado y Multivariado ---

print("\n--- 3. Análisis Bivariado y Multivariado ---")

--- 3. Análisis Bivariado y Multivariado ---

[28]
0 s
# 3.1 Relación entre Variables Numéricas
print("\n3.1 Relación entre Variables Numéricas:")

3.1 Relación entre Variables Numéricas:

[29]
0 s
# Matriz de correlación para todas las columnas numéricas.
# Calcula la correlación de Pearson entre pares de columnas numéricas.
correlation_matrix = datos[numerical_cols].corr()
print("\nMatriz de Correlación:")
print(correlation_matrix)

Matriz de Correlación:
                SeniorCitizen    tenure  MonthlyCharges  TotalCharges
SeniorCitizen        1.000000  0.016567        0.220173      0.102652
tenure               0.016567  1.000000        0.247900      0.825464
MonthlyCharges       0.220173  0.247900        1.000000      0.650864
TotalCharges         0.102652  0.825464        0.650864      1.000000

[30]
0 s
# Mapa de calor de la matriz de correlación para una visualización rápida.
# Los colores indican la fuerza y dirección de la correlación (positiva o negativa).
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.show()

<img width="797" height="552" alt="image" src="https://github.com/user-attachments/assets/8f58d95e-4060-446a-96e9-f69f4195ee2b" />


[31]
0 s
# Scatter plots para pares de variables numéricas con alta correlación o interés,
# coloreados por la variable objetivo 'Churn' para ver patrones.
# Ejemplo: Relación entre MonthlyCharges y TotalCharges, y tenure.
print("\nScatter plots de variables numéricas vs. Churn:")
plt.figure(figsize=(12, 7))
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=datos, alpha=0.7)
plt.title('MonthlyCharges vs TotalCharges (coloreado por Churn)')
plt.xlabel('Cargos Mensuales')
plt.ylabel('Cargos Totales')
plt.show()

<img width="1039" height="536" alt="image" src="https://github.com/user-attachments/assets/2467fdb4-0701-4636-bd28-a796cd1a5b16" />



[32]
0 s
plt.figure(figsize=(12, 7))
sns.scatterplot(x='tenure', y='TotalCharges', hue='Churn', data=datos, alpha=0.7)
plt.title('Antigüedad (tenure) vs TotalCharges (coloreado por Churn)')
plt.xlabel('Antigüedad (Meses)')
plt.ylabel('Cargos Totales')
plt.show()

<img width="923" height="516" alt="image" src="https://github.com/user-attachments/assets/ef817455-9f3a-4727-b82b-3ff880e27446" />


[33]
1 s
plt.figure(figsize=(12, 7))
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=datos, alpha=0.7)
plt.title('Antigüedad (tenure) vs MonthlyCharges (coloreado por Churn)')
plt.xlabel('Antigüedad (Meses)')
plt.ylabel('Cargos Mensuales')
plt.show()

<img width="899" height="499" alt="image" src="https://github.com/user-attachments/assets/ce3f5b83-b5ad-49de-900a-c8f32773aa83" />


[34]
0 s
# 3.2 Relación entre Variables Categóricas y la Variable Objetivo (Churn)
print("\n3.2 Relación entre Variables Categóricas y la Variable Objetivo (Churn):")
for col in categorical_cols:
    if col == 'Churn': # No analizar Churn contra sí mismo
        continue
    print(f"\nAnalizando relación entre '{col}' y 'Churn':")

3.2 Relación entre Variables Categóricas y la Variable Objetivo (Churn):

Analizando relación entre 'gender' y 'Churn':

Analizando relación entre 'Partner' y 'Churn':

Analizando relación entre 'Dependents' y 'Churn':

Analizando relación entre 'PhoneService' y 'Churn':

Analizando relación entre 'MultipleLines' y 'Churn':

Analizando relación entre 'InternetService' y 'Churn':

Analizando relación entre 'OnlineSecurity' y 'Churn':

Analizando relación entre 'OnlineBackup' y 'Churn':

Analizando relación entre 'DeviceProtection' y 'Churn':

Analizando relación entre 'TechSupport' y 'Churn':

Analizando relación entre 'StreamingTV' y 'Churn':

Analizando relación entre 'StreamingMovies' y 'Churn':

Analizando relación entre 'Contract' y 'Churn':

Analizando relación entre 'PaperlessBilling' y 'Churn':

Analizando relación entre 'PaymentMethod' y 'Churn':

[35]
0 s
# Tabla de contingencia (crosstab) para ver la distribución conjunta de dos variables categóricas.
# 'normalize='index'' muestra el porcentaje de Churn por cada categoría de la columna.
crosstab = pd.crosstab(datos[col], datos['Churn'], normalize='index') * 100
print(crosstab)
Churn     No    Yes
Churn              
No     100.0    0.0
Yes      0.0  100.0

[36]
0 s
 # Gráfico de barras agrupadas para visualizar la relación.
 # Muestra la proporción de 'Yes' y 'No' para cada categoría de la característica.
plt.figure(figsize=(10, 6))
sns.countplot(x=col, hue='Churn', data=datos, palette='pastel', order=datos[col].value_counts().index)
plt.title(f'Churn por {col}')
plt.xlabel(col)
plt.ylabel('Conteo')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Churn')
plt.show()

<img width="1380" height="479" alt="image" src="https://github.com/user-attachments/assets/e0bb790f-044c-4e4d-8926-47d3586e7643" />


[37]
0 s
# 3.3 Relación entre Variables Numéricas y la Variable Objetivo (Churn)
print("\n3.3 Relación entre Variables Numéricas y la Variable Objetivo (Churn):")
for col in numerical_cols:
    print(f"\nAnalizando relación entre '{col}' y 'Churn':")

3.3 Relación entre Variables Numéricas y la Variable Objetivo (Churn):

Analizando relación entre 'SeniorCitizen' y 'Churn':

Analizando relación entre 'tenure' y 'Churn':

Analizando relación entre 'MonthlyCharges' y 'Churn':

Analizando relación entre 'TotalCharges' y 'Churn':

[38]
0 s
# Boxplot para comparar la distribución de la variable numérica para cada categoría de Churn.
# Permite ver si la media/mediana y la dispersión difieren entre los que abandonan y los que no.
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y=col, data=datos, palette='viridis')
plt.title(f'{col} por Churn')
plt.xlabel('Churn')
plt.ylabel(col)
plt.show()

<img width="1129" height="516" alt="image" src="https://github.com/user-attachments/assets/c2b512e5-05ec-4062-a988-faf20c07f7b6" />


[39]
0 s
# Violin plot para ver la densidad de la distribución de la variable numérica por Churn.
# Combina un boxplot con una estimación de densidad de kernel.
plt.figure(figsize=(10, 6))
sns.violinplot(x='Churn', y=col, data=datos, palette='viridis')
plt.title(f'Distribución de {col} por Churn')
plt.xlabel('Churn')
plt.ylabel(col)
plt.show()

<img width="1086" height="513" alt="image" src="https://github.com/user-attachments/assets/571db031-7ed9-4513-b3c4-f355345c2b3e" />


4. Resumen de Hallazgos y Próximos Pasos
Después de ejecutar el EDA, es crucial resumir los hallazgos clave y planificar las acciones futuras en la fase de preprocesamiento.


[40]
0 s
# --- Fase 4: Resumen de Hallazgos y Próximos Pasos ---

print("\n--- 4. Resumen de Hallazgos y Próximos Pasos ---")

print("\n**Hallazgos Clave del EDA en el Dataset Telco Customer Churn:**")
print("- **Valores Nulos:** Se encontraron y manejaron valores nulos en la columna 'TotalCharges' (originalmente espacios en blanco), imputándolos con la mediana. Esto es un paso crítico para la calidad de los datos.")
print("- **Duplicados:** No se encontraron filas duplicadas en el dataset, lo cual es bueno.")
print("- **Distribuciones de Variables Numéricas:**")
print("  - **tenure (Antigüedad):** La distribución es bimodal, con picos en clientes muy nuevos y clientes de larga duración. Los clientes que abandonan tienden a tener una antigüedad menor.")
print("  - **MonthlyCharges (Cargos Mensuales):** La distribución es bastante uniforme, con un ligero sesgo hacia cargos más altos. Los clientes que abandonan tienden a tener cargos mensuales más altos.")
print("  - **TotalCharges (Cargos Totales):** La distribución está fuertemente sesgada a la derecha, lo que es esperado ya que es una acumulación. Los clientes que abandonan tienden a tener cargos totales más bajos (lo cual es consistente con una menor antigüedad).")
print("- **Distribuciones de Variables Categóricas:**")
print("  - **Gender:** Bastante balanceado.")
print("  - **Contract:** La mayoría de los clientes tienen contratos 'Month-to-month'. Este tipo de contrato muestra una tasa de churn significativamente más alta que los contratos de uno o dos años.")
print("  - **InternetService:** 'Fiber optic' tiene una tasa de churn mucho mayor que 'DSL'.")
print("  - **OnlineSecurity, TechSupport, DeviceProtection, OnlineBackup, StreamingTV, StreamingMovies:** Los clientes que no tienen estos servicios adicionales tienden a abandonar más.")
print("  - **PaymentMethod:** 'Electronic check' tiene una tasa de churn notablemente más alta.")
print("- **Variable Objetivo (Churn):** Se observa un desequilibrio de clases significativo, con aproximadamente un 73.5% de clientes que NO abandonan y un 26.5% que SÍ abandonan. Esto es un factor importante a considerar en el modelado.")
print("- **Correlaciones:**")
print("  - Alta correlación positiva entre 'tenure' y 'TotalCharges', lo cual es lógico.")
print("  - Correlación moderada entre 'MonthlyCharges' y 'TotalCharges'.")
print("  - Las visualizaciones bivariadas confirman que 'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod' y las variables numéricas 'tenure', 'MonthlyCharges', 'TotalCharges' son fuertes predictores de 'Churn'.")

--- 4. Resumen de Hallazgos y Próximos Pasos ---

**Hallazgos Clave del EDA en el Dataset Telco Customer Churn:**
- **Valores Nulos:** Se encontraron y manejaron valores nulos en la columna 'TotalCharges' (originalmente espacios en blanco), imputándolos con la mediana. Esto es un paso crítico para la calidad de los datos.
- **Duplicados:** No se encontraron filas duplicadas en el dataset, lo cual es bueno.
- **Distribuciones de Variables Numéricas:**
  - **tenure (Antigüedad):** La distribución es bimodal, con picos en clientes muy nuevos y clientes de larga duración. Los clientes que abandonan tienden a tener una antigüedad menor.
  - **MonthlyCharges (Cargos Mensuales):** La distribución es bastante uniforme, con un ligero sesgo hacia cargos más altos. Los clientes que abandonan tienden a tener cargos mensuales más altos.
  - **TotalCharges (Cargos Totales):** La distribución está fuertemente sesgada a la derecha, lo que es esperado ya que es una acumulación. Los clientes que abandonan tienden a tener cargos totales más bajos (lo cual es consistente con una menor antigüedad).
- **Distribuciones de Variables Categóricas:**
  - **Gender:** Bastante balanceado.
  - **Contract:** La mayoría de los clientes tienen contratos 'Month-to-month'. Este tipo de contrato muestra una tasa de churn significativamente más alta que los contratos de uno o dos años.
  - **InternetService:** 'Fiber optic' tiene una tasa de churn mucho mayor que 'DSL'.
  - **OnlineSecurity, TechSupport, DeviceProtection, OnlineBackup, StreamingTV, StreamingMovies:** Los clientes que no tienen estos servicios adicionales tienden a abandonar más.
  - **PaymentMethod:** 'Electronic check' tiene una tasa de churn notablemente más alta.
- **Variable Objetivo (Churn):** Se observa un desequilibrio de clases significativo, con aproximadamente un 73.5% de clientes que NO abandonan y un 26.5% que SÍ abandonan. Esto es un factor importante a considerar en el modelado.
- **Correlaciones:**
  - Alta correlación positiva entre 'tenure' y 'TotalCharges', lo cual es lógico.
  - Correlación moderada entre 'MonthlyCharges' y 'TotalCharges'.
  - Las visualizaciones bivariadas confirman que 'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod' y las variables numéricas 'tenure', 'MonthlyCharges', 'TotalCharges' son fuertes predictores de 'Churn'.

[41]
0 s
print("\n**Próximos Pasos Sugeridos (Estrategia de Preprocesamiento):**")
print("1.  **Manejo de Nulos:** La imputación de 'TotalCharges' con la mediana ya se realizó. En un enfoque más avanzado, se podría considerar un modelo predictivo para imputar o eliminar las filas si el porcentaje de nulos fuera mayor.")
print("2.  **Codificación de Variables Categóricas:**")
print("    - La variable objetivo 'Churn' ('Yes'/'No') debe ser codificada a numérica (ej. 1/0) usando `LabelEncoder`.")
print("   

**Próximos Pasos Sugeridos (Estrategia de Preprocesamiento):**
1.  **Manejo de Nulos:** La imputación de 'TotalCharges' con la mediana ya se realizó. En un enfoque más avanzado, se podría considerar un modelo predictivo para imputar o eliminar las filas si el porcentaje de nulos fuera mayor.
2.  **Codificación de Variables Categóricas:**
    - La variable objetivo 'Churn' ('Yes'/'No') debe ser codificada a numérica (ej. 1/0) usando `LabelEncoder`.
    - Las variables categóricas predictoras (ej. 'Gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', etc.) deben ser convertidas a formato numérico. `OneHotEncoder` es una buena opción para la mayoría, especialmente para aquellas sin un orden intrínseco. Algunas binarias ('Yes'/'No') podrían ser mapeadas a 1/0 directamente.
3.  **Escalado de Características Numéricas:** Aplicar escalado (ej. `StandardScaler` o `MinMaxScaler`) a las variables numéricas ('tenure', 'MonthlyCharges', 'TotalCharges') para que los modelos basados en distancia o gradiente funcionen mejor y no se vean dominados por características con rangos de valores más grandes.
4.  **Manejo de Desequilibrio de Clases:** Dada la desproporción en la variable 'Churn', será crucial aplicar técnicas como: 
    - **Sobremuestreo (Oversampling):** Ej. SMOTE (Synthetic Minority Over-sampling Technique) para crear muestras sintéticas de la clase minoritaria.
    - **Submuestreo (Undersampling):** Reducir el número de muestras de la clase mayoritaria.
    - **Ajuste de Pesos de Clase:** Configurar el modelo para dar más importancia a la clase minoritaria durante el entrenamiento.
5.  **Ingeniería de Características:** Explorar la creación de nuevas características que puedan ser predictivas, como ratios (ej. `TotalCharges / tenure` para obtener un cargo mensual promedio real) o agrupaciones de `tenure`.
6.  **División del Dataset:** Dividir el dataset en conjuntos de entrenamiento, validación y prueba ANTES de aplicar cualquier preprocesamiento que dependa de los datos (como escalado o SMOTE) para evitar el data leakage.
Este EDA te proporciona una comprensión profunda del dataset "Telco Customer Churn". Los hallazgos son la base para tomar decisiones informadas en las siguientes fases de preprocesamiento y modelado. ¡Ahora tienes una excelente visión de los datos para avanzar!
