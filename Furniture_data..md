```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression
import numpy as np
```

# Loading Furniture Sales Data into a Pandas DataFrame


```python
Furniture_data = pd.read_csv(r'C:\Users\Home\Desktop\Important\Portfolio\Furniture.csv')
Furniture_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>cost</th>
      <th>sales</th>
      <th>profit_margin</th>
      <th>inventory</th>
      <th>discount_percentage</th>
      <th>delivery_days</th>
      <th>category</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>brand</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>218.543053</td>
      <td>181.610932</td>
      <td>40</td>
      <td>16.899243</td>
      <td>105</td>
      <td>27.796433</td>
      <td>9</td>
      <td>Bed</td>
      <td>Plastic</td>
      <td>Red</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>BrandA</td>
      <td>3949.165238</td>
    </tr>
    <tr>
      <th>1</th>
      <td>477.821438</td>
      <td>385.033827</td>
      <td>7</td>
      <td>19.418888</td>
      <td>192</td>
      <td>26.943715</td>
      <td>6</td>
      <td>Chair</td>
      <td>Glass</td>
      <td>Blue</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>-3521.002258</td>
    </tr>
    <tr>
      <th>2</th>
      <td>379.397274</td>
      <td>276.736765</td>
      <td>32</td>
      <td>27.058842</td>
      <td>59</td>
      <td>21.948130</td>
      <td>2</td>
      <td>Table</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>14285.560219</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319.396318</td>
      <td>281.841334</td>
      <td>48</td>
      <td>11.758114</td>
      <td>45</td>
      <td>11.009944</td>
      <td>2</td>
      <td>Table</td>
      <td>Glass</td>
      <td>Green</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>BrandD</td>
      <td>12261.073703</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.208388</td>
      <td>69.743681</td>
      <td>19</td>
      <td>41.981019</td>
      <td>35</td>
      <td>3.183763</td>
      <td>9</td>
      <td>Chair</td>
      <td>Glass</td>
      <td>Brown</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>-4588.255733</td>
    </tr>
  </tbody>
</table>
</div>



# Wrangling Data

### Data Formating


```python
Furniture_data.columns = Furniture_data.columns.str.strip()
```


```python
Furniture_data.dtypes
```




    price                  float64
    cost                   float64
    sales                    int64
    profit_margin          float64
    inventory                int64
    discount_percentage    float64
    delivery_days            int64
    category                object
    material                object
    color                   object
    location                object
    season                  object
    store_type              object
    brand                   object
    revenue                float64
    dtype: object



Conclusion: "The data types for each column in the Furniture_data DataFrame have been successfully retrieved. This shows that the dataset contains a mix of data types, including integers, floats, and possibly categorical or string data (object). Based on these data types, further analysis or processing can be tailored, such as numerical operations on numeric columns or encoding for categorical columns if needed.

### Check for Null Values


```python
Furniture_data.isnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>cost</th>
      <th>sales</th>
      <th>profit_margin</th>
      <th>inventory</th>
      <th>discount_percentage</th>
      <th>delivery_days</th>
      <th>category</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>brand</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2495</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2496</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2497</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2498</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2499</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2500 rows × 15 columns</p>
</div>




```python
# Check for the presence of null values in the dataset
Furniture_data.isnull().sum() 
```




    price                  0
    cost                   0
    sales                  0
    profit_margin          0
    inventory              0
    discount_percentage    0
    delivery_days          0
    category               0
    material               0
    color                  0
    location               0
    season                 0
    store_type             0
    brand                  0
    revenue                0
    dtype: int64



Conclusion: "This shows that the dataset contains no null values in any column, as confirmed by running the code above."

# DATA BINS


```python
# Binning 'sales' into 3 equal-sized bins
Furniture_data['sales_Binned_Quantile'] = pd.qcut(Furniture_data['sales'], q=3, labels=['Low', 'Medium', 'High'])
```


```python
Furniture_data['Price_Binned_Quantile'] = pd.qcut(Furniture_data['price'], q=3, labels=['Low', 'Medium', 'High'])
```


```python
Furniture_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>cost</th>
      <th>sales</th>
      <th>profit_margin</th>
      <th>inventory</th>
      <th>discount_percentage</th>
      <th>delivery_days</th>
      <th>category</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>brand</th>
      <th>revenue</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>218.543053</td>
      <td>181.610932</td>
      <td>40</td>
      <td>16.899243</td>
      <td>105</td>
      <td>27.796433</td>
      <td>9</td>
      <td>Bed</td>
      <td>Plastic</td>
      <td>Red</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>BrandA</td>
      <td>3949.165238</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>1</th>
      <td>477.821438</td>
      <td>385.033827</td>
      <td>7</td>
      <td>19.418888</td>
      <td>192</td>
      <td>26.943715</td>
      <td>6</td>
      <td>Chair</td>
      <td>Glass</td>
      <td>Blue</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>-3521.002258</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>2</th>
      <td>379.397274</td>
      <td>276.736765</td>
      <td>32</td>
      <td>27.058842</td>
      <td>59</td>
      <td>21.948130</td>
      <td>2</td>
      <td>Table</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>14285.560219</td>
      <td>Medium</td>
      <td>High</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319.396318</td>
      <td>281.841334</td>
      <td>48</td>
      <td>11.758114</td>
      <td>45</td>
      <td>11.009944</td>
      <td>2</td>
      <td>Table</td>
      <td>Glass</td>
      <td>Green</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>BrandD</td>
      <td>12261.073703</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.208388</td>
      <td>69.743681</td>
      <td>19</td>
      <td>41.981019</td>
      <td>35</td>
      <td>3.183763</td>
      <td>9</td>
      <td>Chair</td>
      <td>Glass</td>
      <td>Brown</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>-4588.255733</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
  </tbody>
</table>
</div>



Conclusion:
By running the above code,2 columns were added to the Furniture_data DataFrame. This column, Price_Binned_Quantile and sales_Binned_Quantile, will contain the bin labels corresponding to the Price values and sales values respectively.

# Dummy variables


```python
# Create dummy variables for the 'Category' column
Furniture_data_dummies = pd.get_dummies(Furniture_data, columns=['category', 'store_type'])
```


```python
# Display the DataFrame with dummy variables
Furniture_data_dummies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>cost</th>
      <th>sales</th>
      <th>profit_margin</th>
      <th>inventory</th>
      <th>discount_percentage</th>
      <th>delivery_days</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>...</th>
      <th>revenue</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>category_Bed</th>
      <th>category_Chair</th>
      <th>category_Desk</th>
      <th>category_Sofa</th>
      <th>category_Table</th>
      <th>store_type_Online</th>
      <th>store_type_Retail</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>218.543053</td>
      <td>181.610932</td>
      <td>40</td>
      <td>16.899243</td>
      <td>105</td>
      <td>27.796433</td>
      <td>9</td>
      <td>Plastic</td>
      <td>Red</td>
      <td>Rural</td>
      <td>...</td>
      <td>3949.165238</td>
      <td>High</td>
      <td>Medium</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>477.821438</td>
      <td>385.033827</td>
      <td>7</td>
      <td>19.418888</td>
      <td>192</td>
      <td>26.943715</td>
      <td>6</td>
      <td>Glass</td>
      <td>Blue</td>
      <td>Rural</td>
      <td>...</td>
      <td>-3521.002258</td>
      <td>Low</td>
      <td>High</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>379.397274</td>
      <td>276.736765</td>
      <td>32</td>
      <td>27.058842</td>
      <td>59</td>
      <td>21.948130</td>
      <td>2</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Suburban</td>
      <td>...</td>
      <td>14285.560219</td>
      <td>Medium</td>
      <td>High</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319.396318</td>
      <td>281.841334</td>
      <td>48</td>
      <td>11.758114</td>
      <td>45</td>
      <td>11.009944</td>
      <td>2</td>
      <td>Glass</td>
      <td>Green</td>
      <td>Rural</td>
      <td>...</td>
      <td>12261.073703</td>
      <td>High</td>
      <td>Medium</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.208388</td>
      <td>69.743681</td>
      <td>19</td>
      <td>41.981019</td>
      <td>35</td>
      <td>3.183763</td>
      <td>9</td>
      <td>Glass</td>
      <td>Brown</td>
      <td>Rural</td>
      <td>...</td>
      <td>-4588.255733</td>
      <td>Medium</td>
      <td>Low</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



To ensure that the categorical data in the dataset is properly handled for analysis and modeling, we transformed certain categorical columns into dummy variables. Dummy variables represent each unique category in a separate column using binary values — true for presence and false for absence. This allows algorithms to interpret categorical information effectively

# Normalization


```python
# Z-score normalization on the 'price' column
Furniture_Norm = (Furniture_data['price'] - Furniture_data['price'].mean())/ Furniture_data['price'].std()
```


```python
Furniture_Norm.head()
```




    0   -0.427447
    1    1.553313
    2    0.801400
    3    0.343022
    4   -1.178676
    Name: price, dtype: float64




```python
# Z-score normalization on the 'revenue' column
Furniture_Norm2 = (Furniture_data['revenue'] - Furniture_data['revenue'].mean())/ Furniture_data['revenue'].std()
```


```python
Furniture_Norm2.head()
```




    0   -0.286508
    1   -1.368711
    2    1.210926
    3    0.917639
    4   -1.523325
    Name: revenue, dtype: float64




```python
Furniture_Norm_Data = pd.DataFrame()
```


```python
Furniture_Norm_Data['revenue_norm'] = Furniture_Norm2

# Display the updated DataFrame
Furniture_Norm_Data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.286508</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.368711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.210926</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.917639</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.523325</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add the normalized revenue and price as a new column in the original DataFrame
Furniture_Norm_Data['price_norm'] = Furniture_Norm

# Display the updated DataFrame
Furniture_Norm_Data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue_norm</th>
      <th>price_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.286508</td>
      <td>-0.427447</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.368711</td>
      <td>1.553313</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.210926</td>
      <td>0.801400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.917639</td>
      <td>0.343022</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.523325</td>
      <td>-1.178676</td>
    </tr>
  </tbody>
</table>
</div>



Conclusion:
In this analysis, we applied normalization to the revenue and price data to standardize the values and ensure they are more comparable across all records. By transforming the revenue and price data into Z-scores, each value is now represented in terms of how many standard deviations it is from the mean. This ensures that:

The mean of the normalized data is 0, and the standard deviation is 1.
We can more easily compare different data points, identifying outliers (values far from the mean) and making the data easier to interpret for statistical modeling.

# Explorating Data Analysis (EDA)


```python
Furniture_data2 = Furniture_data.drop(columns=['profit_margin', 'inventory','delivery_days'])
```


```python
Furniture_data2.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>cost</th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>category</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>brand</th>
      <th>revenue</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2500.000000</td>
      <td>2500.000000</td>
      <td>2500.000000</td>
      <td>2500.000000</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500.000000</td>
      <td>2500</td>
      <td>2500</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>5</td>
      <td>6</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>NaN</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Table</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandA</td>
      <td>NaN</td>
      <td>Low</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>533</td>
      <td>529</td>
      <td>448</td>
      <td>897</td>
      <td>651</td>
      <td>1307</td>
      <td>650</td>
      <td>NaN</td>
      <td>867</td>
      <td>834</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>274.495219</td>
      <td>191.930107</td>
      <td>24.924000</td>
      <td>14.947616</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5926.853657</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>130.898452</td>
      <td>98.590751</td>
      <td>14.050067</td>
      <td>8.621547</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6902.737604</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>50.704297</td>
      <td>26.505895</td>
      <td>1.000000</td>
      <td>0.005556</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-14214.565505</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>159.104890</td>
      <td>106.399135</td>
      <td>13.000000</td>
      <td>7.760214</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1216.719195</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>277.641809</td>
      <td>189.336329</td>
      <td>25.000000</td>
      <td>14.915143</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5523.232714</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>387.378046</td>
      <td>263.188280</td>
      <td>37.000000</td>
      <td>22.292661</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10233.537982</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>499.872953</td>
      <td>447.022911</td>
      <td>49.000000</td>
      <td>29.991229</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32922.078832</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Creating a new DataFrame 'Furniture_EMD' with selected columns from 'Furniture_data2'
Furniture_EMD = Furniture_data2[['price','cost','sales','discount_percentage','material','color',
                                 'location', 'season', 'store_type', 'brand', 'revenue',
                                 'sales_Binned_Quantile', 'Price_Binned_Quantile' ]]
```


```python
Furniture_EMD.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>cost</th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>brand</th>
      <th>revenue</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>218.543053</td>
      <td>181.610932</td>
      <td>40</td>
      <td>27.796433</td>
      <td>Plastic</td>
      <td>Red</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>BrandA</td>
      <td>3949.165238</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>1</th>
      <td>477.821438</td>
      <td>385.033827</td>
      <td>7</td>
      <td>26.943715</td>
      <td>Glass</td>
      <td>Blue</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>-3521.002258</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>2</th>
      <td>379.397274</td>
      <td>276.736765</td>
      <td>32</td>
      <td>21.948130</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>14285.560219</td>
      <td>Medium</td>
      <td>High</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319.396318</td>
      <td>281.841334</td>
      <td>48</td>
      <td>11.009944</td>
      <td>Glass</td>
      <td>Green</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>BrandD</td>
      <td>12261.073703</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.208388</td>
      <td>69.743681</td>
      <td>19</td>
      <td>3.183763</td>
      <td>Glass</td>
      <td>Brown</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>-4588.255733</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
  </tbody>
</table>
</div>



## Calculating Cost on discount and Profit on each sales of furniture


```python
Furniture_EMD['cost_on_discount'] = (Furniture_EMD['price'] * Furniture_data2['sales']) * (Furniture_EMD['discount_percentage'] 
                                                                                           /100)

```

This calculation finds the cost after discount for each furniture item by multiplying the unit price by the number of units sold and adjusting for the discount percentage. This provides an accurate representation of the actual cost paid after discounts are applied, which is crucial for understanding the impact of sales promotions on revenue.


```python
Furniture_EMD['Total_Profit'] = (Furniture_EMD['price'] * Furniture_data2['sales']) - (Furniture_EMD['cost_on_discount'])
```

The Total Profit is calculated by subtracting the discounted cost (cost_on_discount) from the total revenue (calculated as price multiplied by sales). This gives a clear indication of how much profit was made on each item, taking into account the discounts offered. This metric is critical for evaluating the financial performance of different products in the furniture line


```python
Furniture_EMD.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>cost</th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>brand</th>
      <th>revenue</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>218.543053</td>
      <td>181.610932</td>
      <td>40</td>
      <td>27.796433</td>
      <td>Plastic</td>
      <td>Red</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>BrandA</td>
      <td>3949.165238</td>
      <td>High</td>
      <td>Medium</td>
      <td>2429.886974</td>
      <td>6311.835165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>477.821438</td>
      <td>385.033827</td>
      <td>7</td>
      <td>26.943715</td>
      <td>Glass</td>
      <td>Blue</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>-3521.002258</td>
      <td>Low</td>
      <td>High</td>
      <td>901.199926</td>
      <td>2443.550139</td>
    </tr>
    <tr>
      <th>2</th>
      <td>379.397274</td>
      <td>276.736765</td>
      <td>32</td>
      <td>21.948130</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>14285.560219</td>
      <td>Medium</td>
      <td>High</td>
      <td>2664.659446</td>
      <td>9476.053316</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319.396318</td>
      <td>281.841334</td>
      <td>48</td>
      <td>11.009944</td>
      <td>Glass</td>
      <td>Green</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>BrandD</td>
      <td>12261.073703</td>
      <td>High</td>
      <td>Medium</td>
      <td>1687.937139</td>
      <td>13643.086119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.208388</td>
      <td>69.743681</td>
      <td>19</td>
      <td>3.183763</td>
      <td>Glass</td>
      <td>Brown</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>-4588.255733</td>
      <td>Medium</td>
      <td>Low</td>
      <td>72.715843</td>
      <td>2211.243532</td>
    </tr>
  </tbody>
</table>
</div>



## Calculation of Revenue and Expenditure for Furniture Sales Performance


```python
# Calculate Expenditure as the cost multiplied by the number of units sold
Furniture_EMD['Expenditure'] = (Furniture_EMD['cost'] * Furniture_data2['sales']) 
```


```python
# Calculate Revenue by subtracting Expenditure from Total Profit
Furniture_EMD['Revenue'] = (Furniture_EMD['Total_Profit'] - Furniture_EMD['Expenditure']) 
```


```python
Furniture_EMD.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>cost</th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>brand</th>
      <th>revenue</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>218.543053</td>
      <td>181.610932</td>
      <td>40</td>
      <td>27.796433</td>
      <td>Plastic</td>
      <td>Red</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>BrandA</td>
      <td>3949.165238</td>
      <td>High</td>
      <td>Medium</td>
      <td>2429.886974</td>
      <td>6311.835165</td>
      <td>7264.437262</td>
      <td>-952.602098</td>
    </tr>
    <tr>
      <th>1</th>
      <td>477.821438</td>
      <td>385.033827</td>
      <td>7</td>
      <td>26.943715</td>
      <td>Glass</td>
      <td>Blue</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>-3521.002258</td>
      <td>Low</td>
      <td>High</td>
      <td>901.199926</td>
      <td>2443.550139</td>
      <td>2695.236789</td>
      <td>-251.686650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>379.397274</td>
      <td>276.736765</td>
      <td>32</td>
      <td>21.948130</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>14285.560219</td>
      <td>Medium</td>
      <td>High</td>
      <td>2664.659446</td>
      <td>9476.053316</td>
      <td>8855.576485</td>
      <td>620.476831</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319.396318</td>
      <td>281.841334</td>
      <td>48</td>
      <td>11.009944</td>
      <td>Glass</td>
      <td>Green</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>BrandD</td>
      <td>12261.073703</td>
      <td>High</td>
      <td>Medium</td>
      <td>1687.937139</td>
      <td>13643.086119</td>
      <td>13528.384027</td>
      <td>114.702092</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.208388</td>
      <td>69.743681</td>
      <td>19</td>
      <td>3.183763</td>
      <td>Glass</td>
      <td>Brown</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>-4588.255733</td>
      <td>Medium</td>
      <td>Low</td>
      <td>72.715843</td>
      <td>2211.243532</td>
      <td>1325.129948</td>
      <td>886.113585</td>
    </tr>
  </tbody>
</table>
</div>



Conclusion:
Expenditure is calculated by multiplying the cost per unit by the number of units sold. This provides insight into the total cost incurred for selling the products, which helps in evaluating production and procurement expenses.

Revenue is calculated by subtracting Expenditure from Total Profit, representing the net income generated after covering the costs of goods sold. This is a critical metric that gives a clear view of the financial health of the business, helping stakeholders understand profitability at a granular level

#### Dropping Previous revenue from the data


```python
Furniture_EMD = Furniture_EMD.drop(columns=['revenue'])
Furniture_EMD.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>cost</th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>brand</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>218.543053</td>
      <td>181.610932</td>
      <td>40</td>
      <td>27.796433</td>
      <td>Plastic</td>
      <td>Red</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>BrandA</td>
      <td>High</td>
      <td>Medium</td>
      <td>2429.886974</td>
      <td>6311.835165</td>
      <td>7264.437262</td>
      <td>-952.602098</td>
    </tr>
    <tr>
      <th>1</th>
      <td>477.821438</td>
      <td>385.033827</td>
      <td>7</td>
      <td>26.943715</td>
      <td>Glass</td>
      <td>Blue</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>Low</td>
      <td>High</td>
      <td>901.199926</td>
      <td>2443.550139</td>
      <td>2695.236789</td>
      <td>-251.686650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>379.397274</td>
      <td>276.736765</td>
      <td>32</td>
      <td>21.948130</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>Medium</td>
      <td>High</td>
      <td>2664.659446</td>
      <td>9476.053316</td>
      <td>8855.576485</td>
      <td>620.476831</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319.396318</td>
      <td>281.841334</td>
      <td>48</td>
      <td>11.009944</td>
      <td>Glass</td>
      <td>Green</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>BrandD</td>
      <td>High</td>
      <td>Medium</td>
      <td>1687.937139</td>
      <td>13643.086119</td>
      <td>13528.384027</td>
      <td>114.702092</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.208388</td>
      <td>69.743681</td>
      <td>19</td>
      <td>3.183763</td>
      <td>Glass</td>
      <td>Brown</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>Medium</td>
      <td>Low</td>
      <td>72.715843</td>
      <td>2211.243532</td>
      <td>1325.129948</td>
      <td>886.113585</td>
    </tr>
  </tbody>
</table>
</div>



Revenue dropped due to a lack of Accounting stability and This can't be responsible on our Analysis


```python
Furniture_EMD['Revenue_Bin'] = pd.qcut(Furniture_EMD['Revenue'], q=3, labels=['Low', 'Medium', 'High'])
```


```python
Furniture_EMD.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>cost</th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>brand</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Revenue_Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>218.543053</td>
      <td>181.610932</td>
      <td>40</td>
      <td>27.796433</td>
      <td>Plastic</td>
      <td>Red</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>BrandA</td>
      <td>High</td>
      <td>Medium</td>
      <td>2429.886974</td>
      <td>6311.835165</td>
      <td>7264.437262</td>
      <td>-952.602098</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>477.821438</td>
      <td>385.033827</td>
      <td>7</td>
      <td>26.943715</td>
      <td>Glass</td>
      <td>Blue</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>Low</td>
      <td>High</td>
      <td>901.199926</td>
      <td>2443.550139</td>
      <td>2695.236789</td>
      <td>-251.686650</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>379.397274</td>
      <td>276.736765</td>
      <td>32</td>
      <td>21.948130</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>Medium</td>
      <td>High</td>
      <td>2664.659446</td>
      <td>9476.053316</td>
      <td>8855.576485</td>
      <td>620.476831</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319.396318</td>
      <td>281.841334</td>
      <td>48</td>
      <td>11.009944</td>
      <td>Glass</td>
      <td>Green</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>BrandD</td>
      <td>High</td>
      <td>Medium</td>
      <td>1687.937139</td>
      <td>13643.086119</td>
      <td>13528.384027</td>
      <td>114.702092</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.208388</td>
      <td>69.743681</td>
      <td>19</td>
      <td>3.183763</td>
      <td>Glass</td>
      <td>Brown</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>Medium</td>
      <td>Low</td>
      <td>72.715843</td>
      <td>2211.243532</td>
      <td>1325.129948</td>
      <td>886.113585</td>
      <td>Medium</td>
    </tr>
  </tbody>
</table>
</div>



## Function On Binned_Quantile


```python
Furniture_sort_Bin_data = Furniture_EMD.drop(columns=['price','cost'])
```


```python
Furniture_sort_Bin_data.head(
                           ).sort_values(by='brand', ascending = True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>brand</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Revenue_Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>27.796433</td>
      <td>Plastic</td>
      <td>Red</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>BrandA</td>
      <td>High</td>
      <td>Medium</td>
      <td>2429.886974</td>
      <td>6311.835165</td>
      <td>7264.437262</td>
      <td>-952.602098</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>26.943715</td>
      <td>Glass</td>
      <td>Blue</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>Low</td>
      <td>High</td>
      <td>901.199926</td>
      <td>2443.550139</td>
      <td>2695.236789</td>
      <td>-251.686650</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>21.948130</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>Medium</td>
      <td>High</td>
      <td>2664.659446</td>
      <td>9476.053316</td>
      <td>8855.576485</td>
      <td>620.476831</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>11.009944</td>
      <td>Glass</td>
      <td>Green</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>BrandD</td>
      <td>High</td>
      <td>Medium</td>
      <td>1687.937139</td>
      <td>13643.086119</td>
      <td>13528.384027</td>
      <td>114.702092</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>3.183763</td>
      <td>Glass</td>
      <td>Brown</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>BrandD</td>
      <td>Medium</td>
      <td>Low</td>
      <td>72.715843</td>
      <td>2211.243532</td>
      <td>1325.129948</td>
      <td>886.113585</td>
      <td>Medium</td>
    </tr>
  </tbody>
</table>
</div>




```python
Furniture_sort_Bin_data.set_index(['brand'], inplace = True)
```


```python
Furniture_sort_Bin = Furniture_sort_Bin_data.sort_values(by=['sales_Binned_Quantile'], ascending=True)
```


```python
Furniture_sort_Bin.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Revenue_Bin</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandC</th>
      <td>6</td>
      <td>9.502415</td>
      <td>Glass</td>
      <td>Green</td>
      <td>Urban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>Low</td>
      <td>Medium</td>
      <td>133.079178</td>
      <td>1267.398185</td>
      <td>1034.847667</td>
      <td>232.550518</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>10</td>
      <td>19.150253</td>
      <td>Fabric</td>
      <td>White</td>
      <td>Suburban</td>
      <td>Spring</td>
      <td>Retail</td>
      <td>Low</td>
      <td>Low</td>
      <td>196.489478</td>
      <td>829.551706</td>
      <td>731.569033</td>
      <td>97.982673</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>16</td>
      <td>23.254828</td>
      <td>Plastic</td>
      <td>Blue</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>Low</td>
      <td>Medium</td>
      <td>746.787037</td>
      <td>2464.533389</td>
      <td>1756.211931</td>
      <td>708.321457</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>15</td>
      <td>28.595015</td>
      <td>Fabric</td>
      <td>White</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Retail</td>
      <td>Low</td>
      <td>High</td>
      <td>1936.331950</td>
      <td>4835.239701</td>
      <td>3795.959732</td>
      <td>1039.279969</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>3</td>
      <td>5.603424</td>
      <td>Wood</td>
      <td>Red</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>Low</td>
      <td>Medium</td>
      <td>55.311301</td>
      <td>931.786920</td>
      <td>766.806976</td>
      <td>164.979943</td>
      <td>Low</td>
    </tr>
  </tbody>
</table>
</div>




```python
Furniture_sort_Des = Furniture_sort_Bin_data.describe(include='all')
Furniture_sort_Des.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>material</th>
      <th>color</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Revenue_Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2500.000000</td>
      <td>2500.000000</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500.000000</td>
      <td>2500.000000</td>
      <td>2500.000000</td>
      <td>2500.000000</td>
      <td>2500</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>6</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Metal</td>
      <td>Black</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>Low</td>
      <td>Low</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>529</td>
      <td>448</td>
      <td>897</td>
      <td>651</td>
      <td>1307</td>
      <td>867</td>
      <td>834</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>834</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.924000</td>
      <td>14.947616</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1010.361414</td>
      <td>5819.458918</td>
      <td>4785.972204</td>
      <td>1033.486714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.050067</td>
      <td>8.621547</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1077.972771</td>
      <td>4660.425138</td>
      <td>3925.653716</td>
      <td>1500.817953</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.005556</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.113996</td>
      <td>66.653573</td>
      <td>44.773102</td>
      <td>-2970.911824</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13.000000</td>
      <td>7.760214</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>222.637364</td>
      <td>2087.645025</td>
      <td>1687.958370</td>
      <td>106.359084</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>25.000000</td>
      <td>14.915143</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>626.327915</td>
      <td>4611.858075</td>
      <td>3690.251435</td>
      <td>563.759729</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>37.000000</td>
      <td>22.292661</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1443.282541</td>
      <td>8472.903401</td>
      <td>6932.917559</td>
      <td>1567.472373</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>49.000000</td>
      <td>29.991229</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6801.668540</td>
      <td>22819.496739</td>
      <td>20527.692212</td>
      <td>9369.268875</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



OBSERVATION
1) This shows that we had a total of 2,500 sales within the time provided on this data. The company has been operating with five types of materials, six colors, three locations, four seasons, and two modes of store.
2) It observed that Matal is the most selling material with 529 sales with Black colors type being most preferred with 448 sales across all types.
3) THis show that most of our sales come from Rural Areas with 897 sales, due to the development in rural are most of the sales come from the Online store with 1,307 orders and we experience high sales when it comes to Fall Season with 651 sales.

# Growth analysis On Sales Performance						


```python
Furniture_Gro = Furniture_sort_Bin_data.drop(columns=['color','material','material', 'color'])
```


```python
Furniture_Gro.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Revenue_Bin</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandA</th>
      <td>40</td>
      <td>27.796433</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>High</td>
      <td>Medium</td>
      <td>2429.886974</td>
      <td>6311.835165</td>
      <td>7264.437262</td>
      <td>-952.602098</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>7</td>
      <td>26.943715</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>Low</td>
      <td>High</td>
      <td>901.199926</td>
      <td>2443.550139</td>
      <td>2695.236789</td>
      <td>-251.686650</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>32</td>
      <td>21.948130</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>Medium</td>
      <td>High</td>
      <td>2664.659446</td>
      <td>9476.053316</td>
      <td>8855.576485</td>
      <td>620.476831</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>48</td>
      <td>11.009944</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>High</td>
      <td>Medium</td>
      <td>1687.937139</td>
      <td>13643.086119</td>
      <td>13528.384027</td>
      <td>114.702092</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>19</td>
      <td>3.183763</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>Medium</td>
      <td>Low</td>
      <td>72.715843</td>
      <td>2211.243532</td>
      <td>1325.129948</td>
      <td>886.113585</td>
      <td>Medium</td>
    </tr>
  </tbody>
</table>
</div>




```python
Furniture_Gro['Profit_Margin'] = (Furniture_Gro['Revenue'] / Furniture_Gro['Total_Profit'])*100
```


```python
Furniture_Gro.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Revenue_Bin</th>
      <th>Profit_Margin</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandA</th>
      <td>40</td>
      <td>27.796433</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>High</td>
      <td>Medium</td>
      <td>2429.886974</td>
      <td>6311.835165</td>
      <td>7264.437262</td>
      <td>-952.602098</td>
      <td>Low</td>
      <td>-15.092316</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>7</td>
      <td>26.943715</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>Low</td>
      <td>High</td>
      <td>901.199926</td>
      <td>2443.550139</td>
      <td>2695.236789</td>
      <td>-251.686650</td>
      <td>Low</td>
      <td>-10.300040</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>32</td>
      <td>21.948130</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>Medium</td>
      <td>High</td>
      <td>2664.659446</td>
      <td>9476.053316</td>
      <td>8855.576485</td>
      <td>620.476831</td>
      <td>Medium</td>
      <td>6.547840</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>48</td>
      <td>11.009944</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>High</td>
      <td>Medium</td>
      <td>1687.937139</td>
      <td>13643.086119</td>
      <td>13528.384027</td>
      <td>114.702092</td>
      <td>Low</td>
      <td>0.840734</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>19</td>
      <td>3.183763</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>Medium</td>
      <td>Low</td>
      <td>72.715843</td>
      <td>2211.243532</td>
      <td>1325.129948</td>
      <td>886.113585</td>
      <td>Medium</td>
      <td>40.073089</td>
    </tr>
  </tbody>
</table>
</div>



Measuring Efficiency: It highlights how well a company controls its costs relative to its revenue.

Comparing Performance: A higher margin often indicates better financial performance, while lower margins could signal potential issues in cost management or pricing strategy.

Assessing Sustainability: Consistently declining margins may suggest that costs are rising or pricing power is weakening.

Investment Decisions: High or improving margins often make a company more attractive to investors.                                    

## How Each Brand Inputs To Growth


```python
# Select only numeric columns
Furniture_Gro_numeric = Furniture_Gro.select_dtypes(include=['float64', 'int64'])
```


```python
Furniture_Gro_numeric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Profit_Margin</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandA</th>
      <td>40</td>
      <td>27.796433</td>
      <td>2429.886974</td>
      <td>6311.835165</td>
      <td>7264.437262</td>
      <td>-952.602098</td>
      <td>-15.092316</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>7</td>
      <td>26.943715</td>
      <td>901.199926</td>
      <td>2443.550139</td>
      <td>2695.236789</td>
      <td>-251.686650</td>
      <td>-10.300040</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>32</td>
      <td>21.948130</td>
      <td>2664.659446</td>
      <td>9476.053316</td>
      <td>8855.576485</td>
      <td>620.476831</td>
      <td>6.547840</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>48</td>
      <td>11.009944</td>
      <td>1687.937139</td>
      <td>13643.086119</td>
      <td>13528.384027</td>
      <td>114.702092</td>
      <td>0.840734</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>19</td>
      <td>3.183763</td>
      <td>72.715843</td>
      <td>2211.243532</td>
      <td>1325.129948</td>
      <td>886.113585</td>
      <td>40.073089</td>
    </tr>
  </tbody>
</table>
</div>




```python
Furniture_Gro_grouped = Furniture_Gro_numeric.groupby(Furniture_Gro_numeric.index).sum()
```


```python
Furniture_Gro_grouped['discount_percentage'] = Furniture_Gro_grouped['discount_percentage'] / 100
```


```python
Furniture_Gro_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Profit_Margin</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandA</th>
      <td>16000</td>
      <td>98.971678</td>
      <td>623484.255642</td>
      <td>3.665288e+06</td>
      <td>3.014405e+06</td>
      <td>650882.514035</td>
      <td>11072.077473</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>14511</td>
      <td>84.880456</td>
      <td>563785.330947</td>
      <td>3.375580e+06</td>
      <td>2.756390e+06</td>
      <td>619189.765332</td>
      <td>10470.179421</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>16127</td>
      <td>97.703727</td>
      <td>690158.559940</td>
      <td>3.860391e+06</td>
      <td>3.199370e+06</td>
      <td>661020.974084</td>
      <td>10445.159037</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>15672</td>
      <td>92.134540</td>
      <td>648475.389003</td>
      <td>3.647389e+06</td>
      <td>2.994766e+06</td>
      <td>652623.531355</td>
      <td>10726.247283</td>
    </tr>
  </tbody>
</table>
</div>



## correlation_matrix


```python
Furniture_cor_mat = Furniture_Gro_grouped.corr()
```


```python
Furniture_cor_mat.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Profit_Margin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sales</th>
      <td>1.000000</td>
      <td>0.964544</td>
      <td>0.879394</td>
      <td>0.941448</td>
      <td>0.934214</td>
      <td>0.974339</td>
      <td>0.411032</td>
    </tr>
    <tr>
      <th>discount_percentage</th>
      <td>0.964544</td>
      <td>1.000000</td>
      <td>0.749533</td>
      <td>0.868004</td>
      <td>0.863058</td>
      <td>0.881279</td>
      <td>0.516760</td>
    </tr>
    <tr>
      <th>cost_on_discount</th>
      <td>0.879394</td>
      <td>0.749533</td>
      <td>1.000000</td>
      <td>0.973390</td>
      <td>0.971595</td>
      <td>0.951172</td>
      <td>-0.041652</td>
    </tr>
    <tr>
      <th>Total_Profit</th>
      <td>0.941448</td>
      <td>0.868004</td>
      <td>0.973390</td>
      <td>1.000000</td>
      <td>0.999625</td>
      <td>0.962637</td>
      <td>0.080097</td>
    </tr>
    <tr>
      <th>Expenditure</th>
      <td>0.934214</td>
      <td>0.863058</td>
      <td>0.971595</td>
      <td>0.999625</td>
      <td>1.000000</td>
      <td>0.954862</td>
      <td>0.060690</td>
    </tr>
    <tr>
      <th>Revenue</th>
      <td>0.974339</td>
      <td>0.881279</td>
      <td>0.951172</td>
      <td>0.962637</td>
      <td>0.954862</td>
      <td>1.000000</td>
      <td>0.268766</td>
    </tr>
    <tr>
      <th>Profit_Margin</th>
      <td>0.411032</td>
      <td>0.516760</td>
      <td>-0.041652</td>
      <td>0.080097</td>
      <td>0.060690</td>
      <td>0.268766</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



1: Perfect positive correlation. As one variable increases, the other variable also increases perfectly.

-1: Perfect negative correlation. As one variable increases, the other variable decreases perfectly.

0: No correlation. There is no linear relationship between the variables.

# Chi-square: To test what drives sales 

                     ###  sales_Binned_Quantile VS Price_Binned_Quantile


```python
# Create a contingency table
cont_table = pd.crosstab(Furniture_Gro['sales_Binned_Quantile'], Furniture_Gro['Price_Binned_Quantile'])

# Perform Chi-Square Test
chi2_stat, p_value, dof, expected = chi2_contingency(cont_table, correction=True)

# Display results
print(f"Contingency Table:\n{cont_table}")
print(f"\nChi-Square Statistic: {chi2_stat}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies:\n{expected}")
```

    Contingency Table:
    Price_Binned_Quantile  Low  Medium  High
    sales_Binned_Quantile                   
    Low                    296     271   300
    Medium                 258     294   264
    High                   280     268   269
    
    Chi-Square Statistic: 4.77000061174587
    P-value: 0.31172112869632207
    Degrees of Freedom: 4
    Expected Frequencies:
    [[289.2312 288.8844 288.8844]
     [272.2176 271.8912 271.8912]
     [272.5512 272.2244 272.2244]]
    

In this analysis, we tested whether there is a significant relationship between price bins and sales bins. The results of the Chi-Square Test show that the relationship between the two variables is statistically insignificant (p-value = 0.3117). This means that changes in price category do not appear to have a clear impact on sales category.

For decision-making, this suggests that other factors may be influencing sales beyond price alone, and further investigation into those factors may be warranted. Pricing adjustments within the ranges studied here are unlikely to lead to significant changes in overall sales distribution based on this dataset.

Since the p-value is greater than 0.05, we fail to reject the null hypothesis. This means there is no significant association between the price bins and sales bins based on this dataset.

In practical terms, this result suggests that changes in price (across the binned price categories) do not significantly impact sales distribution (across the binned sales categories)

                    ###  sales_Binned_Quantile VS discount_Bin


```python
# Create a contingency table
cont_table = pd.crosstab(Furniture_Gro_BIN['sales_Binned_Quantile'], Furniture_Gro_BIN['discount_Bin'])

# Perform Chi-Square Test
chi2_stat, p_value, dof, expected = chi2_contingency(cont_table, correction=True)

# Display results
print(f"Contingency Table:\n{cont_table}")
print(f"\nChi-Square Statistic: {chi2_stat}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies:\n{expected}")
```

    Contingency Table:
    discount_Bin           Low  Medium  High
    sales_Binned_Quantile                   
    Low                    287     291   289
    Medium                 262     287   267
    High                   285     255   277
    
    Chi-Square Statistic: 3.0860554185888827
    P-value: 0.5435289398673451
    Degrees of Freedom: 4
    Expected Frequencies:
    [[289.2312 288.8844 288.8844]
     [272.2176 271.8912 271.8912]
     [272.5512 272.2244 272.2244]]
    

The Chi-Square Test conducted on the relationship between discount levels and sales quantiles suggests that discounts do not have a significant impact on sales distribution in this dataset (p-value = 0.5435). This implies that the sales performance, categorized as low, medium, or high, is relatively similar across different discount levels (low, medium, high).

For decision-making, this result suggests that discounting strategies alone may not be the primary driver of sales performance in this context. Other factors, such as product quality, marketing, or seasonality, may play a more significant role in influencing sales. Therefore, while discounting remains an important tool, relying solely on it may not lead to substantial sales increases.

### HeatMap ON Brand


```python
pd.options.display.float_format = '{:.2f}'.format
```


```python
# Create a heatmap
sns.heatmap(Furniture_Gro_grouped, annot=True, cmap='coolwarm')

plt.savefig(r"C:\Users\Home\Desktop\Important\Portfolio\Furniture\Correlation Matrix Heatmap2.jpg")

# Show the plot
plt.show()

```


```python
plt.figure(figsize=(9, 5))
sns.heatmap(Furniture_cor_mat, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')

plt.savefig(r"C:\Users\Home\Desktop\Important\Portfolio\Furniture\Correlation Matrix Heatmap.jpg")

plt.show()
```


    
![png](output_85_0.png)
    


## How Discount on the Sale of furniture affect The Revenue and Decisions to make


```python
pd.set_option('display.max_row', None)
```


```python
Furniture_Gro_BIN = Furniture_Gro.drop(columns= ['location','season','store_type','Profit_Margin','Total_Profit',
                                                 'Expenditure','Revenue','cost_on_discount'])
```


```python
Furniture_Gro_BIN['discount_Bin'] = pd.qcut(Furniture_Gro_BIN['discount_percentage'], q=3, labels=['Low', 'Medium', 'High'])
```


```python
Furniture_Gro_BIN_sorted = Furniture_Gro_BIN.sort_values(by='sales', ascending=False)
```


```python
Furniture_Gro_BIN_sorted = Furniture_Gro_BIN_sorted.drop(columns= ['sales','discount_percentage'])
```


```python
Furniture_Gro_BIN_sorted.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>Revenue_Bin</th>
      <th>discount_Bin</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Medium</td>
      <td>Low</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>Medium</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Medium</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Medium</td>
      <td>Low</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Low</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Low</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Medium</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Low</td>
      <td>Low</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Medium</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Medium</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Low</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Medium</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Medium</td>
      <td>Low</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Low</td>
      <td>Low</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Medium</td>
      <td>Low</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>Medium</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>High</td>
      <td>Low</td>
      <td>Medium</td>
      <td>High</td>
    </tr>
    <tr>
      <th>BrandA</th>
      <td>High</td>
      <td>Medium</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>Medium</td>
    </tr>
  </tbody>
</table>
</div>



OBSERVATION
1) This helps in decision-making based on discount percentage with the assumption of not going above the present high discount
2) This helps in observing the cause of low revenue assuming the cost of furniture is fixed
3) This also helps to understand the reaction of customers to discounts on furniture
4) This shows the brand to focus on and reduce its concentration in case of (low sales, low prices, and high discounts)
5) This shows how we can adjust discounts to improve revenue and sales
6) This shows How irritation our customer is

# Visualiztion


```python
Furniture_Gro.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Revenue_Bin</th>
      <th>Profit_Margin</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandA</th>
      <td>40</td>
      <td>27.796433</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>High</td>
      <td>Medium</td>
      <td>2429.886974</td>
      <td>6311.835165</td>
      <td>7264.437262</td>
      <td>-952.602098</td>
      <td>Low</td>
      <td>-15.092316</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>7</td>
      <td>26.943715</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>Low</td>
      <td>High</td>
      <td>901.199926</td>
      <td>2443.550139</td>
      <td>2695.236789</td>
      <td>-251.686650</td>
      <td>Low</td>
      <td>-10.300040</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>32</td>
      <td>21.948130</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>Medium</td>
      <td>High</td>
      <td>2664.659446</td>
      <td>9476.053316</td>
      <td>8855.576485</td>
      <td>620.476831</td>
      <td>Medium</td>
      <td>6.547840</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>48</td>
      <td>11.009944</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>High</td>
      <td>Medium</td>
      <td>1687.937139</td>
      <td>13643.086119</td>
      <td>13528.384027</td>
      <td>114.702092</td>
      <td>Low</td>
      <td>0.840734</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>19</td>
      <td>3.183763</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>Medium</td>
      <td>Low</td>
      <td>72.715843</td>
      <td>2211.243532</td>
      <td>1325.129948</td>
      <td>886.113585</td>
      <td>Medium</td>
      <td>40.073089</td>
    </tr>
  </tbody>
</table>
</div>



## CORRELATION VIEW


```python
plt.figure(figsize=(12, 4))
sns.regplot(x='sales', y='Total_Profit', data=Furniture_Gro, scatter_kws={'color':'blue'}, line_kws={'color':'red'})  # Scatter plot
plt.title('Scatter Plot of sales vs Total_Profit')
plt.xlabel('sales')
plt.ylabel('Total_Profit')
plt.grid(True)

plt.savefig(r"C:\Users\Home\Desktop\Important\Portfolio\Furniture\Scatter Plot of sales vs Total_Profit.jpg")

plt.show()
```


    
![png](output_97_0.png)
    


1) This shows that Sales and Profit a Positive related relationship; which means an increase in sales leads to almost increase in profit
2) The Relationship is affected by discount on sales 


```python
plt.figure(figsize=(12, 4))
sns.regplot(x='sales', y='discount_percentage', data=Furniture_Gro, scatter_kws={'color':'blue'}, line_kws={'color':'red'})  # Scatter plot
plt.title('Scatter Plot of sales vs discount_percentage')
plt.xlabel('sales')
plt.ylabel('discount_percentage')
plt.grid(True)

plt.savefig(r"C:\Users\Home\Desktop\Important\Portfolio\Furniture\Scatter Plot of sales vs Total_Profit.jpg")

plt.show()
```


    
![png](output_99_0.png)
    


1) This shows that Sales and discount_percentage have no  relationship; which means an increase in discount does not lead to an increase in sales
2) This shows that for dales to increase discounts do not need to increase Sales;
3) It shows that it is assumed that a steady or small decrease in sales will not affect sales


```python
# Create the line graph
plt.figure(figsize=(12, 6))
plt.plot(Furniture_Gro['Total_Profit'], Furniture_Gro['Expenditure'], color='blue', marker='o', linestyle='-', label='Trend')

# Add labels and title
plt.title('Total_Profit vs. Expenditure')
plt.xlabel('Total_Profit')
plt.ylabel('Expenditure')

# Add grid and legend
plt.grid(True)
plt.legend()

plt.savefig(r"C:\Users\Home\Desktop\Important\Portfolio\Furniture\Total_Profit vs. Expenditure.jpg")
# Show the plot
plt.show()
```


    
![png](output_101_0.png)
    



```python
# Prepare the data for the regression
X = Furniture_Norm2[['discount_percentage']].values  # Predictor variable
y = Furniture_Norm2['Total_Profit'].values    # Response variable

# Create the linear regression model and fit it
model = LinearRegression()
model.fit(X, y)

# Get the intercept and slope
intercept = model.intercept_
slope = model.coef_[0]

print(f"Intercept: {intercept}, Slope: {slope}")

# Plotting the points and line of best fit
plt.scatter(X, y, color='blue')  # Scatter plot
plt.plot(X, model.predict(X), color='red')  # Line of best fit
plt.title(f"Line Graph with Intercept: {intercept:.2f}")
plt.xlabel("discount_percentage")
plt.ylabel("Total_Profit")

plt.savefig(r"C:\Users\Home\Desktop\Important\Portfolio\Furniture\linear regression model.jpg")

plt.show()

```

    Intercept: -2.8111246508166243e-17, Slope: -0.14986809150240904
    


    
![png](output_102_1.png)
    


# Predictive Analysis on Discount for The company growth


```python
Furniture_Gro.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>location</th>
      <th>season</th>
      <th>store_type</th>
      <th>sales_Binned_Quantile</th>
      <th>Price_Binned_Quantile</th>
      <th>cost_on_discount</th>
      <th>Total_Profit</th>
      <th>Expenditure</th>
      <th>Revenue</th>
      <th>Revenue_Bin</th>
      <th>Profit_Margin</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandA</th>
      <td>40</td>
      <td>27.796433</td>
      <td>Rural</td>
      <td>Spring</td>
      <td>Online</td>
      <td>High</td>
      <td>Medium</td>
      <td>2429.886974</td>
      <td>6311.835165</td>
      <td>7264.437262</td>
      <td>-952.602098</td>
      <td>Low</td>
      <td>-15.092316</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>7</td>
      <td>26.943715</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Online</td>
      <td>Low</td>
      <td>High</td>
      <td>901.199926</td>
      <td>2443.550139</td>
      <td>2695.236789</td>
      <td>-251.686650</td>
      <td>Low</td>
      <td>-10.300040</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>32</td>
      <td>21.948130</td>
      <td>Suburban</td>
      <td>Fall</td>
      <td>Online</td>
      <td>Medium</td>
      <td>High</td>
      <td>2664.659446</td>
      <td>9476.053316</td>
      <td>8855.576485</td>
      <td>620.476831</td>
      <td>Medium</td>
      <td>6.547840</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>48</td>
      <td>11.009944</td>
      <td>Rural</td>
      <td>Summer</td>
      <td>Retail</td>
      <td>High</td>
      <td>Medium</td>
      <td>1687.937139</td>
      <td>13643.086119</td>
      <td>13528.384027</td>
      <td>114.702092</td>
      <td>Low</td>
      <td>0.840734</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>19</td>
      <td>3.183763</td>
      <td>Rural</td>
      <td>Fall</td>
      <td>Online</td>
      <td>Medium</td>
      <td>Low</td>
      <td>72.715843</td>
      <td>2211.243532</td>
      <td>1325.129948</td>
      <td>886.113585</td>
      <td>Medium</td>
      <td>40.073089</td>
    </tr>
  </tbody>
</table>
</div>




```python
Furniture_Gro_Pre = Furniture_Gro.drop(columns= ['location','season','store_type','Price_Binned_Quantile',
                                                'sales_Binned_Quantile', 'Expenditure',
                                                'Revenue_Bin','cost_on_discount','Profit_Margin'])
```


```python
Furniture_Norm2_grouped2 = Furniture_Gro_Pre.groupby(Furniture_Gro_Pre.index).mean()
```


```python
# Create the line graph
plt.figure(figsize=(8, 6))
plt.plot(Furniture_Norm2_grouped2['sales'], Furniture_Norm2_grouped2['discount_percentage'], color='red', marker='o', linestyle='-', label='Trend')

# Add labels and title
plt.title('sales vs. discount_percentage')
plt.xlabel('sales')
plt.ylabel('discount_percentage')

# Add grid and legend
plt.grid(True)
plt.legend()

plt.savefig(r"C:\Users\Home\Desktop\Important\Portfolio\Furniture\sales vs. discount_percentage.jpg")
# Show the plot
plt.show()
```


    
![png](output_107_0.png)
    


This shows the customer reaction to Discount and Sale
In validation on the p_value chi_square that gave the hypothesis as independent variables
1) A high and low discount does not presume a high or low return on sales
2) The sales was driven most when the discount ran at 15%, this shows a fairly encouragement on sale


```python
Furniture_Gro_Pre.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>discount_percentage</th>
      <th>Total_Profit</th>
      <th>Revenue</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandA</th>
      <td>40</td>
      <td>27.796433</td>
      <td>6311.835165</td>
      <td>-952.602098</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>7</td>
      <td>26.943715</td>
      <td>2443.550139</td>
      <td>-251.686650</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>32</td>
      <td>21.948130</td>
      <td>9476.053316</td>
      <td>620.476831</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>48</td>
      <td>11.009944</td>
      <td>13643.086119</td>
      <td>114.702092</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>19</td>
      <td>3.183763</td>
      <td>2211.243532</td>
      <td>886.113585</td>
    </tr>
  </tbody>
</table>
</div>



### Normalisation


```python
Furniture_Norm2 = pd.DataFrame()
```


```python
# Z-score normalization on the 'Total_Profit' column
Furniture_Norm2['Total_Profit'] = (Furniture_Gro_Pre['Total_Profit'] - Furniture_Gro_Pre['Total_Profit'].mean())/ Furniture_Gro_Pre['Total_Profit'].std()
                                                                           
```


```python
# Z-score normalization on the 'discount_percentage' column
Furniture_Norm2['discount_percentage'] = (Furniture_Gro_Pre['discount_percentage'] - Furniture_Gro_Pre['discount_percentage'].mean())/Furniture_Gro_Pre['discount_percentage'].std()
                                                                                    
```


```python
# Z-score normalization on the 'sales' column
Furniture_Norm2['sales'] = (Furniture_Gro_Pre['sales'] - Furniture_Gro_Pre['sales'].mean())/ Furniture_Gro_Pre['sales'].std()
```


```python
Furniture_Norm2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total_Profit</th>
      <th>discount_percentage</th>
      <th>sales</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandA</th>
      <td>0.105651</td>
      <td>1.490315</td>
      <td>1.073020</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>-0.724378</td>
      <td>1.391409</td>
      <td>-1.275723</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>0.784605</td>
      <td>0.811979</td>
      <td>0.503627</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>1.678737</td>
      <td>-0.456725</td>
      <td>1.642412</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>-0.774225</td>
      <td>-1.364471</td>
      <td>-0.421635</td>
    </tr>
  </tbody>
</table>
</div>




```python
Furniture_Norm2_grouped = Furniture_Norm2.groupby(Furniture_Norm2.index).sum()
```


```python
Furniture_Norm2_grouped.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total_Profit</th>
      <th>discount_percentage</th>
      <th>sales</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrandA</th>
      <td>-25.182412</td>
      <td>21.019127</td>
      <td>-14.277512</td>
    </tr>
    <tr>
      <th>BrandB</th>
      <td>-9.926555</td>
      <td>-34.930234</td>
      <td>-10.271268</td>
    </tr>
    <tr>
      <th>BrandC</th>
      <td>24.173584</td>
      <td>16.714863</td>
      <td>5.405241</td>
    </tr>
    <tr>
      <th>BrandD</th>
      <td>10.935384</td>
      <td>-2.803756</td>
      <td>19.143538</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(Furniture_Gro.columns)
```

    Index(['sales', 'discount_percentage', 'location', 'season', 'store_type',
           'sales_Binned_Quantile', 'Price_Binned_Quantile', 'cost_on_discount',
           'Total_Profit', 'Expenditure', 'Revenue', 'Revenue_Bin',
           'Profit_Margin'],
          dtype='object')
    


```python
Furniture_Gro['Discount_Bin'] = pd.qcut(Furniture_data['discount_percentage'], q=3, labels=['Low', 'Medium', 'High'])
```


```python
file_path = r'C:\Users\Home\Desktop\Important\Portfolio\Furniture\Furniture EDA.csv'
Furniture_Gro.to_csv(file_path, index=True, header = True)
```


```python

```


```python

```
