```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
Data = pd.read_csv(r"C:\Users\Home\Desktop\Important\QVI_data.csv")
```

## Select control stores

#### Using this Matrixs


```python
store_rating = Data.groupby('STORE_NBR').agg(
    Total_sales=('TOT_SALES', 'sum'),
    Avg_sales_per_txn=('TOT_SALES', 'mean'),
    Total_transactions=('TXN_ID', 'nunique'),
    Customers_behaviour=('PREMIUM_CUSTOMER', 'count'),
    Lifestage_reaction=('LIFESTAGE', 'count')
)
store_rating.head()
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
      <th>Total_sales</th>
      <th>Avg_sales_per_txn</th>
      <th>Total_transactions</th>
      <th>Customers_behaviour</th>
      <th>Lifestage_reaction</th>
    </tr>
    <tr>
      <th>STORE_NBR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2393.60</td>
      <td>4.177312</td>
      <td>572</td>
      <td>573</td>
      <td>573</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005.80</td>
      <td>3.964032</td>
      <td>505</td>
      <td>506</td>
      <td>506</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12802.45</td>
      <td>8.523602</td>
      <td>1489</td>
      <td>1502</td>
      <td>1502</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14647.65</td>
      <td>8.729231</td>
      <td>1667</td>
      <td>1678</td>
      <td>1678</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9500.80</td>
      <td>6.980749</td>
      <td>1358</td>
      <td>1361</td>
      <td>1361</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_stores = store_rating.sort_values(by='Total_sales', ascending=False).head(5)
test_stores.head()
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
      <th>Total_sales</th>
      <th>Avg_sales_per_txn</th>
      <th>Total_transactions</th>
      <th>Customers_behaviour</th>
      <th>Lifestage_reaction</th>
    </tr>
    <tr>
      <th>STORE_NBR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>226</th>
      <td>17605.45</td>
      <td>8.715569</td>
      <td>2008</td>
      <td>2020</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>88</th>
      <td>16333.25</td>
      <td>8.720368</td>
      <td>1857</td>
      <td>1873</td>
      <td>1873</td>
    </tr>
    <tr>
      <th>165</th>
      <td>15973.75</td>
      <td>8.781611</td>
      <td>1806</td>
      <td>1819</td>
      <td>1819</td>
    </tr>
    <tr>
      <th>40</th>
      <td>15559.50</td>
      <td>8.820578</td>
      <td>1756</td>
      <td>1764</td>
      <td>1764</td>
    </tr>
    <tr>
      <th>237</th>
      <td>15539.50</td>
      <td>8.705602</td>
      <td>1773</td>
      <td>1785</td>
      <td>1785</td>
    </tr>
  </tbody>
</table>
</div>




```python
descriptive_store_rating = store_rating.describe()
descriptive_store_rating.head()
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
      <th>Total_sales</th>
      <th>Avg_sales_per_txn</th>
      <th>Total_transactions</th>
      <th>Customers_behaviour</th>
      <th>Lifestage_reaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>272.000000</td>
      <td>272.000000</td>
      <td>272.000000</td>
      <td>272.000000</td>
      <td>272.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7107.040441</td>
      <td>6.850036</td>
      <td>967.371324</td>
      <td>973.654412</td>
      <td>973.654412</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4705.862960</td>
      <td>1.549499</td>
      <td>582.194618</td>
      <td>587.775315</td>
      <td>587.775315</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.200000</td>
      <td>2.600000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2899.425000</td>
      <td>5.811847</td>
      <td>528.250000</td>
      <td>528.750000</td>
      <td>528.750000</td>
    </tr>
  </tbody>
</table>
</div>



#### USing A percentiles statistics analysis


```python
percentiles = store_rating.quantile([0.25, 0.5, 0.75])
percentiles.head()
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
      <th>Total_sales</th>
      <th>Avg_sales_per_txn</th>
      <th>Total_transactions</th>
      <th>Customers_behaviour</th>
      <th>Lifestage_reaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.25</th>
      <td>2899.425</td>
      <td>5.811847</td>
      <td>528.25</td>
      <td>528.75</td>
      <td>528.75</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>5972.625</td>
      <td>6.917889</td>
      <td>677.50</td>
      <td>679.00</td>
      <td>679.00</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>10952.775</td>
      <td>8.487420</td>
      <td>1513.25</td>
      <td>1525.25</td>
      <td>1525.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_store_ids = [226, 88,165]  
test_stores = store_rating.loc[store_rating.index.isin(test_store_ids)]

test_percentiles = test_stores.rank(pct=True)
print("Test Stores Percentiles:")
test_percentiles.head()

```

    Test Stores Percentiles:
    




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
      <th>Total_sales</th>
      <th>Avg_sales_per_txn</th>
      <th>Total_transactions</th>
      <th>Customers_behaviour</th>
      <th>Lifestage_reaction</th>
    </tr>
    <tr>
      <th>STORE_NBR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>88</th>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>165</th>
      <td>0.333333</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>226</th>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
percentile_bounds = (0.40, 0.60) 
```


```python
potential_control_stores = store_rating[
    (store_rating['Total_sales'].rank(pct=True).between(percentile_bounds[0], percentile_bounds[1])) &
    (store_rating['Total_transactions'].rank(pct=True).between(percentile_bounds[0], percentile_bounds[1]))
]
print("Potential Control Stores within the Same Percentile Range:")
potential_control_stores.head(10)
```

    Potential Control Stores within the Same Percentile Range:
    




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
      <th>Total_sales</th>
      <th>Avg_sales_per_txn</th>
      <th>Total_transactions</th>
      <th>Customers_behaviour</th>
      <th>Lifestage_reaction</th>
    </tr>
    <tr>
      <th>STORE_NBR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>9500.80</td>
      <td>6.980749</td>
      <td>1358</td>
      <td>1361</td>
      <td>1361</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9486.05</td>
      <td>6.888925</td>
      <td>1372</td>
      <td>1377</td>
      <td>1377</td>
    </tr>
    <tr>
      <th>19</th>
      <td>8972.90</td>
      <td>7.782220</td>
      <td>1146</td>
      <td>1153</td>
      <td>1153</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5154.90</td>
      <td>8.591500</td>
      <td>598</td>
      <td>600</td>
      <td>600</td>
    </tr>
    <tr>
      <th>24</th>
      <td>9230.00</td>
      <td>6.847181</td>
      <td>1339</td>
      <td>1348</td>
      <td>1348</td>
    </tr>
    <tr>
      <th>25</th>
      <td>5297.65</td>
      <td>8.814725</td>
      <td>600</td>
      <td>601</td>
      <td>601</td>
    </tr>
    <tr>
      <th>27</th>
      <td>5120.70</td>
      <td>8.708673</td>
      <td>588</td>
      <td>588</td>
      <td>588</td>
    </tr>
    <tr>
      <th>28</th>
      <td>9374.35</td>
      <td>6.908143</td>
      <td>1346</td>
      <td>1357</td>
      <td>1357</td>
    </tr>
    <tr>
      <th>45</th>
      <td>9508.50</td>
      <td>6.905229</td>
      <td>1364</td>
      <td>1377</td>
      <td>1377</td>
    </tr>
    <tr>
      <th>54</th>
      <td>5370.85</td>
      <td>8.733089</td>
      <td>615</td>
      <td>615</td>
      <td>615</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_means = test_stores.mean()
control_means = potential_control_stores.mean()

print("Test Stores Means:")
test_means.head()
```

    Test Stores Means:
    




    Total_sales            16637.483333
    Avg_sales_per_txn          8.739183
    Total_transactions      1890.333333
    Customers_behaviour     1904.000000
    Lifestage_reaction      1904.000000
    dtype: float64




```python
control_means = potential_control_stores.mean()

print("Potential_control_stores:")
control_means.head()
```

    Potential_control_stores:
    




    Total_sales            7637.560811
    Avg_sales_per_txn         7.661472
    Total_transactions     1029.081081
    Customers_behaviour    1035.081081
    Lifestage_reaction     1035.081081
    dtype: float64




```python
data2 = potential_control_stores.merge(Data,on ='STORE_NBR', how= 'inner')
data2.head()
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
      <th>STORE_NBR</th>
      <th>Total_sales</th>
      <th>Avg_sales_per_txn</th>
      <th>Total_transactions</th>
      <th>Customers_behaviour</th>
      <th>Lifestage_reaction</th>
      <th>LYLTY_CARD_NBR</th>
      <th>DATE</th>
      <th>TXN_ID</th>
      <th>PROD_NBR</th>
      <th>PROD_NAME</th>
      <th>PROD_QTY</th>
      <th>TOT_SALES</th>
      <th>PACK_SIZE</th>
      <th>BRAND</th>
      <th>LIFESTAGE</th>
      <th>PREMIUM_CUSTOMER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>9500.8</td>
      <td>6.980749</td>
      <td>1358</td>
      <td>1361</td>
      <td>1361</td>
      <td>5000</td>
      <td>2018-07-22</td>
      <td>4356</td>
      <td>111</td>
      <td>Smiths Chip Thinly  Cut Original 175g</td>
      <td>2</td>
      <td>6.0</td>
      <td>175</td>
      <td>SMITHS</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>9500.8</td>
      <td>6.980749</td>
      <td>1358</td>
      <td>1361</td>
      <td>1361</td>
      <td>5000</td>
      <td>2018-08-06</td>
      <td>4357</td>
      <td>57</td>
      <td>Old El Paso Salsa   Dip Tomato Mild 300g</td>
      <td>2</td>
      <td>10.2</td>
      <td>300</td>
      <td>OLD</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9500.8</td>
      <td>6.980749</td>
      <td>1358</td>
      <td>1361</td>
      <td>1361</td>
      <td>5000</td>
      <td>2018-09-11</td>
      <td>4358</td>
      <td>86</td>
      <td>Cheetos Puffs 165g</td>
      <td>2</td>
      <td>5.6</td>
      <td>165</td>
      <td>CHEETOS</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>9500.8</td>
      <td>6.980749</td>
      <td>1358</td>
      <td>1361</td>
      <td>1361</td>
      <td>5000</td>
      <td>2018-09-15</td>
      <td>4359</td>
      <td>15</td>
      <td>Twisties Cheese     270g</td>
      <td>2</td>
      <td>9.2</td>
      <td>270</td>
      <td>TWISTIES</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>9500.8</td>
      <td>6.980749</td>
      <td>1358</td>
      <td>1361</td>
      <td>1361</td>
      <td>5000</td>
      <td>2018-10-16</td>
      <td>4360</td>
      <td>10</td>
      <td>RRD SR Slow Rst     Pork Belly 150g</td>
      <td>2</td>
      <td>5.4</td>
      <td>150</td>
      <td>RRD</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
  </tbody>
</table>
</div>




```python
control_stores_data = data2.drop(columns=['Total_sales','Avg_sales_per_txn','Total_transactions',	'Customers_behaviour','Lifestage_reaction'])
```


```python
control_stores_data .head()
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
      <th>STORE_NBR</th>
      <th>LYLTY_CARD_NBR</th>
      <th>DATE</th>
      <th>TXN_ID</th>
      <th>PROD_NBR</th>
      <th>PROD_NAME</th>
      <th>PROD_QTY</th>
      <th>TOT_SALES</th>
      <th>PACK_SIZE</th>
      <th>BRAND</th>
      <th>LIFESTAGE</th>
      <th>PREMIUM_CUSTOMER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>5000</td>
      <td>2018-07-22</td>
      <td>4356</td>
      <td>111</td>
      <td>Smiths Chip Thinly  Cut Original 175g</td>
      <td>2</td>
      <td>6.0</td>
      <td>175</td>
      <td>SMITHS</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>5000</td>
      <td>2018-08-06</td>
      <td>4357</td>
      <td>57</td>
      <td>Old El Paso Salsa   Dip Tomato Mild 300g</td>
      <td>2</td>
      <td>10.2</td>
      <td>300</td>
      <td>OLD</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>5000</td>
      <td>2018-09-11</td>
      <td>4358</td>
      <td>86</td>
      <td>Cheetos Puffs 165g</td>
      <td>2</td>
      <td>5.6</td>
      <td>165</td>
      <td>CHEETOS</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>5000</td>
      <td>2018-09-15</td>
      <td>4359</td>
      <td>15</td>
      <td>Twisties Cheese     270g</td>
      <td>2</td>
      <td>9.2</td>
      <td>270</td>
      <td>TWISTIES</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5000</td>
      <td>2018-10-16</td>
      <td>4360</td>
      <td>10</td>
      <td>RRD SR Slow Rst     Pork Belly 150g</td>
      <td>2</td>
      <td>5.4</td>
      <td>150</td>
      <td>RRD</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
  </tbody>
</table>
</div>



Setting Date column as Date format incase of any issues


```python
control_stores_data['DATE'] = pd.to_datetime(control_stores_data['DATE'])
```

## filter to stores that are present 
throughout the pre-trial period.


```python
specific_date = '2019-02-01'

filtered_data3 = control_stores_data[control_stores_data['DATE'] >= specific_date]
filtered_data3.head()
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
      <th>STORE_NBR</th>
      <th>LYLTY_CARD_NBR</th>
      <th>DATE</th>
      <th>TXN_ID</th>
      <th>PROD_NBR</th>
      <th>PROD_NAME</th>
      <th>PROD_QTY</th>
      <th>TOT_SALES</th>
      <th>PACK_SIZE</th>
      <th>BRAND</th>
      <th>LIFESTAGE</th>
      <th>PREMIUM_CUSTOMER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>5000</td>
      <td>2019-02-22</td>
      <td>4362</td>
      <td>92</td>
      <td>WW Crinkle Cut      Chicken 175g</td>
      <td>2</td>
      <td>3.4</td>
      <td>175</td>
      <td>WOOLWORTHS</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>5000</td>
      <td>2019-06-23</td>
      <td>4363</td>
      <td>24</td>
      <td>Grain Waves         Sweet Chilli 210g</td>
      <td>2</td>
      <td>7.2</td>
      <td>210</td>
      <td>GRNWVES</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>5001</td>
      <td>2019-03-12</td>
      <td>4368</td>
      <td>56</td>
      <td>Cheezels Cheese Box 125g</td>
      <td>2</td>
      <td>4.2</td>
      <td>125</td>
      <td>CHEEZELS</td>
      <td>OLDER FAMILIES</td>
      <td>Premium</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>5001</td>
      <td>2019-05-10</td>
      <td>4369</td>
      <td>37</td>
      <td>Smiths Thinly       Swt Chli&amp;S/Cream175G</td>
      <td>2</td>
      <td>6.0</td>
      <td>175</td>
      <td>SMITHS</td>
      <td>OLDER FAMILIES</td>
      <td>Premium</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5</td>
      <td>5002</td>
      <td>2019-02-03</td>
      <td>4375</td>
      <td>81</td>
      <td>Pringles Original   Crisps 134g</td>
      <td>2</td>
      <td>7.4</td>
      <td>134</td>
      <td>PRINGLES</td>
      <td>YOUNG SINGLES/COUPLES</td>
      <td>Mainstream</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.set_option('display.max_rows', None)
```

## Calculate these measures over time for each store


```python

measures_over_time = filtered_data3.groupby([pd.Grouper(key='DATE', freq='M'), 'STORE_NBR']).agg(
    Total_Sales=('TOT_SALES', 'sum'),
    Num_Transactions=('TXN_ID', 'count'),
    Avg_Transaction_value=('TOT_SALES', 'mean')
)
```


```python
measures_over_time = measures_over_time.reset_index()
```


```python
 measures_over_time.rename(columns={'DATE': 'Month_ID'},inplace=True)
```


```python
print("After renaming:", measures_over_time.columns)
```

    After renaming: Index(['Month_ID', 'STORE_NBR', 'Total_Sales', 'Num_Transactions',
           'Avg_Transaction_value'],
          dtype='object')
    


```python
measures_over_time.head()
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
      <th>Month_ID</th>
      <th>STORE_NBR</th>
      <th>Total_Sales</th>
      <th>Num_Transactions</th>
      <th>Avg_Transaction_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-02-28</td>
      <td>5</td>
      <td>727.0</td>
      <td>106</td>
      <td>6.858491</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-02-28</td>
      <td>15</td>
      <td>700.8</td>
      <td>96</td>
      <td>7.300000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-02-28</td>
      <td>19</td>
      <td>742.1</td>
      <td>91</td>
      <td>8.154945</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-02-28</td>
      <td>21</td>
      <td>406.8</td>
      <td>46</td>
      <td>8.843478</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-02-28</td>
      <td>24</td>
      <td>693.2</td>
      <td>93</td>
      <td>7.453763</td>
    </tr>
  </tbody>
</table>
</div>




```python
measures_over_time['Month_ID'] = pd.to_datetime(measures_over_time['Month_ID'])
```


```python
measures_over_time['YEARMONTH'] =measures_over_time['Month_ID'].dt.strftime('%Y%m')
```


```python
measures_over_time.head()
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
      <th>Month_ID</th>
      <th>STORE_NBR</th>
      <th>Total_Sales</th>
      <th>Num_Transactions</th>
      <th>Avg_Transaction_value</th>
      <th>YEARMONTH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-02-28</td>
      <td>5</td>
      <td>727.0</td>
      <td>106</td>
      <td>6.858491</td>
      <td>201902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-02-28</td>
      <td>15</td>
      <td>700.8</td>
      <td>96</td>
      <td>7.300000</td>
      <td>201902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-02-28</td>
      <td>19</td>
      <td>742.1</td>
      <td>91</td>
      <td>8.154945</td>
      <td>201902</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-02-28</td>
      <td>21</td>
      <td>406.8</td>
      <td>46</td>
      <td>8.843478</td>
      <td>201902</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-02-28</td>
      <td>24</td>
      <td>693.2</td>
      <td>93</td>
      <td>7.453763</td>
      <td>201902</td>
    </tr>
  </tbody>
</table>
</div>




```python
measures_over_time.dtypes
```




    Month_ID                 datetime64[ns]
    STORE_NBR                         int64
    Total_Sales                     float64
    Num_Transactions                  int64
    Avg_Transaction_value           float64
    YEARMONTH                        object
    dtype: object




```python
measures_over_time['YEARMONTH'] = pd.to_numeric(measures_over_time['YEARMONTH'], errors='coerce')
measures_over_time.dtypes
```




    Month_ID                 datetime64[ns]
    STORE_NBR                         int64
    Total_Sales                     float64
    Num_Transactions                  int64
    Avg_Transaction_value           float64
    YEARMONTH                         int64
    dtype: object




```python
stores_with_full_obs = measures_over_time.groupby('STORE_NBR').filter(lambda x: len(x) == 12)['STORE_NBR'].unique()
print(stores_with_full_obs)
```

    []
    


```python
import numpy as np
```


```python
filtered_numeric = filtered_data3.select_dtypes(include=[float, int])
correlation_matrix = filtered_numeric.corr()
```


```python
correlation_matrix.head()
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
      <th>STORE_NBR</th>
      <th>LYLTY_CARD_NBR</th>
      <th>TXN_ID</th>
      <th>PROD_NBR</th>
      <th>PROD_QTY</th>
      <th>TOT_SALES</th>
      <th>PACK_SIZE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>STORE_NBR</th>
      <td>1.000000</td>
      <td>0.999999</td>
      <td>0.999690</td>
      <td>0.013875</td>
      <td>0.014643</td>
      <td>0.016749</td>
      <td>-0.003950</td>
    </tr>
    <tr>
      <th>LYLTY_CARD_NBR</th>
      <td>0.999999</td>
      <td>1.000000</td>
      <td>0.999694</td>
      <td>0.013852</td>
      <td>0.014582</td>
      <td>0.016847</td>
      <td>-0.003934</td>
    </tr>
    <tr>
      <th>TXN_ID</th>
      <td>0.999690</td>
      <td>0.999694</td>
      <td>1.000000</td>
      <td>0.013998</td>
      <td>0.014730</td>
      <td>0.016952</td>
      <td>-0.003781</td>
    </tr>
    <tr>
      <th>PROD_NBR</th>
      <td>0.013875</td>
      <td>0.013852</td>
      <td>0.013998</td>
      <td>1.000000</td>
      <td>0.004191</td>
      <td>-0.171494</td>
      <td>-0.222583</td>
    </tr>
    <tr>
      <th>PROD_QTY</th>
      <td>0.014643</td>
      <td>0.014582</td>
      <td>0.014730</td>
      <td>0.004191</td>
      <td>1.000000</td>
      <td>0.423181</td>
      <td>-0.010038</td>
    </tr>
  </tbody>
</table>
</div>




```python
 calc_corr_table = pd.DataFrame(columns=["Store1", "Store2", "corr_measure"])
```


```python
store_numbers = filtered_data3['STORE_NBR'].unique()
print(store_numbers)
```

    [  5  15  19  21  24  25  27  28  45  54  56  66  70  78  84 107 110 129
     134 142 144 148 173 200 202 208 209 212 222 235 241 242 243 246 260 262
     271]
    


```python
metric1 = filtered_data3[filtered_data3['STORE_NBR'] == store1]['TOT_SALES']
print(metric1)
```

    6        3.4
    7        7.2
    12       4.2
    13       6.0
    19       7.4
    20       6.6
    21       5.4
    22       6.0
    26       3.8
    30       7.2
    31       5.4
    32       9.2
    36       6.0
    37       5.2
    42       3.8
    43       8.4
    44       9.2
    52       5.2
    53       5.4
    54       9.2
    55       3.8
    60       6.6
    61       7.4
    62       8.8
    63       5.2
    68       5.8
    69       7.2
    74       7.6
    75       7.4
    76      10.8
    77       7.4
    82       7.8
    83       9.2
    85       7.6
    86       8.8
    88       5.2
    89       6.6
    91       7.4
    92       7.4
    93       9.2
    94       6.0
    97       3.8
    98       4.2
    99       9.2
    104     10.8
    108      4.2
    109     10.2
    110      4.4
    117     11.4
    118      7.6
    125      3.8
    132      5.2
    133      6.0
    134     11.4
    139      9.2
    140      3.4
    141      6.0
    145     10.8
    146      7.4
    147      4.2
    148      4.2
    149      8.8
    158      7.6
    159     10.2
    160      8.8
    161      5.4
    162      5.4
    163      6.6
    173      7.6
    179      3.4
    180     10.2
    181      5.2
    182      4.2
    183      2.7
    187      5.4
    188      6.0
    189      7.2
    190      7.6
    191     10.8
    192      5.7
    195      9.2
    196      8.8
    200      9.2
    201      6.0
    202      5.8
    206      4.8
    207      6.0
    208      5.2
    209      6.0
    214      3.8
    215      5.2
    216      2.7
    225      8.4
    226      8.8
    227     13.0
    234      7.6
    235      6.2
    240      7.4
    241      5.4
    244      6.6
    245      7.4
    246      3.8
    247      9.2
    250      3.8
    251      6.0
    252     10.8
    257     10.2
    263      9.2
    265      8.8
    266      7.6
    267      6.0
    268      5.4
    269      9.2
    271      7.6
    272      9.2
    273      3.8
    274      5.4
    277      7.2
    284      9.2
    286      3.8
    292      5.8
    293      5.2
    294      8.8
    297      4.2
    298      8.8
    299     11.4
    305      6.0
    306      5.4
    307      5.4
    308      5.8
    314      6.0
    319      7.4
    320     10.8
    321      6.2
    322      6.6
    325      3.0
    326      4.2
    327     10.8
    328      6.0
    334      6.0
    338      8.8
    339      6.0
    340      5.6
    343      3.8
    344      8.8
    345      9.2
    349      3.4
    350      5.4
    354     10.8
    355     11.4
    356      4.2
    357      6.0
    360      9.2
    361      6.6
    362      5.4
    366      6.0
    367      5.2
    370      7.4
    375      6.0
    376      7.4
    380      3.8
    381     18.4
    385     10.8
    386      8.8
    393     10.8
    397      5.8
    398     10.2
    406      6.0
    407      5.2
    408      6.0
    409      4.6
    412      4.2
    413      3.6
    414      4.8
    417      7.2
    418      8.8
    420      5.6
    421      7.2
    422      3.8
    424      7.4
    425      9.2
    426      7.6
    428      9.2
    429      8.4
    430     10.8
    436      4.2
    437     10.2
    441      9.2
    442      6.0
    443      6.0
    448      9.2
    449      6.0
    450     11.8
    451      6.6
    455      3.8
    456      5.4
    457      7.2
    458      6.0
    460     15.0
    467      3.6
    474      8.8
    475      8.8
    476      3.4
    480      3.8
    481     10.2
    487      6.0
    496      6.6
    499      3.6
    500     11.4
    501      9.2
    502      8.4
    503      8.8
    506      7.4
    508      6.0
    514     11.8
    515      3.4
    516      7.6
    517      7.4
    523     10.2
    524      7.4
    525      7.6
    526      7.8
    527      4.8
    528      5.8
    535      6.6
    537      4.6
    539      6.6
    540      3.0
    543      6.0
    544      6.0
    545      6.0
    546      7.4
    547      6.6
    553     11.4
    554      8.8
    555      3.8
    560      3.8
    567      6.0
    569      5.8
    570      3.4
    571      8.8
    573      7.4
    574      7.2
    583      8.8
    590      3.8
    594      5.2
    595      6.0
    596     10.2
    600      7.4
    601      6.0
    602      9.2
    603      7.2
    604      7.2
    608      6.0
    609      4.8
    610      9.2
    611      3.8
    617      5.4
    618      8.8
    619      6.0
    623      5.6
    624      6.0
    631      5.2
    632      6.0
    633      3.8
    634      7.4
    638      7.6
    647     11.4
    652      6.0
    653      3.8
    655      3.8
    656      3.8
    657      3.4
    658      6.6
    659      6.2
    663      4.2
    666     11.4
    667      7.4
    668      6.0
    669      4.8
    671      7.4
    672      6.0
    673     11.4
    675     11.1
    681     10.8
    686      9.2
    687      4.6
    688     10.8
    692     11.4
    693      8.8
    700      3.4
    701      6.0
    702      6.0
    703      7.2
    704      7.6
    718     11.8
    719     11.4
    723      5.4
    724      6.6
    725      6.6
    730      9.2
    731      5.8
    732      6.0
    738      9.2
    739      3.4
    740      5.6
    741      5.2
    742      9.2
    744      5.7
    747      7.4
    748      9.2
    749      6.0
    754      8.8
    755      6.0
    757     11.4
    758      7.6
    759      5.8
    760      9.2
    761      7.4
    764      7.6
    765      8.6
    771      6.0
    772      7.8
    773      8.8
    776      7.2
    777      8.4
    782      7.6
    783      9.2
    789      4.2
    790      3.4
    799      9.2
    800      3.9
    803      6.6
    804      7.6
    805      6.0
    807     13.0
    808      6.0
    809      3.8
    814      5.2
    815      6.0
    818     11.8
    819      7.6
    822      7.8
    823      7.6
    830      7.6
    834      9.2
    840      5.4
    841      7.6
    842      5.8
    843      7.2
    844      4.2
    847      6.6
    852      9.2
    853      7.2
    854      7.4
    859      5.8
    864      5.4
    868      3.4
    869     13.0
    870      8.4
    871      5.4
    875      8.8
    876     10.8
    877     10.2
    882      5.4
    887      7.6
    888      7.6
    892      6.6
    893      9.2
    894      7.4
    899      5.1
    908      6.6
    909      3.0
    910      9.2
    916      5.2
    917      9.2
    918      8.4
    919      6.0
    920      6.6
    931      9.2
    932      9.2
    939      7.6
    940     10.8
    947      4.2
    958      3.8
    977      5.4
    980      7.6
    981      5.2
    982      4.2
    983      5.2
    984      7.4
    989      3.8
    998      3.8
    1006     6.0
    1007     7.4
    1008     6.0
    1009     4.2
    1010     5.4
    1011     6.0
    1012     7.6
    1015     8.8
    1021    10.2
    1022     3.8
    1024     8.8
    1025     5.4
    1026     5.6
    1027     5.8
    1028     9.2
    1035     8.8
    1036     7.8
    1037     6.0
    1044     8.8
    1045    10.8
    1046    10.8
    1047     5.4
    1048     5.2
    1054     6.0
    1055     5.8
    1058     5.2
    1066     5.2
    1067     5.2
    1071     5.4
    1072     9.2
    1073     5.4
    1074     7.4
    1077     6.0
    1078     5.4
    1079     3.8
    1082     8.4
    1085     5.2
    1086     5.8
    1087     5.4
    1088    10.8
    1099     7.4
    1100     9.2
    1102     6.6
    1103     5.8
    1108     6.0
    1109     6.2
    1114     6.0
    1115     6.0
    1121     5.4
    1122     4.6
    1123    10.8
    1124     3.4
    1125     7.4
    1126     8.4
    1127    10.8
    1132     2.6
    1138     6.0
    1139     8.6
    1142     6.0
    1143     7.6
    1144     6.0
    1145     6.0
    1153     6.0
    1154     9.2
    1158     7.4
    1159     6.0
    1160     7.6
    1162     4.2
    1167     7.6
    1168     3.6
    1174     6.0
    1175     6.0
    1176     5.8
    1177     6.0
    1178     3.6
    1185     6.0
    1186     5.8
    1187     7.2
    1189     8.4
    1190     3.4
    1191     7.2
    1192     7.6
    1195     6.6
    1199     3.0
    1200     8.4
    1201     8.6
    1202     3.8
    1209    13.0
    1215     6.0
    1216     9.2
    1217     6.0
    1222     6.0
    1223     5.2
    1226    11.4
    1229     9.2
    1230     6.0
    1231    11.4
    1235     5.6
    1236     5.2
    1241    10.2
    1242     3.4
    1243     3.3
    1246     9.2
    1247     5.2
    1252     6.2
    1253     8.8
    1254     5.2
    1255     7.8
    1263     9.2
    1264     3.4
    1268     7.6
    1269     8.8
    1274     6.2
    1275     6.6
    1276     2.9
    1280     9.2
    1284     4.8
    1287     3.0
    1288     5.8
    1292     6.0
    1297     7.4
    1298     8.4
    1301    10.8
    1302     9.2
    1303     7.6
    1307     6.6
    1312     7.6
    1313    11.8
    1314     8.8
    1316     2.7
    1320     7.6
    1321    10.8
    1322     9.2
    1323     6.0
    1326    10.2
    1327     6.0
    1328    10.8
    1330     2.1
    1336     6.0
    1337     5.2
    1338     6.0
    1343     7.6
    1344     7.6
    1345     6.0
    1346    10.2
    1351     5.4
    1357     6.0
    1358     3.8
    1359     5.8
    1360     5.2
    Name: TOT_SALES, dtype: float64
    


```python
metric2 = filtered_data3[filtered_data3['STORE_NBR'] == store2]['TOT_SALES']
```


```python
 corr_measure = metric1.corr(metric2)
print(corr_measure)
```

    nan
    


```python
 new_row = pd.DataFrame({
                    "Store1": [store1], 
                    "Store2": [store2], 
                    "corr_measure": [corr_measure]
                })
```


```python
 calc_corr_table = pd.concat([calc_corr_table, new_row], ignore_index=True)
```


```python
calc_corr_table.head()
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
      <th>Store1</th>
      <th>Store2</th>
      <th>corr_measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>15</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
standardized_metric_table = pd.DataFrame(columns=["Store", "standardized_metric"])
```


```python
trial_performance = filtered_data3[filtered_data3['STORE_NBR'] == store_numbers][TOT_SALES].values
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[85], line 1
    ----> 1 trial_performance = filtered_data3[filtered_data3['STORE_NBR'] == store_numbers][TOT_SALES].values
    

    File ~\anaconda3\Lib\site-packages\pandas\core\ops\common.py:76, in _unpack_zerodim_and_defer.<locals>.new_method(self, other)
         72             return NotImplemented
         74 other = item_from_zerodim(other)
    ---> 76 return method(self, other)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\arraylike.py:40, in OpsMixin.__eq__(self, other)
         38 @unpack_zerodim_and_defer("__eq__")
         39 def __eq__(self, other):
    ---> 40     return self._cmp_method(other, operator.eq)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\series.py:5803, in Series._cmp_method(self, other, op)
       5800 lvalues = self._values
       5801 rvalues = extract_array(other, extract_numpy=True, extract_range=True)
    -> 5803 res_values = ops.comparison_op(lvalues, rvalues, op)
       5805 return self._construct_result(res_values, name=res_name)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\ops\array_ops.py:323, in comparison_op(left, right, op)
        318 if isinstance(rvalues, (np.ndarray, ABCExtensionArray)):
        319     # TODO: make this treatment consistent across ops and classes.
        320     #  We are not catching all listlikes here (e.g. frozenset, tuple)
        321     #  The ambiguous case is object-dtype.  See GH#27803
        322     if len(lvalues) != len(rvalues):
    --> 323         raise ValueError(
        324             "Lengths must match to compare", lvalues.shape, rvalues.shape
        325         )
        327 if should_extension_dispatch(lvalues, rvalues) or (
        328     (isinstance(rvalues, (Timedelta, BaseOffset, Timestamp)) or right is NaT)
        329     and lvalues.dtype != object
        330 ):
        331     # Call the method on lvalues
        332     res_values = op(lvalues, rvalues)
    

    ValueError: ('Lengths must match to compare', (15648,), (37,))



```python
store_numbers = filtered_data3['STORE_NBR'].unique()
```


```python
for store in store_numbers:
        if store != STORE_NBR: 
            calculated_measure = pd.DataFrame({
                "Store1": ['STORE_NBR'] * len(filtered_data3[filtered_data3['STORE_NBR'] == store]),
                "Store2": [store] * len(filtered_data3[filtered_data3['STORE_NBR'] == store]),
                "YEARMONTH": filtered_data3[filtered_data3['STORE_NBR'] == store]['YEARMONTH'].values,
                "measure": abs(filtered_data3[filtered_data3['STORE_NBR'] == STORE_NBR][TOT_SALES].values -
                               filtered_data3[filtered_data3['STORE_NBR'] == store][TOT_SALES].values)
            })
    
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[81], line 2
          1 for store in store_numbers:
    ----> 2         if store != STORE_NBR: 
          3             calculated_measure = pd.DataFrame({
          4                 "Store1": ['STORE_NBR'] * len(filtered_data3[filtered_data3['STORE_NBR'] == store]),
          5                 "Store2": [store] * len(filtered_data3[filtered_data3['STORE_NBR'] == store]),
       (...)
          8                                filtered_data3[filtered_data3['STORE_NBR'] == store][TOT_SALES].values)
          9             })
    

    NameError: name 'STORE_NBR' is not defined



```python

```
