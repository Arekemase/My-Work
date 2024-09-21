```python
import requests
import pandas as pd
```


```python
# Replace 'YOUR-API-KEY' with your actual API key
api_key = 'd6dc0690e27c4dfe970e0951b59dad8e'
url = f'https://openexchangerates.org/api/latest.json?app_id={api_key}&base=USD'

def fetch_exchange_rates(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        rates = response.json().get('rates')
        if rates:
            print("Successfully fetched exchange rates.")
            return rates
        else:
            print("No rates found in the response.")
            return {}
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return {}

# Fetch the exchange rates and print them to verify
exchange_rates = fetch_exchange_rates(url)
print(exchange_rates)

```

    Successfully fetched exchange rates.
    {'AED': 3.67305, 'AFN': 70.353508, 'ALL': 91.742705, 'AMD': 384.569398, 'ANG': 1.786886, 'AOA': 878.351667, 'ARS': 933.25, 'AUD': 1.550597, 'AWG': 1.8, 'AZN': 1.7, 'BAM': 1.791649, 'BBD': 2, 'BDT': 116.496569, 'BGN': 1.78761, 'BHD': 0.376912, 'BIF': 2857.009091, 'BMD': 1, 'BND': 1.320044, 'BOB': 6.851285, 'BRL': 5.7282, 'BSD': 1, 'BTC': 1.8991536e-05, 'BTN': 83.019209, 'BWP': 13.416511, 'BYN': 3.24411, 'BZD': 1.998479, 'CAD': 1.38778, 'CDF': 2815.818837, 'CHF': 0.85239, 'CLF': 0.034221, 'CLP': 944.25818, 'CNH': 7.138325, 'CNY': 7.1368, 'COP': 4041.981181, 'CRC': 518.948757, 'CUC': 1, 'CUP': 25.75, 'CVE': 101.011338, 'CZK': 23.071229, 'DJF': 176.554116, 'DKK': 6.814729, 'DOP': 58.976028, 'DZD': 133.945, 'EGP': 49.4476, 'ERN': 15, 'ETB': 80.28563, 'EUR': 0.91332, 'FJD': 2.261, 'FKP': 0.781538, 'GBP': 0.781538, 'GEL': 2.705, 'GGP': 0.781538, 'GHS': 15.41851, 'GIP': 0.781538, 'GMD': 70.5, 'GNF': 8545.954169, 'GTQ': 7.684063, 'GYD': 207.416501, 'HKD': 7.78145, 'HNL': 24.544032, 'HRK': 6.882194, 'HTG': 130.531975, 'HUF': 363.540834, 'IDR': 16202.75792, 'ILS': 3.82633, 'IMP': 0.781538, 'INR': 83.846299, 'IQD': 1298.813162, 'IRR': 42105, 'ISK': 137.47, 'JEP': 0.781538, 'JMD': 155.116653, 'JOD': 0.7087, 'JPY': 143.48, 'KES': 130, 'KGS': 84.03, 'KHR': 4072.966794, 'KMF': 453.850072, 'KPW': 900, 'KRW': 1368.519286, 'KWD': 0.305328, 'KYD': 0.826211, 'KZT': 470.571031, 'LAK': 22002.172496, 'LBP': 88777.700836, 'LKR': 299.271954, 'LRD': 198.78803, 'LSL': 18.037596, 'LYD': 4.788296, 'MAD': 9.795717, 'MDL': 17.730591, 'MGA': 4518.941675, 'MKD': 56.348398, 'MMK': 2098, 'MNT': 3398, 'MOP': 7.976567, 'MRU': 39.45018, 'MUR': 46.399999, 'MVR': 15.36, 'MWK': 1719.191834, 'MXN': 19.634, 'MYR': 4.4325, 'MZN': 63.850001, 'NAD': 18.037761, 'NGN': 1610.445, 'NIO': 36.495547, 'NOK': 10.993232, 'NPR': 132.828912, 'NZD': 1.687337, 'OMR': 0.384896, 'PAB': 1, 'PEN': 3.710723, 'PGK': 3.896009, 'PHP': 57.882831, 'PKR': 276.1202, 'PLN': 3.929824, 'PYG': 7509.071017, 'QAR': 3.615393, 'RON': 4.5453, 'RSD': 106.885, 'RUB': 85.092617, 'RWF': 1302.729649, 'SAR': 3.754116, 'SBD': 8.490341, 'SCR': 15.022788, 'SDG': 601.5, 'SEK': 10.559357, 'SGD': 1.322745, 'SHP': 0.781538, 'SLL': 20969.5, 'SOS': 566.584201, 'SRD': 28.843, 'SSP': 130.26, 'STD': 22281.8, 'STN': 22.443502, 'SVC': 8.675259, 'SYP': 2512.53, 'SZL': 18.029753, 'THB': 35.2365, 'TJS': 10.484924, 'TMT': 3.5, 'TND': 3.075916, 'TOP': 2.379468, 'TRY': 33.3602, 'TTD': 6.713936, 'TWD': 32.675999, 'TZS': 2700, 'UAH': 40.908559, 'UGX': 3693.628828, 'USD': 1, 'UYU': 40.369388, 'UZS': 12467.920312, 'VES': 36.609871, 'VND': 25106.70982, 'VUV': 118.722, 'WST': 2.8, 'XAF': 599.098921, 'XAG': 0.03553465, 'XAU': 0.00041101, 'XCD': 2.70255, 'XDR': 0.747421, 'XOF': 599.098921, 'XPD': 0.00117355, 'XPF': 108.988117, 'XPT': 0.00107496, 'YER': 250.350066, 'ZAR': 18.4827, 'ZMW': 25.654298, 'ZWL': 322}
    


```python
pd.jason_normalize('exchange_rates')
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[15], line 1
    ----> 1 pd.jason_normalize('exchange_rates')
    

    AttributeError: module 'pandas' has no attribute 'jason_normalize'



```python
    try:
        # Read the Excel file
        df = pd.read_excel(r"C:\Users\Home\Desktop\Important\data Analysis.xlsx", sheet_name='Work')
        print("Successfully read the Excel file.")
        
        # Initialize a list to store the conversion rates
        conversion_rates = []

        # Iterate through the currencies in the Excel sheet
        for currency in df['Currency']:
            rate = exchange_rates.get(currency)
            if rate:
                conversion_rates.append(rate)
            else:
                conversion_rates.append("N/A")  # Handle currencies not found in the API response

        # Add the conversion rates to the DataFrame
        df['Exchange Rate'] = conversion_rates
        
        # Save the updated DataFrame back to the specified sheet in the Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print("Successfully updated the Excel file.")
    
    except Exception as e:
        print(f"Error updating the Excel file: {e}")

# Specify the path to your Excel file and the sheet name
file_path = r'C:\Users\Home\Desktop\Important\data Analysis.xlsx'  
sheet_name = 'Work'  

# Update the Excel sheet and print success message
df.head()
```

    Successfully read the Excel file.
    Successfully updated the Excel file.
    




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
      <th>Location</th>
      <th>Currency</th>
      <th>Audience</th>
      <th>Age Group</th>
      <th>Revenue</th>
      <th>Music_Genre</th>
      <th>Exchange Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Zimbabwe</td>
      <td>ZWL</td>
      <td>45000</td>
      <td>18-35</td>
      <td>20000000</td>
      <td>ZimDancehall/Sungura</td>
      <td>322</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zambia</td>
      <td>ZMW</td>
      <td>25000</td>
      <td>18-35</td>
      <td>10000000</td>
      <td>Kalindula/Hip-Hop</td>
      <td>25.654298</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South Africa</td>
      <td>ZAR</td>
      <td>50000</td>
      <td>18-40</td>
      <td>30000000</td>
      <td>Afrobeat/Jazz</td>
      <td>18.4827</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ivory Coast</td>
      <td>XOF</td>
      <td>40000</td>
      <td>18-35</td>
      <td>20000000</td>
      <td>Coupé-Décalé/Pop</td>
      <td>599.098921</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Senegal</td>
      <td>XOF</td>
      <td>25000</td>
      <td>18-35</td>
      <td>10000000</td>
      <td>Mbalax/Jazz</td>
      <td>599.098921</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
df = pd.read_excel(r"C:\Users\Home\Desktop\Important\data Analysis.xlsx",sheet_name = 'Work')
```


```python
df.head(300)
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
      <th>Location</th>
      <th>Currency</th>
      <th>Audience</th>
      <th>Age Group</th>
      <th>Revenue</th>
      <th>Music_Genre</th>
      <th>Exchange Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Zimbabwe</td>
      <td>ZWL</td>
      <td>45000</td>
      <td>18-35</td>
      <td>20000000</td>
      <td>ZimDancehall/Sungura</td>
      <td>322.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zambia</td>
      <td>ZMW</td>
      <td>25000</td>
      <td>18-35</td>
      <td>10000000</td>
      <td>Kalindula/Hip-Hop</td>
      <td>25.654298</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South Africa</td>
      <td>ZAR</td>
      <td>50000</td>
      <td>18-40</td>
      <td>30000000</td>
      <td>Afrobeat/Jazz</td>
      <td>18.482700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ivory Coast</td>
      <td>XOF</td>
      <td>40000</td>
      <td>18-35</td>
      <td>20000000</td>
      <td>Coupé-Décalé/Pop</td>
      <td>599.098921</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Senegal</td>
      <td>XOF</td>
      <td>25000</td>
      <td>18-35</td>
      <td>10000000</td>
      <td>Mbalax/Jazz</td>
      <td>599.098921</td>
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
    </tr>
    <tr>
      <th>200</th>
      <td>UK</td>
      <td>EUR</td>
      <td>30000</td>
      <td>18-35</td>
      <td>12000000</td>
      <td>Various</td>
      <td>0.913320</td>
    </tr>
    <tr>
      <th>201</th>
      <td>UK</td>
      <td>EUR</td>
      <td>25000</td>
      <td>18-35</td>
      <td>12000000</td>
      <td>Folk/Acoustic</td>
      <td>0.913320</td>
    </tr>
    <tr>
      <th>202</th>
      <td>UK</td>
      <td>EUR</td>
      <td>200000</td>
      <td>18-35</td>
      <td>100000000</td>
      <td>Various</td>
      <td>0.913320</td>
    </tr>
    <tr>
      <th>203</th>
      <td>UK</td>
      <td>EUR</td>
      <td>25000</td>
      <td>18-40</td>
      <td>10000000</td>
      <td>Indie/Folk</td>
      <td>0.913320</td>
    </tr>
    <tr>
      <th>204</th>
      <td>UK</td>
      <td>EUR</td>
      <td>20000</td>
      <td>18-35</td>
      <td>10000000</td>
      <td>Indie</td>
      <td>0.913320</td>
    </tr>
  </tbody>
</table>
<p>205 rows × 7 columns</p>
</div>




```python

```
