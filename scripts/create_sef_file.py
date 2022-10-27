
import SEF

records = {
           'ID': 'DWR_Zarate', 'Name': 'Zarate', 'Source': 'C3S_SouthAmerica', 
           'Lat': -34.0958, 'Lon': -59.0242,
           'Link': 'https://data-rescue.copernicus-climate.eu/lso/1086267', 
           'Vbl': 'mslp', 'Stat': 'point',
           'Units': 'Pa', 'Meta': 'Data policy=GNU GPL v3.0', 
           'Year': [1902, 1902], 'Month': [9, 9], 'Day': [2, 3],
           'Hour': [11, 11], 'Minute': [17, 17], 'Value': [102245, 101952], 
           'Period': [0, 0], 
           'Meta2': ['Orig=766.9mm|Orig.time=', 'Orig.=764.7mm|Orig.time='], 
           'orig_time': ['7am', '7am']
}

obs = SEF.create(records)

SEF.write_file(obs, '/home/users/pmcraig/output.tsv')