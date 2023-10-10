from datetime import date, timedelta
data = str(date.today())
data = data.split("-")
year = float(data[0]) * 365
month = float(data[1]) * 30
day = float(data[2])
date_index =  year + month + day
exp_date_index = date_index + 90
exp_date = date.today() + timedelta(days=90)
exp_date = str(exp_date)
print(exp_date)


