import numpy as np
from matplotlib import pyplot as plt

d = np.genfromtxt('Aufgabe2\london_weather.csv', delimiter=",", skip_header=1 )

dt =  d[:,0] 

day = (dt % 100).astype('i')
month = (dt % 10000 / 100).astype('i')
year = (dt % 100000000 / 10000).astype('i')

temp = d[:,5]

temp1979 = temp[year == 1979]
temp1980 = temp[year == 1980]
temp1981 = temp[year == 1981]
temp1982 = temp[year == 1982]

# 1.1
# Interpretation: Diese Visualisierung zeigt die Verteilung der Temperaturen (Median, Quartile, Extremwerte) für jedes Jahr.
# besonders fällt auf dass die meisten Temeraturen zwischen 5 und 15 Grad liegen.
plt.boxplot([temp1979, temp1980, temp1981, temp1982], labels=['1979', '1980', '1981', '1982'])
plt.show()

# 1.2
# Interpretation: Dieser Plot zeigt die täglichen Temperaturschwankungen im Jahr 1979,
# wobei die meisten Temperaturen zwischen 0°C und 20°C liegen, mit einigen wenigen Ausreißern unter 0°C.
plt.scatter(day[year == 1979], temp1979)
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Temperature in 1979')
plt.show()

# 1.3
extrema_low_1979 = np.quantile(temp1979, 0.05)
extrema_high_1979 = np.quantile(temp1979, 0.95)
extrema_low_2020 = np.quantile(temp1980, 0.05)
extrema_high_2020 = np.quantile(temp1980, 0.95)
print(f'Extrema 1979: {extrema_low_1979} - {extrema_high_1979}')
print(f'Extrema 2020: {extrema_low_2020} - {extrema_high_2020}')
# Output:
# Extrema 1979: 0.07999999999999972 - 18.88 °C
# Extrema 2020: 2.025 - 18.275 °C
# Interpretation: Die Extremwerte für 1979 und 2020 sind sehr ähnlich, wobei die niedrigsten Temperaturen bei 0°C und die höchsten bei 18°C liegen.

# 1.4
# Interpretation: Durchschnittstemperatur für jedes Jahr von 2010 bis 2020
# 2010 und 2013 waren die Durchschittstemperaturen am niedrigsten, 2014 am höchsten. 
# Allgemein liegt sie zwischen 10.5 und 12.5 Grad.
for i in range(2010, 2021):
    temp_year = temp[year == i]
    plt.bar(i, np.mean(temp_year))
    plt.xticks(np.arange(2010, 2020))
plt.show()

# 1.5 alle Durchschnittstemperaturen für jeden Monat im Jahr 1980
# Interpretation: Man sieht eindeutig, dass es 1980 in den Sommermonaten wärmer war als in den Wintermonaten.
# Im März war es jedoch im Durchschnitt kälter als im Feber.
# Die höchste Durchschnittstemperatur wurde im August gemessen mit ca. 17,8 °C.
avg_temp = []
months = np.arange(1, 13)
for m in months:
    avg_temp.append(np.mean(temp[(year == 1980) & (month == m)]))

plt.plot(months, avg_temp, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Temperature')
plt.xticks(months)
plt.grid(True)
plt.show()
