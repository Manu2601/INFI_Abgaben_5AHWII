import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Einlesen des Datensatzes
df = pd.read_excel('Aufgabe3/Zeitreihe-Winter-2024011810.xlsx')

# Spalten umbenennen, um Probleme mit Jahreszahlen zu vermeiden
base = ['Bezirk', 'Gemnr', 'Gemeinde']
years = df.columns[3:].astype(str)
base.extend('x' + years)
df.columns = base
print(years)


# 2.1 Wachstum in Innsbruck darstellen
# Zahlenwerte zu den einzelnen Jahren in Innsbruck extrahieren
innsbruck = df[df.Bezirk == 'I'].values[0, 3:]  # Beachte: 2D-Array trotz einer Zeile

jahre = df.columns[3:]  # Spaltennamen der Jahre
# Zeitlichen Verlauf als Punktdiagramm
plt.scatter(jahre, innsbruck, marker='x')
plt.xticks(rotation=90)
plt.xlabel('Jahre')
plt.ylabel('Übernachtungen')
plt.title('Zeitlicher Verlauf der Übernachtungen in Innsbruck')
plt.show()
# Interpretation: Die Grafik zeigt einen Anstieg der Übernachtungen in Innsbruck von 2000 bis etwa 2019,
# gefolgt von einem starken Rückgang um 2020 (durch Corona). Danach ist eine leichte Erholung sichtbar.



# 2.2 Wachstum des eigenen Bezirks
bezirk = 'IM'  

bezirk_data = df[df['Bezirk'] == bezirk]
bezirk_sum = bezirk_data.iloc[:, 3:].sum(axis=0)

# Ausgabe der Zahlenwerte in der Konsole
print(f"Summierte Werte für Bezirk {bezirk}:")
print(bezirk_sum)

# Zeitlichen Verlauf als Liniendiagramm darstellen
plt.plot(years, bezirk_sum, linestyle='-', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Jahre')
plt.ylabel('Summierte Werte')
plt.title(f'Wachstum im Bezirk {bezirk} über die Jahre')
plt.show()

# Interpretatoion: zeigt dass die summierten Werte vom Bezirk Imst bis 2019 gewachsen sind.
# 2020 - 2021 ist ein starker Rückgang zu erkennen, was auf die Corona-Pandemie zurückzuführen ist.
# danach steigen die Werte wieder an, fast auf das Niveau von 2019.


# 3.1 Min, Max, Range, Avg
# Berechnung der Statistiken

print(df.iloc[:, 3:].dtypes)

df['min'] = df.iloc[1:, 3:].min(axis=1)
df['max'] = df.iloc[1:, 3:].max(axis=1)       # Maximum pro Gemeinde
df['range'] = df['max'] - df['min']          # Range (max - min)
df['avg'] = df.iloc[:, 3:].mean(axis=1)      # Durchschnitt pro Gemeinde

print(df[['Gemeinde', 'min', 'max', 'range', 'avg']])

# 3.1.1 
# Standardisierung der Werte
standard = (df['range'] - df['range'].mean()) / df['range'].std()


# 3.2
yearly_sum = df.iloc[:, 3:].sum(axis=0)  # Start bei der 3. Spalte für die Jahreszahlen
print("Gesamtzahl der Touristen pro Jahr:\n", yearly_sum)

# Gesamtzahl der Touristen über alle Jahre
total_sum = yearly_sum.sum()
print("Gesamtzahl der Touristen über alle Jahre:", total_sum)

# Zusammenfassung nach Bezirken (Gesamtzahl der Touristen je Bezirk über alle Jahre)
sum_bez = df.groupby('Bezirk').sum(numeric_only=True)
print("Zusammenfassung der Gesamtzahl der Touristen nach Bezirk:\n", sum_bez)

# Plotten der Zusammenfassung nach Bezirk als Balkendiagramm
sum_bez.sum(axis=1).plot.bar() 
plt.ylabel('Gesamtzahl der Touristen')
plt.title('Touristen nach Bezirk')
plt.show()

# Interpretation
# Bezirk Innsbruck hat die wenigsten Touristen, während Bezirk LA die meisten Touristen hat.
# An zweiter Steille ist Salzburg.


# 4.1
# Methode a: Boxplot mit pandas
df.boxplot(column='range', by='Bezirk')
plt.title('Boxplot der standardisierten Ranges nach Bezirk (Pandas)')
plt.suptitle('')  # Entfernt den automatischen Titel von pandas
plt.xlabel('Bezirk')
plt.ylabel('Standardisierte Range')
plt.show()

# Interpretation:
# Der Boxplot zeigt die Verteilung der standardisierten Ranges für die verschiedenen Bezirke.
# Bezirk Imst und Landeck haben relativ große Ausreißer, Innsbruck und Lienz haben die geringste Streuung.
# Die größte Standardabweichungen haben Kitzbühel und Landeck.


# Methode b: Boxplot mit matplotlib (manuell gruppiert)
labels = df['Bezirk'].unique()
pos = 0
for label in labels:
    bez_data = df[df['Bezirk'] == label]
    plt.boxplot(bez_data['range'], positions=[pos])
    pos += 1
plt.xticks(range(len(labels)), labels)
plt.title('Boxplot der standardisierten Ranges nach Bezirk (Matplotlib)')
plt.xlabel('Bezirk')
plt.ylabel('Standardisierte Range')
plt.show()

# Methode c: Boxplot mit Seaborn
sns.boxplot(x='Bezirk', y='range', data=df, palette="Set2")
plt.title('Boxplot der standardisierten Ranges nach Bezirk (Seaborn)')
plt.xlabel('Bezirk')
plt.ylabel('Standardisierte Range')
plt.show()

# 4.2 Innsbruck Jahreswerte
sns.barplot(x=years, y=innsbruck, palette='terrain')
plt.xticks(rotation=70)
plt.xlabel("Jahre")
plt.ylabel("Übernachtungen")
plt.title("Jährliche Übernachtungen in Innsbruck")
plt.show()

# Interpretation:
# Der Balkendiagramm zeigt die jährlichen Übernachtungen in Innsbruck. Es ist ein starker Anstieg von 2000 bis 2019.
# 2020 ist ein starker Rückgang zu erkennen, was auf die Corona-Pandemie zurückzuführen ist.
# danach steigen die Werte wieder an, fast auf das Niveau von 2019.
# der Peak war 2019 mit fast 800.000 Übernachtungen.

# 5 Gegenüberstellung mit den Einwohnerzahlen
df_population = pd.read_excel('Aufgabe3\\bev_meld.xlsx')
# a)
df = pd.merge(df, df_population, how='inner', on='Gemnr')
# doppelte Spalten entfernen
df = df.drop(columns=['Gemnr', 'Gemeinde_y', 'Bezirk_y'])

df = df.rename(columns={'Gemeinde_x': 'Gemeinde', 'Bezirk_x': 'Bezirk'})
year_column_2018 = '2018' if '2018' in df.columns else 'x2018'
df['tourist_person_2018'] = df[year_column_2018] / df[2018]

# b)
sns.boxplot(x='Bezirk', y='tourist_person_2018', data=df, palette="Set2")
plt.title('Touristen pro Einwohner im Jahr 2018 nach Bezirk')
plt.xlabel('Bezirk')
plt.ylabel('Touristen pro Einwohner (2018)')
plt.show()

# Interpretation:
# Der Boxplot zeigt, dass die Bezirke LA und SZ die höchsten Werte und viele Ausreißer bei Touristen pro Einwohner im Jahr 2018 aufweisen,
# was auf touristische Hotspots hindeutet. Bezirke wie IM, IL, und KU haben hingegen niedrige und stabile Werte, was auf weniger
# touristische Aktivität schließen lässt. Insgesamt zeigt der Plot eine große Variabilität zwischen den Bezirken, wobei einige
# Regionen viel stärker vom Tourismus betroffen sind als andere.

# c) größte Verhältniszahl
df_high = df.sort_values('x2018', ascending=False).head(10)
print(df_high)

# kleinste Verhältniszahl
df_low = df.sort_values('x2018', ascending=True).head(10)
print(df_low)

# d) Verhältnis in der Heimatgemeinde - Haiming
haiming = df[df['Gemeinde'].str.contains('Haiming', case=False, na=False)]