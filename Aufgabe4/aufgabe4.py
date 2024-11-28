import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df = pd.read_excel('Aufgabe3\\bev_meld.xlsx')
print(df.columns)

years = df.columns[3:].astype(int)
total_population = df[years].sum()
df_reg = pd.DataFrame({"total_population": total_population.values, "years": years})

model = smf.ols('total_population ~ years', data=df_reg).fit()

print(model.summary())
print("RSQ:", model.rsquared)
print("a:", model.params[1])
print("b:", model.params[0])
print("fittedvalues:", model.fittedvalues)
print("resid:", model.resid)

plt.scatter(df_reg['years'], df_reg['total_population'], label='Einwohner')
plt.plot(df_reg['years'], model.fittedvalues, color='red', label='Regressionsgerade')
plt.title("Regressionsanalyse der Gesamtbevölkerung")
plt.xlabel("Jahre")
plt.ylabel("Gesamtbevölkerung")
plt.legend()
plt.show()

# Interpretation:
# Das Diagramm zeigt einen linearen Anstieg der Gesamtbevölkerung von 1995 bis 2020, was auf ein stetiges Bevölkerungswachstum
# hinweist. Die Regressionsgerade (rot) passt gut zu den Datenpunkten, was eine gleichmäßige Wachstumsrate über die Jahre nahelegt. 
# Der Trend deutet darauf hin, dass die Bevölkerung jedes Jahr leicht zunimmt.

def predict_population(years_to_predict, model):
    future_years = pd.DataFrame({'years': years_to_predict})
    future_years['predicted_population'] = model.predict(future_years)
    return future_years

year_2030 = 2030
predicted_population_2030 = model.params['years'] * year_2030 + model.params[1]
print(f"Prognostizierte Bevölkerung für 2030: {predicted_population_2030:.2f}")

years_to_predict = list(range(2030, 2101))
predictions = predict_population(years_to_predict, model)

print(predictions)

plt.plot(predictions['years'], predictions['predicted_population'], label='Prognose', color='green')
plt.title("Prognose der Gesamtbevölkerung (2030-2100)")
plt.xlabel("Jahre")
plt.ylabel("Prognostizierte Gesamtbevölkerung")
plt.legend()
plt.show()

# Interpretation:
# Die Prognose zeigt, dass die Gesamtbevölkerung von 2033 bis 2100 weiterhin linear ansteigen wird.
# 2030 starten wir mit einer Bevölkerung von ca. 800.000 und erreichen bis 2100 fast 1,1 Millionen Einwohner.


years_to_predict = list(range(2021, 2101))
predictions = predict_population(years_to_predict, model)
plt.scatter(df_reg['years'], df_reg['total_population'], label='Einwohner')
plt.plot(predictions['years'], predictions['predicted_population'], color='green', label='Prognose')
plt.title("Prognose der Gesamtbevölkerung (2030-2100)")
plt.xlabel("Jahre")
plt.ylabel("Prognostizierte Gesamtbevölkerung mit tatsächlichen Werten")
plt.legend()
plt.show()

# Interpretation:
# zeigt die tatsächliche Bevölkerung, kombiniert mit der Prognose.
# Prinzipiell sind es die beiden vorherigen Diagramme kombiniert.

# 3
years = df.columns[3:].astype(int)
df=df[df['Gemeinde']=='Haiming']
total_population = df[years].sum()
df_reg = pd.DataFrame({"total_population": total_population.values, "years": years})

model = smf.ols('total_population ~ years', data=df_reg).fit()

years_to_predict = list(range(2021, 2101))
predictions = predict_population(years_to_predict, model)
plt.scatter(df_reg['years'], df_reg['total_population'], label='Einwohner')
plt.plot(predictions['years'], predictions['predicted_population'], color='green', label='Prognose')
plt.title("Prognose der Gesamtbevölkerung (2030-2100) für Haiming")
plt.xlabel("Jahre")
plt.ylabel("Prognostizierte Gesamtbevölkerung")
plt.legend()
plt.show()

# Interpretation:
# Zeigt den Verlauf der Bevölkerungsentwicklung in Haiming von 1995 bis 2020 und die Prognose von 2021 bis 2100.
# Der Wachstum läuft fast linear und die Prognose zeigt, dass die Bevölkerung von Haiming bis 2100 weiterhin auf über 
# 8.000 Einwohner ansteigen wird.


# 4
df = pd.read_excel('Aufgabe3\\bev_meld.xlsx')
years = df.columns[3:].astype(int)
df_il = df[df['Bezirk'] == 'IL']
df_re = df[df['Bezirk'] == 'RE']


total_population_il = df_il[years].sum(axis=0) 
total_population_re = df_re[years].sum(axis=0) 

df_reg_il = pd.DataFrame({"total_population": total_population_il.values, "years": years})
df_reg_re = pd.DataFrame({"total_population": total_population_re.values, "years": years})

model_il = smf.ols('total_population ~ years', data=df_reg_il).fit()
model_re = smf.ols('total_population ~ years', data=df_reg_re).fit()

years_to_predict = list(range(years.min(), 2101))
prognose_il = pd.DataFrame({"years": years_to_predict})
prognose_il['predicted_population'] = model_il.predict(prognose_il)

prognose_re = pd.DataFrame({"years": years_to_predict})
prognose_re['predicted_population'] = model_re.predict(prognose_re)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

axes[0].scatter(df_reg_il['years'], df_reg_il['total_population'], label='IL: Einwohner', color='blue')
axes[0].plot(prognose_il['years'], prognose_il['predicted_population'], label='IL: Prognose', color='cyan')
axes[0].set_title("Bevölkerungsentwicklung: Bezirk IL")
axes[0].set_xlabel("Jahre")
axes[0].set_ylabel("Gesamtbevölkerung")
axes[0].legend()
axes[0].set_xlim([years.min(), 2100])

axes[1].scatter(df_reg_re['years'], df_reg_re['total_population'], label='RE: Einwohner', color='red')
axes[1].plot(prognose_re['years'], prognose_re['predicted_population'], label='RE: Prognose', color='orange')
axes[1].set_title("Bevölkerungsentwicklung: Bezirk RE")
axes[1].set_xlabel("Jahre")
axes[1].legend()
axes[1].set_xlim([years.min(), 2100])

fig.tight_layout()
plt.show()

# Interpretation:
# In Bezirk Innsbruck-Land wächst die Bevölkerung von rund 150.000 im Jahr 2000 auf über 200.000 und soll laut Prognose
# bis 2100 auf über 250.000 steigen. In Bezirk Reutte liegt die Bevölkerung konstant bei etwa 20.000, mit einem 
# leichten Anstieg in der Prognose.