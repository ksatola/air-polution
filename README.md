# air-pollution
This is work in progress, for the recent status, see [phase 1 summary](https://github.com/ksatola/air-pollution/blob/master/phase01/air-pollution-phase-01.ipynb). Phase 2 is about to start soon.


## TODO:

- odswiezyc dane
- dodac info na temat configu srodowiska dla Windows (w notebooku 010)
- znalezc TODO i dodac notebooki + ulepszyc kod i zautomatyzowac (zaczynajac od sekcji Data Preparation)
- describe folders structure and modules of the project
- Put link to phase 1 study (copy from satola.net)
- zrobic nowy wpis w satola.net i zlinkowac go z tym readme (jako data projekt - tool for analysis and forecasting)

phase 1 - to co a agh
phase 2 - dodatkowe badania (metody, deep learning) + aplikacja EDA interactive + rozszerzyc na pozostale 3 pollutants (zob. problem description)
phase 3 - forecasting tool with using API for current forecasts + deployed publicly

The ultimate goal of this research is creation of:

- An air pollution level prediction tool providing 24 hours and 7 days forecasts of particulate Matter (PM), nitrogen dioxide (NO2), sulphur dioxide (SO2) and ground-level ozone (O3) in an easy to understand and use form (web application) to be used for planning open air activities.
- A long-term monitoring tool of air pollutants levels and trends to see if the improvement is being made in the fight for fresh air.

- potem zrobic data product (web app)
- liste powiazanych artykulow (notebooki) poruszajacych specyficzne tematy na bazie wspolnego kodu oraz danych



- Use cross validation to ensure the results are less dependent on a specific point in time split between past and future.
- Use API for up-to-date data access for real-time prediction.
- Consider external influencers on air-pollution, like global climate change, new and environment-friendly heating stoves, local community efforts to improve air conditions.
- Consider sensors' specification to check quality regarding their measurement ranges, sensivity and distributions.
- Perform multivariate analysis on climate and other air pollutants data.
- Augment current data sets with 2019/2020 data points.
- Build a web application for easier use of prediction models.
- Build on top of existing models and modelling techniques (add new variables, try different techniques and algorithms) and improve incrementally predictive performance.
- Continue research on existing tools and specilised packets for advances machine learning (mlens, prophet, etc.) to see if different techniques and algorithms could give better results on PM2.5 forecasting.
- Think about the problem as classificiation problem - we do not need to know the exact value of PM2.5 but if we are within acceptable range of these values

Modelling
- Check, which features (like t-1, t-2, t-30...) are statistically 
- sprawdzic OLS linear regression assumptions
- For TS apply cross validation method for testing (use different time ranges to choose a 24 and 7 test datasets and canculate mean RMSE for all of them)
- Add a trend, either directly or as a summary, as a new input variable to the supervised learning problem to predict the output variable.
- Return to SARIMAX and apply validation function with Sesonar ARIMA model (050b_final_Model_PM25_04_SARIMA.ipynb)

## Appendices

### English - Polish Vocabulary (temporary)

- carbon black (sadza)
- electrical precipitator (filtr elektrostatyczny)

- Reproducible research is the idea that data analyses, and more generally, scientific claims, are published with their data and software code so that others may verify the findings and build upon them.

## Resources

### Natural Environment

1. [Air pollution: Our health still insufficiently protected](https://op.europa.eu/webpub/eca/special-reports/air-quality-23-2018/en/)
1. [European Environment Agency (EEA): Air Pollution](https://www.eea.europa.eu/themes/air)
1. [The European environment – state and outlook 2020](http://www.gios.gov.pl/en/eea/highlights-eea-nfp-pl/649-the-european-environment-state-and-outlook-2020)
1. [LIFE Integrated Project "Implementation of Air Quality Plan for Małopolska Region – Małopolska in a healthy atmosphere"](https://powietrze.malopolska.pl/en/life-project/)
1. [Review of evidence on health aspects of air pollution – REVIHAAP Project (Technical Report)](http://www.euro.who.int/__data/assets/pdf_file/0004/193108/REVIHAAP-Final-technical-report-final-version.pdf?ua=1)
1. [Zanieczyszczenie powietrza (Krakow)](https://powietrze.malopolska.pl/baza/jakosc-powietrza-w-polsce-na-tle-unii-europejskiej/)

### Time Series Forecasting
1. [Forecasting: Principles and Practice](https://otexts.com/fpp2/)
1. [Forecasting Time Series Data using Autoregression](https://pythondata.com/forecasting-time-series-autoregression/)

### Machine Learning
1. [Super Learner](https://www.degruyter.com/view/journals/sagmb/6/1/article-sagmb.2007.6.1.1309.xml.xml)
1. [Super Learner In Prediction](https://biostats.bepress.com/ucbbiostat/paper266/)
1. [How to Develop Super Learner Ensembles in Python](https://machinelearningmastery.com/super-learner-ensemble-in-python/)
1. [ML-Ensemble](https://mlens.readthedocs.io/)


