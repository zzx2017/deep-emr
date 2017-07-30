import pandas
import numpy as np
import matplotlib.pyplot as plt

f1 = [0.9276, 0.9268, 0.9185, 0.9171, 0.9156, 0.9081, 0.8973, 0.8909, 0.8776, 0.8747, 0.8798, 0.8900, 0.9026, 0.9006, 0.9081]

models = np.array(['NLM', 'Harbin-Grad', 'Kaiser', 'Linguamatics', 'Nottingham', 'Ohio', 'TMUNSW', 'NCU', 'UNIMAN', 'Utah', 'CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
model_positions = np.arange(len(models))

plt.bar(model_positions[0:10], f1[0:10], color = '.75')
plt.bar(model_positions[10::], f1[10::], color = '.50')
plt.xticks(model_positions, models, rotation = 'vertical')

plt.title('Comparison of System Performance')
plt.xlabel('Teams/Models')
plt.ylabel('F1')
plt.show()

f1 = [0.8798, 0.8900, 0.9026, 0.9006, 0.9081, 0.815, 0.9276]

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM', '2014 i2b2 (Mean)', '2014 i2b2 (Max)'])
model_positions = np.arange(len(models))

plt.bar(model_positions[0:5], f1[0:5], color = '.50')
plt.bar(model_positions[5::], f1[5::], color = '.75')
plt.xticks(model_positions, models, rotation = 'vertical')

plt.title('Comparison of System Performance')
plt.xlabel('Models')
plt.ylabel('F1')
plt.show()

results_overall = pandas.read_csv('results-overall.csv', header=None).values

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
positions = np.arange(len(models))

overall = results_overall[0, 1::]
cad = results_overall[1, 1::]
diabetes = results_overall[2, 1::]
obese = results_overall[3, 1::]
hyperlipidemia = results_overall[4, 1::]
hypertension = results_overall[5, 1::]
medication = results_overall[6, 1::]
smoker = results_overall[7, 1::] 
family_hist = results_overall[8, 1::]

plt.scatter(positions, overall, color = '.5', marker = '*', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Overall')
plt.scatter(positions, cad, color = '.5', marker = 's', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'CAD')
plt.scatter(positions, diabetes, color = '.5', marker = '^', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Diabetes')
plt.scatter(positions, obese, color = '.5', marker = '.', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Obese')
plt.scatter(positions, hyperlipidemia, color = '.5', marker = '<', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Hyperlipidemia')
plt.scatter(positions, hypertension, color = '.5', marker = 'h', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Hypertension')
plt.scatter(positions, medication, color = '.5', marker = 'd', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Medication')
plt.scatter(positions, smoker, color = '.5', marker = '8', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Smoker')
plt.scatter(positions, family_hist, color = '.5', marker = '>', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Family History')
plt.xticks(positions, models)

plt.title('F1 on Individual Categories')
plt.xlabel('Models')
plt.ylabel('F1')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()

results_cad = pandas.read_csv('results-cad.csv', header=None).values

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
positions = np.arange(len(models))

overall = results_cad[0, 1::]
mention = results_cad[1, 1::]
event = results_cad[2, 1::]
test = results_cad[3, 1::]
symptom = results_cad[4, 1::]

plt.scatter(positions, overall, color = '.5', marker = '*', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Overall')
plt.scatter(positions, mention, color = '.5', marker = '.', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Mention')
plt.scatter(positions, event, color = '.5', marker = '<', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Event')
plt.scatter(positions, test, color = '.5', marker = 'h', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Test')
plt.scatter(positions, symptom, color = '.5', marker = 'd', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Symptom')
plt.xticks(positions, models)

plt.title('CAD Indicator Breakdown by F1')
plt.xlabel('Models')
plt.ylabel('F1')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()

results_diabetes = pandas.read_csv('results-diabetes.csv', header=None).values

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
positions = np.arange(len(models))

overall = results_diabetes[0, 1::]
mention = results_diabetes[1, 1::]
alc = results_diabetes[2, 1::]
glucose = results_diabetes[3, 1::]

plt.scatter(positions, overall, color = '.5', marker = '*', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Overall')
plt.scatter(positions, mention, color = '.5', marker = 'd', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Mention')
plt.scatter(positions, alc, color = '.5', marker = '<', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'AlC')
plt.scatter(positions, glucose, color = '.5', marker = '.', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Glucose')
plt.xticks(positions, models)

plt.title('Diabetes Indicator Breakdown by F1')
plt.xlabel('Models')
plt.ylabel('F1')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()

results_obese = pandas.read_csv('results-obese.csv', header=None).values

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
positions = np.arange(len(models))

overall = results_obese[0, 1::]
mention = results_obese[1, 1::]
bmi = results_obese[2, 1::]

plt.scatter(positions, overall, color = '.5', marker = '*', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Overall')
plt.scatter(positions, mention, color = '.5', marker = '.', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Mention')
plt.scatter(positions, bmi, color = '.5', marker = '<', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'BMI')
plt.xticks(positions, models)

plt.title('Obesity Indicator Breakdown by F1')
plt.xlabel('Models')
plt.ylabel('F1')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()

results_hyperlipidemia = pandas.read_csv('results-hyperlipidemia.csv', header=None).values

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
positions = np.arange(len(models))

overall = results_hyperlipidemia[0, 1::]
mention = results_hyperlipidemia[1, 1::]
high_ldl = results_hyperlipidemia[2, 1::]
high_chol = results_hyperlipidemia[3, 1::]

plt.scatter(positions, overall, color = '.5', marker = '*', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Overall')
plt.scatter(positions, mention, color = '.5', marker = 'd', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Mention')
plt.scatter(positions, high_ldl, color = '.5', marker = '.', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'High LDL')
plt.scatter(positions, high_chol, color = '.5', marker = '<', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'High Cholesterol')
plt.xticks(positions, models)

plt.title('Hyperlipidemia Indicator Breakdown by F1')
plt.xlabel('Models')
plt.ylabel('F1')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()

results_hypertension = pandas.read_csv('results-hypertension.csv', header=None).values

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
positions = np.arange(len(models))

overall = results_hypertension[0, 1::]
mension = results_hypertension[1, 1::]
high_bp = results_hypertension[2, 1::]

plt.scatter(positions, overall, color = '.5', marker = '*', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Overall')
plt.scatter(positions, mension, color = '.5', marker = '.', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Mension')
plt.scatter(positions, high_bp, color = '.5', marker = '<', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'High BP')
plt.xticks(positions, models)

plt.title('Hypertension Indicator Breakdown by F1')
plt.xlabel('Models')
plt.ylabel('F1')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()

results_medication = pandas.read_csv('results-medication.csv', header=None).values

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
positions = np.arange(len(models))

overall = results_medication[0, 1::]
ace_inhibitor = results_medication[1, 1::]
anti_diabetes = results_medication[2, 1::]
arb = results_medication[3, 1::]
aspirin = results_medication[4, 1::]
beta_blocker = results_medication[5, 1::]
calcium_channel_blocker = results_medication[6, 1::]
diuretic = results_medication[7, 1::]
dpp4_inhibitors = results_medication[8, 1::]
ezetimibe = results_medication[9, 1::]
fibrate = results_medication[10, 1::]
insulin = results_medication[11, 1::]
metformin = results_medication[12, 1::]
niacin = results_medication[13, 1::]
nitrate = results_medication[14, 1::]
statin = results_medication[15, 1::]
sulfonylureas = results_medication[16, 1::]
thiazolidinedione = results_medication[17, 1::]
thienopyridine = results_medication[18, 1::]

plt.scatter(positions, overall, color = '.5', marker = '*', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Overall')
plt.scatter(positions, ace_inhibitor, color = '.5', marker = 's', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'ACE Inhibitor')
plt.scatter(positions, anti_diabetes, color = '.5', marker = '8', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Anti Diabetes')
plt.scatter(positions, arb, color = '.5', marker = '>', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'ARB')
plt.scatter(positions, aspirin, color = '.5', marker = '<', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Aspirin')
plt.scatter(positions, beta_blocker, color = '.5', marker = '^', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Beta Blocker')
plt.scatter(positions, calcium_channel_blocker, color = '.5', marker = 'v', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Calcium Channel Blocker')
plt.scatter(positions, diuretic, color = '.5', marker = 'o', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Diuretic')
plt.scatter(positions, dpp4_inhibitors, color = '.5', marker = 'X', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'DPP4 Inhibitors')
plt.scatter(positions, ezetimibe, color = '.5', marker = 'P', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Ezetimibe')
plt.scatter(positions, fibrate, color = '.5', marker = 'd', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Fibrate')
plt.scatter(positions, insulin, color = '.5', marker = 'D', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Insulin')
plt.scatter(positions, metformin, color = '.5', marker = 'H', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Metformin')
plt.scatter(positions, niacin, color = '.5', marker = 'h', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Niacin')
plt.scatter(positions, nitrate, color = '.5', marker = 'p', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Nitrate')
plt.scatter(positions, statin, color = '.5', marker = '.', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Statin')
plt.scatter(positions, sulfonylureas, color = '.25', marker = '+', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Sulfonylureas')
plt.scatter(positions, thiazolidinedione, color = '.25', marker = 'x', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Thiazolidinedione')
plt.scatter(positions, thienopyridine, color = '.25', marker = '1', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Thienopyridine')
plt.xticks(positions, models)

plt.title('Medication Indicator Breakdown by F1')
plt.xlabel('Models')
plt.ylabel('F1')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()

results_smoker = pandas.read_csv('results-smoker.csv', header=None).values

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
positions = np.arange(len(models))

overall = results_smoker[0, 1::]
current = results_smoker[1, 1::]
ever = results_smoker[2, 1::]
never = results_smoker[3, 1::]
past = results_smoker[4, 1::]
unknown = results_smoker[5, 1::]

plt.scatter(positions, overall, color = '.5', marker = '*', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Overall')
plt.scatter(positions, current, color = '.5', marker = '<', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Current')
plt.scatter(positions, ever, color = '.5', marker = '.', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Ever')
plt.scatter(positions, never, color = '.5', marker = 'd', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Never')
plt.scatter(positions, past, color = '.5', marker = 'h', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Past')
plt.scatter(positions, unknown, color = '.5', marker = 's', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Unknown')
plt.xticks(positions, models)

plt.title('Smoker Indicator Breakdown by F1')
plt.xlabel('Models')
plt.ylabel('F1')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()

results_family_hist = pandas.read_csv('results-family_hist.csv', header=None).values

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
positions = np.arange(len(models))

overall = results_family_hist[0, 1::]
present = results_family_hist[1, 1::]
not_present = results_family_hist[2, 1::]

plt.scatter(positions, overall, color = '.5', marker = '*', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Overall')
plt.scatter(positions, present, color = '.5', marker = '.', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Present')
plt.scatter(positions, not_present, color = '.5', marker = '<', edgecolor = '.25', linewidth = '0.3', alpha = '.5', label = 'Not Present')
plt.xticks(positions, models)

plt.title('Family History Indicator Breakdown by F1')
plt.xlabel('Models')
plt.ylabel('F1')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()
