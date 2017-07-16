import numpy
import pandas 

def evaluate(tp, fp, fn):
	recall = tp / (tp + fn) if (tp + fn) != 0 else 0
	precision = tp / (tp + fp) if (tp + fp) != 0 else 0
	f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
	return {'Recall': recall, 'Precision': precision, 'F-measure': f1}

def write_to_file(filename, data):
	file = open(filename, 'w')
	file.write("Label,Expected,TP,FP,FN,Recall,Precision,F-measure\n")
	for entry in data:
		file.write("%s\n" % (','.join(str(x) for x in entry)))
	file.close()

indices = [91 ,55 ,94 ,8 ,1 ,30 ,66 ,0 ,20 ,17 ,28 ,68 ,59 ,99 ,2 ,14 ,39 ,29 ,41 ,69 ,21 ,96 ,49 ,95 ,13 ,33 ,61 ,43 ,76 ,80 ,74 ,101 ,5 ,27 ,4 ,42 ,71 ,47 ,11 ,98 ,67 ,88 ,77 ,25 ,52 ,50 ,58 ,9 ,53 ,51 ,38 ,7 ,57 ,81 ,70 ,18 ,37 ,90 ,62 ,26 ,93 ,92 ,82 ,60 ,86 ,78 ,32 ,65 ,54 ,46 ,6 ,36 ,87 ,56 ,15 ,83 ,84 ,72 ,34 ,35 ,63 ,23 ,40 ,79 ,45 ,10 ,64 ,89 ,19 ,73 ,97 ,100 ,31 ,48 ,85 ,75 ,24 ,16 ,44 ,12 ,3 ,22]

results = pandas.read_csv("rnn-performance.csv", header=None).values
results = [results[x] for x in indices]

indices = [7, 10, 11, 14, 16, 22, 23, 26, 27, 28, 30, 34, 36, 37, 38, 42, 43, 46, 47, 54, 55, 60, 61, 62, 85, 86, 88, 89, 93, 105, 117, 132, 134]
entries = [["cad.event.continuing",0,0,0,0,0,0,0], ["cad.test.after_dct",0,0,0,0,0,0,0], ["cad.test.continuing",0,0,0,0,0,0,0],
			["cad.symptom.after_dct",0,0,0,0,0,0,0], ["diabetes.mention.before_dct",0,0,0,0,0,0,0], ["diabetes.a1c.after_dct",0,0,0,0,0,0,0],
			["diabetes.a1c.continuing",0,0,0,0,0,0,0], ["diabetes.glucose.after_dct",0,0,0,0,0,0,0], ["diabetes.glucose.continuing",0,0,0,0,0,0,0],
			["obese.mention.before_dct",0,0,0,0,0,0,0], ["obese.mention.after_dct",0,0,0,0,0,0,0], ["obese.bmi.after_dct",0,0,0,0,0,0,0],
			["hyperlipidemia.mention.before_dct",0,0,0,0,0,0,0], ["hyperlipidemia.mention.during_dct",0,0,0,0,0,0,0], ["hyperlipidemia.mention.after_dct",0,0,0,0,0,0,0],
			["hyperlipidemia.high_ldl.after_dct",0,0,0,0,0,0,0], ["hyperlipidemia.high_ldl.continuing",0,0,0,0,0,0,0], ["hyperlipidemia.high_chol..after_dct",0,0,0,0,0,0,0],
			["hyperlipidemia.high_chol..continuing",0,0,0,0,0,0,0], ["hypertension.high_bp.after_dct",0,0,0,0,0,0,0], ["hypertension.high_bp.continuing",0,0,0,0,0,0,0],
			["medication.anti_diabetes.before_dct",0,0,0,0,0,0,0], ["medication.anti_diabetes.during_dct",0,0,0,0,0,0,0], ["medication.anti_diabetes.after_dct",0,0,0,0,0,0,0],
			["medication.dpp4_inhibitors.during_dct",0,0,0,0,0,0,0], ["medication.dpp4_inhibitors.after_dct",0,0,0,0,0,0,0], ["medication.ezetimibe.before_dct",0,0,0,0,0,0,0],
			["medication.ezetimibe.during_dct",0,0,0,0,0,0,0], ["medication.fibrate.during_dct",0,0,0,0,0,0,0], ["medication.niacin.during_dct",0,0,0,0,0,0,0],
			["medication.sulfonylureas.during_dct",0,0,0,0,0,0,0], ["smoker.unknown",0,0,0,0,0,0,0], ["family_hist.not_present",0,0,0,0,0,0,0]]

for i in range(len(entries)):
	results.insert(indices[i], entries[i])

smoker_expected = 514 - results[128][1] - results[129][1] - results[130][1] - results[131][1]
smoker_fn = results[128][3] + results[129][3] + results[130][3] + results[131][3]
smoker_tp = smoker_expected - smoker_fn
results[132] = [results[132][0], smoker_expected, smoker_tp, 0, smoker_fn, 0, 0, 0]

family_hist_expected = 514 - results[133][1]
family_hist_fn = results[133][3]
family_hist_tp = family_hist_expected - family_hist_fn
results[134] = [results[134][0], family_hist_expected, family_hist_tp, 0, family_hist_fn, 0, 0, 0]

indices = [3 ,7 ,11 ,15 ,19 ,23 ,27 ,31 ,35 ,39 ,43 ,47 ,51 ,55 ,59 ,63 ,67 ,71 ,75 ,79 ,83 ,87 ,91 ,95 ,99 ,103 ,107 ,111 ,115 ,119 ,123 ,127]

for i in range(3, 128, 4):
	results[i-3] = [results[i-3][0], results[i-3][1] + results[i][1], results[i-3][2] + results[i][2], results[i-3][3] + results[i][3], results[i-3][4] + results[i][4], 0, 0, 0]
	results[i-2] = [results[i-2][0], results[i-2][1] + results[i][1], results[i-2][2] + results[i][2], results[i-2][3] + results[i][3], results[i-2][4] + results[i][4], 0, 0, 0]
	results[i-1] = [results[i-1][0], results[i-1][1] + results[i][1], results[i-1][2] + results[i][2], results[i-1][3] + results[i][3], results[i-1][4] + results[i][4], 0, 0, 0]

results = [x for i, x in enumerate(results) if i not in indices]

for i in range(len(results)):
	tp = results[i][2]
	fp = results[i][3]
	fn = results[i][4]
	performance = evaluate(tp, fp, fn)
	results[i][5] = performance['Recall']
	results[i][6] = performance['Precision']
	results[i][7] = performance['F-measure']

results = numpy.array(results)

write_to_file("rnn-performance-dct.csv", results)

micro_tp = numpy.sum(results[:, 2])
micro_fp = numpy.sum(results[:, 3])
micro_fn = numpy.sum(results[:, 4])

micro_performance = evaluate(micro_tp, micro_fp, micro_fn)
print(micro_performance)

indicator_level_results = list()

for i in range(2, 96, 3):
	expected = numpy.sum(results[i-2:i+1, 1])
	tp = numpy.sum(results[i-2:i+1, 2])
	fp = numpy.sum(results[i-2:i+1, 3])
	fn = numpy.sum(results[i-2:i+1, 4])
	performance = evaluate(tp, fp, fn)
	label = results[i][0].split('.')
	label = label[0] + '.' + label[1]
	entry = [label, expected, tp, fp, fn, performance['Recall'], performance['Precision'], performance['F-measure']]
	indicator_level_results.append(entry)

for i in range(96, 103):
	tp = results[i][2]
	fp = results[i][3]
	fn = results[i][4]
	performance = evaluate(tp, fp, fn)
	entry = [results[i][0], results[i][1], tp, fp, fn, performance['Recall'], performance['Precision'], performance['F-measure']]
	indicator_level_results.append(entry)

write_to_file("rnn-performance-indicator.csv", indicator_level_results)

risk_level_results = list()

indices = [(0, 12), (12, 21), (21, 27), (27, 36), (36, 42), (42, 96), (96, 101), (101, 103)]

for i in indices:
	expected = numpy.sum(results[i[0]:i[1], 1])
	tp = numpy.sum(results[i[0]:i[1], 2])
	fp = numpy.sum(results[i[0]:i[1], 3])
	fn = numpy.sum(results[i[0]:i[1], 4])
	performance = evaluate(tp, fp, fn)
	label = results[i[0]][0].split('.')
	label = label[0]
	entry = [label, expected, tp, fp, fn, performance['Recall'], performance['Precision'], performance['F-measure']]
	risk_level_results.append(entry)

write_to_file("rnn-performance-risk.csv", risk_level_results)
