import numpy as np
import matplotlib.pyplot as plt
   
dataset = [[762., 243., 79., 78., 1542., 88., 21., 409., 14., 993., 25., 8., 1538., 359., 948., 3., 283., 1271., 1391., 542., 314., 1., 36., 64., 629., 536., 20., 318., 1274., 462., 118., 281., 57., 9., 183., 148., 0., 21., 0.], [516., 139., 59., 70., 1065., 82., 33., 245., 17., 711., 29., 11., 1098., 195., 612., 0., 193., 798., 835., 385., 222., 6., 36., 90., 395., 356., 25., 271., 817., 288., 61., 284., 33., 3., 120., 113., 245., 19., 495]]

name_list = np.array(['cad.mention', 'cad.event', 'cad.test', 'cad.symptom', 'diabetes.mention', 'diabetes.alc', 'diabetes.glucose', 'obese.mention', 'obese.bmi', 'hyperlipidemia.mention', 'hyperlipidemia.high-ldl', 'hyperlipidemia.high-chol', 'hypertension.mention', 'hypertension.high-bp', 'medication.ace-inhibitor', 'medication.anti-diabetes', 'medication.arb', 'medication.aspirin', 'medication.beta-blocker', 'medication.calcium-channel-blocker', 'medication.diuretic', 'medication.dpp4-inhibitors', 'medication.ezetimibe', 'medication.fibrate', 'medication.insulin', 'medication.metformin', 'medication.niacin', 'medication.nitrate', 'medication.statin', 'medication.sulfonylureas', 'medication.thiazolidinedione', 'medication.thienopyridine', 'smoker.current', 'smoker.ever', 'smoker.never', 'smoker.past', 'smoker.unknown', 'family-hist.present', 'family-hist.not-present']);
pos_list = np.arange(len(name_list))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(pos_list, dataset[0], color = '.25', width = 0.40)
plt.bar(pos_list + 0.40, dataset[1], color = '.75', width = 0.40)
plt.xticks(pos_list + 0.2, name_list, rotation = 'vertical')

plt.title('Number of Training/Test Data')
plt.ylabel('Number of Instances')

plt.savefig('data-indicators.pdf', format='pdf', bbox_inches='tight')
plt.close()

data = [dataset[0][0:4], dataset[1][0:4]]

name_list = np.array(['mention', 'event', 'test', 'symptom'])
pos_list = np.arange(len(name_list))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(pos_list, data[0], color = '.25', width = 0.40)
plt.bar(pos_list + 0.40, data[1], color = '.75', width = 0.40)
plt.xticks(pos_list + 0.2, name_list)

plt.title('CAD: Number of Training/Test Data')
plt.ylabel('Number of Instances')

plt.savefig('data-cad.pdf', format='pdf', bbox_inches='tight')
plt.close()

data = [dataset[0][4:7], dataset[1][4:7]]

name_list = np.array(['mention', 'alc', 'glucose'])
pos_list = np.arange(len(name_list))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(pos_list, data[0], color = '.25', width = 0.40)
plt.bar(pos_list + 0.40, data[1], color = '.75', width = 0.40)
plt.xticks(pos_list + 0.2, name_list)

plt.title('Diabetes: Number of Training/Test Data')
plt.ylabel('Number of Instances')

plt.savefig('data-diabetes.pdf', format='pdf', bbox_inches='tight')
plt.close()

data = [dataset[0][7:9], dataset[1][7:9]]

name_list = np.array(['mention', 'bmi'])
pos_list = np.arange(len(name_list))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(pos_list, data[0], color = '.25', width = 0.40)
plt.bar(pos_list + 0.40, data[1], color = '.75', width = 0.40)
plt.xticks(pos_list + 0.2, name_list)

plt.title('Obesity: Number of Training/Test Data')
plt.ylabel('Number of Instances')

plt.savefig('data-obese.pdf', format='pdf', bbox_inches='tight')
plt.close()

data = [dataset[0][9:12], dataset[1][9:12]]

name_list = np.array(['mention', 'high-ldl', 'high-chol'])
pos_list = np.arange(len(name_list))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(pos_list, data[0], color = '.25', width = 0.40)
plt.bar(pos_list + 0.40, data[1], color = '.75', width = 0.40)
plt.xticks(pos_list + 0.2, name_list)

plt.title('Hyperlipidemia: Number of Training/Test Data')
plt.ylabel('Number of Instances')

plt.savefig('data-hyperlipidemia.pdf', format='pdf', bbox_inches='tight')
plt.close()

data = [dataset[0][12:14], dataset[1][12:14]]

name_list = np.array(['mention', 'high-bp'])
pos_list = np.arange(len(name_list))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(pos_list, data[0], color = '.25', width = 0.40)
plt.bar(pos_list + 0.40, data[1], color = '.75', width = 0.40)
plt.xticks(pos_list + 0.2, name_list)

plt.title('Hypertension: Number of Training/Test Data')
plt.ylabel('Number of Instances')

plt.savefig('data-hypertension.pdf', format='pdf', bbox_inches='tight')
plt.close()

data = [dataset[0][14:32], dataset[1][14:32]]

name_list = np.array(['ace-inhibitor', 'anti-diabetes', 'arb', 'aspirin', 'beta-blocker', 'calcium-channel-blocker', 'diuretic', 'dpp4-inhibitors', 'ezetimibe', 'fibrate', 'insulin', 'metformin', 'niacin', 'nitrate', 'statin', 'sulfonylureas', 'thiazolidinedione', 'thienopyridine'])
pos_list = np.arange(len(name_list))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(pos_list, data[0], color = '.25', width = 0.40)
plt.bar(pos_list + 0.40, data[1], color = '.75', width = 0.40)
plt.xticks(pos_list + 0.2, name_list, rotation = 'vertical')

plt.title('Medication: Number of Training/Test Data')
plt.ylabel('Number of Instances')

plt.savefig('data-medication.pdf', format='pdf', bbox_inches='tight')
plt.close()

data = [dataset[0][32:37], dataset[1][32:37]]

name_list = np.array(['current', 'ever', 'never', 'past', 'unknown'])
pos_list = np.arange(len(name_list))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(pos_list, data[0], color = '.25', width = 0.40)
plt.bar(pos_list + 0.40, data[1], color = '.75', width = 0.40)
plt.xticks(pos_list + 0.2, name_list)

plt.title('Smoker: Number of Training/Test Data')
plt.ylabel('Number of Instances')

plt.savefig('data-smoker.pdf', format='pdf', bbox_inches='tight')
plt.close()

data = [dataset[0][37::], dataset[1][37::]]

name_list = np.array(['present', 'not-present'])
pos_list = np.arange(len(name_list))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(pos_list, data[0], color = '.25', width = 0.40)
plt.bar(pos_list + 0.40, data[1], color = '.75', width = 0.40)
plt.xticks(pos_list + 0.2, name_list)

plt.title('Family History: Number of Training/Test Data')
plt.ylabel('Number of Instances')

plt.savefig('data-family_hist.pdf', format='pdf', bbox_inches='tight')
plt.close()

data = [[1162, 1651, 423, 1026, 1897, 8491, 397, 21], [784, 1180, 262, 751, 1293, 5674, 514, 514]]

name_list = np.array(['cad', 'diabetes', 'obese', 'hyperlipidemia', 'hypertension', 'medication', 'smoker', 'family-hist']);
pos_list = np.arange(len(name_list))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(pos_list, data[0], color = '.25', width = 0.40)
plt.bar(pos_list + 0.40, data[1], color = '.75', width = 0.40)
plt.xticks(pos_list + 0.2, name_list, rotation = 'vertical')

plt.title('Number of Training/Test Data')
plt.ylabel('Number of Instances')

plt.savefig('data-risks.pdf', format='pdf', bbox_inches='tight')
plt.close()
