
# collate metric results into csv file

df = pd.DataFrame(res, index=[f'CV{i+1}' for i in range(len(res['acc']))])
df.loc['average'] = df.mean(axis=0)
#df['auc'] = [1- x for x in df2['auc']] 
print(df)
#rename = {'0': 'CV1','1': 'CV2','2': 'CV3','3': 'CV4','4': 'CV5','5': 'average'}
#df_metric.rename(index = {str(len(res['acc'])) : 'average'})

metric_col = ['acc', 'f_score', 'auc', 'auprc']
df_metric = df[metric_col]

print(df_metric.columns)

fprs = []
tprs = []
mean_fpr = np.linspace(0, 1, 100)

for i in range(len(df['roc'])-1):
    fpr_,tpr_ = df['roc'][i]
    tprs.append(np.interp(mean_fpr, fpr_, tpr_))

    
mean_tpr = 1-np.mean(tprs, axis=0)
mean_tpr[0] = 0
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)


precisions = []
mean_recall = np.linspace(0, 1, 100)

for i in range(len(df['prc'])-1):
    recall,precision = df['prc'][i]
    precisions.append(np.interp(mean_recall, recall,precision))

mean_precision = np.mean(precisions, axis=0)
std_precision = np.std(precisions, axis=0)
precisions_upper = np.minimum(mean_precision + std_precision, 1)
precisions_lower = np.maximum(mean_precision - std_precision, 0)


#+ 'selected clinical_data'  ''_optimal clinical_' + '_'+
fn = (options['classifier'] +'+' + '_bayesian search_' +  options['kernel'] +'_' + 
      str(category) + '_' + pred_level +'.xlsx')
print(fn)

writer = pd.ExcelWriter(fn, engine = 'xlsxwriter')
df_metric.to_excel(writer, sheet_name = 'metric', float_format = '%.2f')

#append confusion matrix and ROC/PRC curve values
df_c = pd.DataFrame(df['confusion'].sum(), columns = ['N', 'P'], index = ['N', 'P'])#retrieve confusion matrix
df_d = np.transpose(pd.DataFrame([mean_fpr, mean_tpr, mean_recall, mean_precision])) #to plot ROC/PRC curve in excel
df_c.to_excel(writer, sheet_name = 'confusion matrix', float_format = '%.0f')
df_d.to_excel(writer, sheet_name = 'curve', float_format = '%.2f')
writer.save()
writer.close()

    
#print out ROC 
plt.figure(0)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(mean_fpr, mean_tpr, label = 'area = {:.3f}'.format(df_metric['auc']['average']))
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(fn[:-5] + 'roc' +'.jpg')
plt.show()

#print out PRC curves
plt.figure(0)
plt.plot(mean_recall, mean_precision, label = 'area = {:.3f}'.format(df_metric['auprc']['average']))
plt.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PRC curve')
plt.legend(loc='best')
plt.savefig(fn[:-5] + 'prc' +'.jpg')
plt.show()

model_name = fn[:-4] + 'sav'
pickle.dump(model, open(model_name, 'wb'))
