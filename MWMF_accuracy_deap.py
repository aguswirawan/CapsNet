import xlrd
import pandas as pd
import numpy as np

subjects = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
            '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'] # For Dominance, the 's27' should be removed.
dataset_name = 'deap' # the name of dataset: deap or dreamer
model_version = 'v1' # the version of model
epoch = '40' # v0 20; v2 40
debase = 'yes' # with or without debaseline

summary_valence = pd.DataFrame()
summary_arousal = pd.DataFrame()
summary_dominance = pd.DataFrame()

for label in ['all']: # ,'arousal','dominance']:#['valence','arousal','dominance']:
    test_accuracy_allsub = np.zeros(shape=[0], dtype=float)
    test_loss_allsub = np.zeros(shape=[0], dtype=float)
    train_time_allsub = np.zeros(shape=[0], dtype=float)
    test_time_allsub = np.zeros(shape=[0], dtype=float)
    epoch_allsub = np.zeros(shape=[0], dtype=float)
    lr_allsub = np.zeros(shape=[0], dtype=float)
    batch_size_allsub = np.zeros(shape=[0], dtype=float)
    for sub in subjects:
        print(sub)
        xl = xlrd.open_workbook(r'result_MWMF_div/sub_dependent_v0/yes/s'+sub+'_'+label+epoch+'/summary_s'+sub+'.xlsx')
        table = xl.sheets()[1]
        acc = table.cell(1,0).value
        loss = table.cell(1,1).value
        train_time = table.cell(1,2).value
        test_time = table.cell(1, 3).value
        epochs = table.cell(1, 4).value
        lr = table.cell(1, 5).value
        batch_size = table.cell(1, 6).value
        
        

        test_accuracy_allsub = np.append(test_accuracy_allsub, acc)
        test_loss_allsub = np.append(test_loss_allsub, loss)
        train_time_allsub = np.append(train_time_allsub, train_time)
        test_time_allsub =np.append(test_time_allsub,test_time)
        epoch_allsub =np.append(epoch_allsub,epochs)
        lr_allsub = np.append(lr_allsub,lr)
        batch_size_allsub = np.append(batch_size_allsub,batch_size)

    summary = pd.DataFrame(
        {'Subjects': subjects, 'average acc of 10 folds': test_accuracy_allsub, 'average loss of 10 fold': test_loss_allsub, 'average train time of 10 folds': train_time_allsub,
         'average test time of 10 folds': test_time_allsub, 'epochs': epoch_allsub,'lr':lr_allsub, 'batch size': batch_size_allsub})
    if label == 'all':
        summary_all = summary
    elif label == 'arousal':
        summary_arousal = summary
    elif label == 'valence':
        summary_valence = summary
    else:
        summary_dominance = summary

writer = pd.ExcelWriter('result_MWMF_div_'+label+'.xlsx')
summary_all.to_excel(writer, 'all', index=False)
# summary_valence.to_excel(writer, 'valence', index=False)
# summary_arousal.to_excel(writer, 'arousal', index=False)
# summary_dominance.to_excel(writer, 'dominance', index=False)
writer.save()



