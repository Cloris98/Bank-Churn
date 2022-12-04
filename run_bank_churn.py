from data_prep_bank import *
from model_train_bank import *
from model_evaluation_bank import *


if __name__ == '__main__':
    """
    
    """
    FILE_PATH = 'bank.data.csv'

    sampling_mode = ['org', 'rus', 'ros', 'smote']
    for mode in sampling_mode:

        if mode == 'org':
            print(
                '****************************',
                'Start Analysing original data',
                '****************************',
                )
        if mode == 'rus':
            print(
                '****************************',
                'Start Analysing Random Under Sampled Data',
                '****************************',
                )
        if mode == 'ros':
            print('****************************',
                  'Start Analysing Random Over Sampled Data',
                  '****************************',
                  )
        if mode == 'smote':
            print('****************************',
                  'Start Analysing data under SMOTE',
                  '****************************',
                  )
        # Get the data
        DATA = Preprocessing(FILE_PATH).process(mode)
        LR, kNN, RF = TrainModel(DATA).best_models()
        # print(DATA[2], DATA[3])

        # Evaluate the best model
        ModelEval(LR, kNN, RF).eval(DATA[2], DATA[3], LR, kNN, RF)
    # Get the best train model



