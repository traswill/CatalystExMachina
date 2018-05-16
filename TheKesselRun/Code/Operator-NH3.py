from TheKesselRun.Code.Learner import Learner
from TheKesselRun.Code.Catalyst import Catalyst

import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_nh3_catalysts(learner, featgen=0):
    """ Import NH3 data from Katie's HiTp dataset(cleaned). """

    if learner.average_data:
        df = pd.read_csv(r"..\Data\Processed\AllData_Condensed.csv", index_col=0)
    else:
        df = pd.read_csv(r"..\Data\Processed\AllData.csv", index_col=0)

    cl_atom_df = pd.read_excel(r'..\Data\Catalyst_Synthesis_Parameters.xlsx', index_col=0)

    for index, row in df.iterrows():
        cat = Catalyst()
        cat.ID = row['ID']
        cat.add_element(row['Ele1'], row['Wt1'])
        cat.add_element(row['Ele2'], row['Wt2'])
        cat.add_element(row['Ele3'], row['Wt3'])
        # cat.input_reactor_number(int(row['Reactor']))
        cat.input_temperature(row['Temperature'])
        cat.input_space_velocity(row['Space Velocity'])
        cat.input_ammonia_concentration(row['NH3'])
        # cat.input_n_Cl_atoms(cl_atom_df.loc[row['ID']].values[0])
        if learner.average_data:
            cat.input_standard_error(row['Standard Error'])
            cat.input_n_averaged_samples(row['nAveraged'])
        cat.activity = row['Concentration']
        cat.feature_add_n_elements()
        # cat.feature_add_oxidation_states()

        feature_generator = {
            0: cat.feature_add_elemental_properties,
            1: cat.feature_add_statistics,
            2: cat.feature_add_weighted_average
        }
        feature_generator.get(featgen, lambda: print('No Feature Generator Selected'))()

        learner.add_catalyst(index='{ID}_{T}'.format(ID=cat.ID, T=row['Temperature']), catalyst=cat)

    learner.create_master_dataset()


def predict_all_binaries():
    # TODO migrate into learner
    def create_catalyst(e1, w1, e2, w2, e3, w3, tmp, reactnum, space_vel, ammonia_conc):
        cat = Catalyst()
        cat.ID = 'A'
        cat.add_element(e1, w1)
        cat.add_element(e2, w2)
        cat.add_element(e3, w3)
        cat.input_reactor_number(reactnum)
        cat.input_temperature(tmp)
        cat.input_space_velocity(space_vel)
        cat.input_ammonia_concentration(ammonia_conc)
        cat.feature_add_n_elements()

        feature_generator = {
            0: cat.feature_add_elemental_properties,
            1: cat.feature_add_statistics,
            2: cat.feature_add_weighted_average
        }
        feature_generator.get(0, lambda: print('No Feature Generator Selected'))()

        return cat

    skynet = Learner(
        average_data=True,
        element_filter=0,
        temperature_filter=None,
        version='v20-pred',
        regression=True
    )

    if skynet.regression:
        skynet.set_learner(learner='rfr', params='default')
    else:
        skynet.set_learner(learner='rfc', params='default')

    eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [12] + list(range(20, 31)) + list(range(38, 43)) + \
               list(range(44, 51)) + list(range(74, 80)) + [56, 72, 82, 83]

    ele_df = eledf[eledf['Atomic Number'].isin(ele_list)]
    eles = ele_df.index.values
    combos = list(itertools.combinations(eles, r=2))

    for vals in combos:
        tmp = 250

        cat1 = create_catalyst(e1=vals[0], w1=3, e2=vals[1], w2=1, e3='K', w3=12,
                               tmp=tmp, reactnum=1, space_vel=2000, ammonia_conc=1)
        skynet.add_catalyst('Predict', cat1)

        cat2 = create_catalyst(e1=vals[0], w1=2, e2=vals[1], w2=2, e3='K', w3=12,
                               tmp=tmp, reactnum=1, space_vel=2000, ammonia_conc=1)
        skynet.add_catalyst('Predict', cat2)

        cat3 = create_catalyst(e1=vals[0], w1=1, e2=vals[1], w2=3, e3='K', w3=12,
                               tmp=tmp, reactnum=1, space_vel=2000, ammonia_conc=1)
        skynet.add_catalyst('Predict', cat3)

    load_nh3_catalysts(skynet, featgen=0)
    skynet.filter_master_dataset()
    skynet.train_data()
    skynet.predict_dataset()


def predict_half_Ru_catalysts():
    def create_catalyst(e1, w1, e2, w2, e3, w3, tmp, reactnum, space_vel, ammonia_conc):
        cat = Catalyst()
        cat.ID = 'A'
        cat.add_element(e1, w1)
        cat.add_element(e2, w2)
        cat.add_element(e3, w3)
        cat.input_reactor_number(reactnum)
        cat.input_temperature(tmp)
        cat.input_space_velocity(space_vel)
        cat.input_ammonia_concentration(ammonia_conc)
        cat.feature_add_n_elements()

        feature_generator = {
            0: cat.feature_add_elemental_properties,
            1: cat.feature_add_statistics,
            2: cat.feature_add_weighted_average
        }
        feature_generator.get(0, lambda: print('No Feature Generator Selected'))()

        return cat

    skynet = Learner(
        average_data=True,
        element_filter=0,
        temperature_filter=None,
        version='v20-pred',
        regression=True
    )

    if skynet.regression:
        skynet.set_learner(learner='rfr', params='default')
    else:
        skynet.set_learner(learner='rfc', params='default')

    eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [12] + list(range(20, 31)) + list(range(38, 43)) + \
               list(range(44, 51)) + list(range(74, 80)) + [56, 72, 82, 83]

    ele_df = eledf[eledf['Atomic Number'].isin(ele_list)]
    eles = ele_df.index.values

    for val in eles:
        tmp = 300

        cat1 = create_catalyst(e1='Ru', w1=0.5, e2=val, w2=4, e3='K', w3=12,
                               tmp=tmp, reactnum=1, space_vel=2000, ammonia_conc=1)
        skynet.add_catalyst('Predict', cat1)

        cat2 = create_catalyst(e1='Ru', w1=0.5, e2=val, w2=2, e3='K', w3=12,
                               tmp=tmp, reactnum=1, space_vel=2000, ammonia_conc=1)
        skynet.add_catalyst('Predict', cat2)

    load_nh3_catalysts(skynet, featgen=0)
    skynet.filter_master_dataset()
    skynet.train_data()
    skynet.predict_dataset()


def prediction_pipeline(learner):
    learner.filter_master_dataset()
    learner.train_data()
    learner.predict_from_masterfile(catids=[65, 66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 82, 83], svnm='SS8')
    learner.predict_from_masterfile(catids=[38, 84, 85, 86, 87, 89, 90, 91, 93], svnm='SS9')


def temperature_slice(learner):
    for t in [None]: #['not350', 250, 300, 350, 400, 450, None]:
        learner.set_temp_filter(t)
        learner.filter_master_dataset()

        learner.train_data()
        learner.extract_important_features(sv=True, prnt=True)
        learner.predict_crossvalidate(kfold=10)
        if learner.regression:
            learner.evaluate_regression_learner()
        else:
            learner.evaluate_classification_learner()
        learner.preplotcessing()
        learner.plot_basic()
        learner.plot_error(metadata=True)
        # learner.plot_features_colorbar(x_feature='Predicted Conversion', c_feature='ammonia_concentration')
        learner.bokeh_predictions()
        learner.bokeh_by_elements()


def test_all_ML_models():
    from sklearn.metrics import r2_score, explained_variance_score, \
        mean_absolute_error, roc_curve, recall_score, precision_score, mean_squared_error

    skynet = Learner(
        average_data=True,
        element_filter=3,
        temperature_filter=None,
        ammonia_filter=1,
        space_vel_filter=2000,
        version='v20',
        regression=True
    )

    load_nh3_catalysts(skynet, featgen=0)
    skynet.filter_master_dataset()
    eval_dict = dict()

    for algs in ['rfr','adaboost','tree','neuralnet','svr','knnr','krr','etr','gbr','ridge','lasso']:
        if algs == 'neuralnet':
            skynet.set_learner(learner=algs, params='nnet')
        else:
            skynet.set_learner(learner=algs, params='empty')

        skynet.predict_crossvalidate(kfold=10)
        eval_dict[algs] = mean_absolute_error(skynet.labels_df.values, skynet.predictions)

    print(eval_dict)
    return eval_dict

def plot_all_ML_models(d):
    nm_dict = {
        'rfr': 'Random Forest',
        'adaboost': 'AdaBoost',
        'tree': 'Decision Tree',
        'neuralnet': 'Neural Net',
        'svr': 'Support Vector Machine',
        'knnr': 'k-Nearest Neighbor Regression',
        'krr': 'Kernel Ridge Regression',
        'etr': 'Extremely Randomized Trees',
        'gbr': 'Gradient Tree Boosting',
        'ridge': 'Ridge Regressor',
        'lasso': 'Lasso Regressor'
    }

    # names = ['Random Forest', 'Adaboost', 'Decision Tree', 'Neural Net', 'Support Vector Machine',
    #          'k-Nearest Neighbor Regression', 'Kernel Ridge Regression', 'Extra Tree Regressor',
    #          'Gradient Boosting Regressor', 'Ridge Regressor', 'Lasso Regressor']
    # vals = [0.121, 0.158, 0.152, 0.327, 0.327, 0.245, 0.168, 0.109, 0.119, 0.170, 0.188]
    # df = pd.DataFrame([names, vals],  index=['Algorithm', 'Mean Absolute Error']).T

    names = d.keys()
    vals = d.values()

    df = pd.DataFrame([names, vals], index=['rgs', 'Mean Absolute Error']).T
    df['Machine Learning Algorithm'] = [nm_dict.get(x, 'ERROR') for x in df['rgs'].values]
    df.sort_values(by='Mean Absolute Error', inplace=True, ascending=False)

    g = sns.barplot(x='Machine Learning Algorithm', y='Mean Absolute Error', data=df, palette="GnBu_d")
    g.set_xticklabels(g.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # d = test_all_ML_models()
    # plot_all_ML_models(d)
    # exit()

    # predict_all_binaries()
    # predict_half_Ru_catalysts()
    # exit()

    # Begin Machine Learning
    skynet = Learner(
        average_data=True,
        element_filter=3,
        temperature_filter=None,
        ammonia_filter=1,
        space_vel_filter=2000,
        version='v20-etr',
        regression=True
    )

    if skynet.regression:
        skynet.set_learner(learner='etr', params='etr')
        # skynet.set_learner(learner='gbr', params='gbr')
        # skynet.set_learner(learner='rfr', params='forest')
    else:
        skynet.set_learner(learner='rfc', params='forest')

    load_nh3_catalysts(learner=skynet, featgen=0) # 0 is elemental, 1 is statistics,  2 is statmech
    # skynet.filter_master_dataset()
    # skynet.hyperparameter_tuning()
    # exit()
    temperature_slice(learner=skynet)
    prediction_pipeline(learner=skynet)