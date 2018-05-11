from TheKesselRun.Code.Learner import Learner
from TheKesselRun.Code.Catalyst import Catalyst

import itertools
import pandas as pd


def load_nh3_catalysts(learner, featgen=0):
    """ Import NH3 data from Katie's HiTp dataset(cleaned). """

    if learner.average_data:
        df = pd.read_csv(r"..\Data\Processed\AllData_Condensed.csv", index_col=0)
    else:
        df = pd.read_csv(r"..\Data\Processed\AllData.csv", index_col=0)

    for index, row in df.iterrows():
        cat = Catalyst()
        cat.ID = row['ID']
        cat.add_element(row['Ele1'], row['Wt1'])
        cat.add_element(row['Ele2'], row['Wt2'])
        cat.add_element(row['Ele3'], row['Wt3'])
        cat.input_reactor_number(int(row['Reactor']))
        cat.input_temperature(row['Temperature'])
        cat.input_space_velocity(row['Space Velocity'])
        cat.input_ammonia_concentration(row['NH3'])
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
        version='v13-pred',
        feature_generator=0,  # 0 is elemental, 1 is statistics,  2 is statmech
        regression=True
    )
    if skynet.regression:
        skynet.set_learner(learner='rfr', params='default')
    else:
        skynet.set_learner(learner='rfc', params='default')

    eledf = pd.read_csv(r'./Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [12] + list(range(20, 31)) + list(range(38, 43)) + \
               list(range(44, 51)) + list(range(74, 80)) + [56, 72, 82, 83]

    ele_df = eledf[eledf['Atomic Number'].isin(ele_list)]
    eles = ele_df.index.values
    combos = list(itertools.combinations(eles, r=2))

    final_list = list()
    for vals in combos:
        tmp = 300

        cat1 = create_catalyst(e1=vals[0], w1=3, e2=vals[1], w2=1, e3='K', w3=12,
                               tmp=tmp, reactnum=8, space_vel=2000, ammonia_conc=0.01)
        skynet.add_catalyst('Predict', cat1)

        cat2 = create_catalyst(e1=vals[0], w1=2, e2=vals[1], w2=2, e3='K', w3=12,
                               tmp=tmp, reactnum=8, space_vel=2000, ammonia_conc=0.01)
        skynet.add_catalyst('Predict', cat2)

        cat3 = create_catalyst(e1=vals[0], w1=1, e2=vals[1], w2=3, e3='K', w3=12,
                               tmp=tmp, reactnum=8, space_vel=2000, ammonia_conc=0.01)
        skynet.add_catalyst('Predict', cat3)

    load_nh3_catalysts(skynet, featgen=0)
    skynet.filter_master_dataset()
    skynet.train_data()
    skynet.predict_from_masterfile()

def prediction_pipeline(learner):
    learner.filter_master_dataset()
    learner.train_data()
    learner.predict_from_masterfile(catids=[65, 66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 82, 83], svnm='SS8')
    learner.predict_from_masterfile(catids=[38, 84, 85, 86, 87, 89, 90, 91, 93], svnm='SS9')

def temperature_slice(learner):
    for t in [250, 300, 350, 400, 450, None]:
        learner.set_temp_filter(t)
        learner.filter_master_dataset()

        learner.train_data()
        learner.extract_important_features(sv=True, prnt=True)
        learner.predict_crossvalidate()
        if learner.regression:
            learner.evaluate_regression_learner()
        else:
            learner.evaluate_classification_learner()
        learner.preplotcessing()
        learner.plot_basic()
        learner.plot_error()
        # learner.plot_features_colorbar(x_feature='Predicted Conversion', c_feature='ammonia_concentration')
        learner.bokeh_predictions()
        learner.bokeh_by_elements()


if __name__ == '__main__':
    # Begin Machine Learning
    skynet = Learner(
        average_data=True,
        element_filter=3,
        temperature_filter=None,
        version='v14',
        regression=True
    )

    if skynet.regression:
        skynet.set_learner(learner='rfr', params='default')
    else:
        skynet.set_learner(learner='rfc', params='default')

    load_nh3_catalysts(learner=skynet, featgen=0) # 0 is elemental, 1 is statistics,  2 is statmech

    temperature_slice(learner=skynet)
    # prediction_pipeline(learner=skynet)