from TheKesselRun.Code.Learner import Learner
from TheKesselRun.Code.Catalyst import Catalyst

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


def prediction_pipeline(learner):
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


def temperature_slice(learner):
    for t in [250, 300, 350, 400, 450, None]:
        learner.set_temp_filter(t)
        learner.filter_master_dataset()
        # skynet.predict_testdata(catids=[65, 66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 82, 83])
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