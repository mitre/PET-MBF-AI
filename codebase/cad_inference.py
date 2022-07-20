import numpy as np
from joblib import dump, load
import json
from absl import flags
from absl import app

"""
This script takes a JSON with 17 segment rest, stress, reserve, and difference
data and outputs a JSON of probabilities that the study contains CAD. The output
JSON contains fields abnormal, which contains the per patient probability of
either scar or ischemia.  It also contains fields scar_lad, scar_rca, scar_lcx,
ischemia_lad, ischemia_rca, ischemia_lcx for probabilities of each abnormality
in each region.
"""

flags.DEFINE_string('data_17s', '',
    """JSON in string containing fields "rest", "stress", "reserve", and
       "difference". Each field corresponds to either a 1D list of 17 segment 
        measurements in standard order (for prediction of a single study) or
        a 2d list of these measurement lists (for simultanious prediciton of
        multiple studies).""")

flags.DEFINE_string('data_17s_path', '',
    """Path to JSON file containing fields "rest", "stress", "reserve", and
       "difference". Each field corresponds to either a 1D list of 17 segment 
        measurements in standard order (for prediction of a single study) or
        a 2d list of these measurement lists (for simultanious prediciton of
        multiple studies).""")

flags.DEFINE_string('savepath', None,
                    """Optional path to results json file to be created on
                       completion of this script.  If not provided, results
                       will print to stdout.""")

flags.DEFINE_string('localization_model_path', None,
                    """Path to localization model""")
flags.DEFINE_string('detection_model_path', None,
                    """Path to detection model""")

FLAGS = flags.FLAGS


def main(argv):
    # Format data
    if (FLAGS.data_17s != '') & (FLAGS.data_17s_path != ''):
        raise ValueError('Can only one of --data_17s and --data_17s_path')
    if FLAGS.data_17s != '':
        measurements = json.loads(FLAGS.data_17s)
    elif FLAGS.data_17s_path != '':
        with open(FLAGS.data_17s_path) as data_p:
            measurements = json.load(data_p)

    for key in measurements.keys():
        measurements[key] = np.array(measurements[key])

    rest = measurements['rest']
    stress = measurements['stress']
    reserve = measurements['reserve']
    difference = measurements['difference']

    # reshape to 2d matrix if single measurement
    rest = rest.reshape(-1, 17)
    stress = stress.reshape(-1, 17)
    reserve = reserve.reshape(-1, 17)
    difference = difference.reshape(-1, 17)

    measurements = np.hstack((rest, stress, reserve, difference))

    # load models
    rf_loc = load(FLAGS.localization_model_path)
    rf_det = load(FLAGS.detection_model_path)

    probs = {}
    abn_probs = rf_det.predict_proba(measurements)
    probs['abnormal'] = abn_probs[:,1]

    preds = rf_loc.predict_proba(measurements)
    outcomes = ['scar_lad', 'scar_rca', 'scar_lcx', 'ischemia_lad', 'ischemia_rca', 'ischemia_lcx']

    # preds is list of np arrays, 1 for each region
    for i, outcome in enumerate(outcomes):
        probs[outcome] = preds[i][:,1]

    # Convert to lists to JSONify
    for key in probs.keys():
        probs[key] = probs[key].tolist()

    # output probabilities
    if not FLAGS.savepath:
        probs = json.dumps(probs)
        print(probs)
    else:
        with open(FLAGS.savepath, 'w') as f:
            json.dump(probs, f)

if __name__ == '__main__':
    app.run(main)
