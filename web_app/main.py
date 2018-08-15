from flask import Flask, render_template, request
# from flask.ext.bootstrap import Bootstrap
# from flask.ext.scss import Scss
from flaskext.sass import sass
# from flaskext.lesscss import lesscss
from sklearn.externals import joblib
from helper_functions import *
import numpy as np

# sass(app, input_dir='assets/scss', output_dir='static/css')

rf_clf = joblib.load("rf_final2_model.pkl")
scaler = unpickle_object("scaler_27_features.pkl")
print("model loaded")
order = ('int_rate',
 'term_ 60 months',
 'revol_util',
 'installment',
 'loan_amnt',
 'mo_sin_old_rev_tl_op',
 'total_acc',
 'mo_sin_old_il_acct',
 'acc_open_past_24mths',
 'mths_since_last_delinq',
 'num_bc_tl',
 'fico_range_low',
 'fico_range_high',
 'il_util',
 'dti_log',
 'bc_open_to_buy_log',
 'avg_cur_bal_log',
 'tot_hi_cred_lim_log',
 'revol_bal_log',
 'tot_cur_bal_log',
 'total_bal_ex_mort_log',
 'mths_since_recent_bc_log',
 'mo_sin_rcnt_tl_log',
 'mo_sin_rcnt_rev_tl_op_log',
 'mths_since_recent_inq_log',
 'annual_inc_log',
 'max_bal_bc_log')


app = Flask(__name__)
app.debug = True

sass(app, input_dir='assets/scss', output_dir='static/css')


@app.route('/')

def index():
	return render_template('index.html')

@app.route('/data_entry')
def data_entry():
	return render_template("data_entry.html")

@app.route('/predict', methods = ['POST'])
def result():
	if request.method == 'POST':
		result = request.form
		dict_of_data = dict(result.items())
		dict_of_data['loan_status_Late'] = 1
		# print(dict_of_data)
		entries = []
		for label in order:
			entries.append(dict_of_data[label])
		array_entries = np.array(entries)
		scaled_entries = scaler.transform(array_entries.reshape(1,-1))
		prediction = rf_clf.predict(scaled_entries)[0]

		if prediction == 0:
			prediction = "Result: Will Pay on time"
		else:
			prediction = "Result: Will NOT pay on time"
		return render_template("predict.html", result = result, prediction=prediction)


if __name__ == '__main__':
    app.debug = True #Uncomment to enable debugging
    app.run()
