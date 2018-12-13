import datetime
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# load dataset
dataframe = pandas.read_csv("data/Car_sales.csv")
dataset = dataframe.values

# split into input (X) and output (Y) variables
prices = []
features = []

makes = {}
models = {}
vehicle_type = {}
makes_i = 0
models_i = 0
vehicle_types_i = 0

# Go through the dataset and find all the possible make, model, type classes.
for i in dataset:
    d = list(i)
    i_make = d[0]
    i_model = d[1]
    i_vehicle_type = d[4]

    if i_make.lower() not in makes:
        makes[i_make.lower()] = makes_i
        makes_i += 1

    if i_model.lower() not in models:
        models[i_model.lower()] = models_i
        models_i += 1

    if i_vehicle_type.lower() not in vehicle_type:
        vehicle_type[i_vehicle_type.lower()] = vehicle_types_i
        vehicle_types_i += 1

# Clean the data.
for i in dataset:
    p = i[5]
    # Any nulls in the data will kill the neural network.
    if str(p) == "nan":
        continue

    prices.append(i[5])

    # Make a copy to operate on
    d = list(i)

    # Find the indices for the categorical data.
    i_make = makes[d[0].lower()]
    i_model = models[d[1].lower()]
    i_vehicle_type = vehicle_type[d[4].lower()]

    # Get the year, we could do it like the reg plates for a bit more accuracy.
    i_made = datetime.datetime.strptime(d[14], "%m/%d/%Y")
    i_made = i_made.year

    # Get rid of the string values.
    del d[14]
    del d[4]
    del d[1]
    del d[0]

    # Add our cleaned values to the end.
    d.append(i_make)
    d.append(i_model)
    d.append(i_vehicle_type)
    d.append(i_made)
    d = list(numpy.nan_to_num(d))
    features.append(numpy.array(d))

X = numpy.array(features)
Y = numpy.array(prices)

fifth_of_data = int(len(X) / 5)

x_train = X[:fifth_of_data * 4]
y_train = Y[:fifth_of_data * 4]

x_test = X[(fifth_of_data * 4) + 1:]
y_test = Y[(fifth_of_data * 4) + 1:]

def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim=len(features[0]), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

model = baseline_model()
model.fit(x_train, y_train, epochs=100, batch_size=5, verbose=0)

for i in range(len(X)):
    y = y_test[i]
    pred = model.predict(x_test[i].reshape(1, -1))
    print ("predicted price (thousands)", pred[0][0], "actual", y)