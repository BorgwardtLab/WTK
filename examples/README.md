This directory allows to run `WTK` on two example data sets
from the [UCR TS Archive](http://timeseriesclassification.com/). 

### Requirements
The easiest way to get the examples running is by using `pipenv`. If you are using a MacOS based machine, you can easily install pipenv via
```
$ brew install pipenv
```

To install it via `pip` simply
```
$ pip install --user pipenv
```

For other ways of installing `pipenv`, please refer to the [documentation](https://docs.pipenv.org/en/latest/install/).

### Install Dependencies
Once `pipenv` is installed, you can install required dependcies and switch into the installed virtual environment via
```
$ pipenv install
$ pipenv shell
```

### Running the examples
To run the examples, simply execute the `main.py` script and provide the training and test files:
```
$ python main.py ../data/UCR/raw_data/DistalPhalanxTW/DistalPhalanxTW_TRAIN ../data/UCR/raw_data/DistalPhalanxTW/DistalPhalanxTW_TEST
```
