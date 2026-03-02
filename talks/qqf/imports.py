# import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
# allow max rows to be displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, KFold
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline       
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, GradientBoostingRegressor
import numpy as np
from cubist import Cubist
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope
pd.set_option('display.max_rows', 150)
import pickle # for saving and loading models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose, MSTL
from sklearn.tree import DecisionTreeRegressor

from peshbeen.models import (ml_forecaster, ml_bidirect_forecaster, VARModel, MsHmmRegression, MsHmmVar)
from peshbeen.model_selection import (cross_validate,  mv_cross_validate,
                                      cv_tune, mv_cv_tune, prob_param_forecasts,
                                      tune_ets, tune_sarima, ParametricTimeSeriesSplit,
                                      forward_feature_selection, backward_feature_selection,
                                      mv_forward_feature_selection, mv_backward_feature_selection,
                                      hmm_forward_feature_selection, hmm_backward_feature_selection,
                                      hmm_mv_forward_feature_selection, hmm_mv_backward_feature_selection,
                                      hmm_cross_validate, hmm_mv_cross_validate, cv_lag_tune, 
                                      cv_hmm_lag_tune)
from peshbeen.statplots import (plot_ccf, plot_PACF_ACF)
from peshbeen.stattools import (unit_root_test, cross_autocorrelation,
                                lr_trend_model, forecast_trend, pacf_strength, ccf_strength)
from peshbeen.transformations import (fourier_terms, rolling_quantile,
                        rolling_mean, rolling_std, expanding_mean, expanding_std,
                        expanding_quantile, expanding_ets, box_cox_transform,
                        back_box_cox_transform,undiff_ts, seasonal_diff, invert_seasonal_diff,
                        nzInterval, zeroCumulative, kfold_target_encoder, target_encoder_for_test)
from peshbeen.metrics import (MAPE, MASE, MSE, MAE, RMSE, SMAPE, CFE, CFE_ABS, WMAPE, SRMSE, RMSSE, SMAE)
from peshbeen.prob_forecast import (ml_prob_forecasts, var_prob_forecasts, hmm_prob_forecasts, ets_prob_forecasts, arima_prob_forecasts, naive_prob_forecasts)
from sktime.transformations.series.boxcox import BoxCoxTransformer
sns.set_context("talk")