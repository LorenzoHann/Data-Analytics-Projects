{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Airbnb Coursework",
      "private_outputs": true,
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/LorenzoHann/data_analytics_projects/blob/main/Kaggle_pridicting_rent.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WCXBhrOy3cnu"
      },
      "source": [
        "import pandas as pd\n",
        "\n",
        "analysisDataPath = '/content/drive/MyDrive/Airbnb/analysisData.csv'\n",
        "scoringDataPath = '/content/drive/MyDrive/Airbnb/scoringData.csv'"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aEzOiAJ55mfH"
      },
      "source": [
        "# Keep only numeric columns\n",
        "analysisData = pd.read_csv(analysisDataPath)\n",
        "analysisData.drop(columns=['id'], inplace=True)\n",
        "\n",
        "#def to_bool(s):\n",
        " #   return 1 if analysisData['s'] == 't' else 0\n",
        "#to_bool(analysisData['host_is_superhost'])\n",
        "\n",
        "# a=nrow(analysisData)\n",
        "# b=ncol(analysisData)\n",
        "\n",
        "# aaa=analysisData['host_is_superhost']\n",
        "# bbb=[1 if i =='t' else 0 for i in aaa]\n",
        "\n",
        "#np.shape(csv_data)[1]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "h7c3LsmhPxjR"
      },
      "source": [
        "## Pre-Processing Analysis Data Utilities"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Va7QZsiOqupR"
      },
      "source": [
        "analysisData.columns"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xGcq4XpNP1sp"
      },
      "source": [
        "from sklearn.preprocessing import LabelEncoder\n",
        "\n",
        "tf_columns_list = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'is_location_exact', 'has_availability', 'requires_license', 'instant_bookable', 'is_business_travel_ready', 'require_guest_profile_picture', 'require_guest_phone_verification']\n",
        "perc_columns_list = ['host_response_rate', 'host_acceptance_rate']\n",
        "string_columns_list = ['square_feet', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee']\n",
        "encoding_columns_list = ['host_response_time', 'neighbourhood_group_cleansed', 'smart_location', 'country_code', 'property_type', 'room_type', 'bed_type', 'cancellation_policy']\n",
        "le = LabelEncoder()\n",
        "\n",
        "def bool_to_int(input_df):\n",
        "  for column in tf_columns_list:\n",
        "    input_df[column] = input_df[column].apply(lambda x: 1 if x == 't' else 0)\n",
        "\n",
        "def label_encoder(input_df):\n",
        "  input_df[encoding_columns_list] = input_df[encoding_columns_list].apply(le.fit_transform)\n",
        "\n",
        "def string_to_int(input_df):\n",
        "  input_df[string_columns_list] = input_df[string_columns_list].astype(float)\n",
        "  \n",
        "def perc_to_float(input_df):\n",
        "  for column in perc_columns_list:\n",
        "    input_df[column] = input_df[column].apply(lambda x: float(x.strip('%'))/100 if not pd.isna(x) else x)\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "87Djeb_acLbX"
      },
      "source": [
        "def pre_process(input_df):\n",
        "  bool_to_int(input_df)\n",
        "  label_encoder(input_df)\n",
        "  string_to_int(input_df)\n",
        "  perc_to_float(input_df)\n",
        "  input_df = input_df['zipcode'].apply(lambda x: pd.to_numeric(x,errors='coerce'))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "THjeAadfRhOM"
      },
      "source": [
        "pre_process(analysisData)\n",
        "analysisData = analysisData.select_dtypes(include='number')\n",
        "analysisData.fillna(0, inplace=True)\n",
        "\n",
        "analysisData.head()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "31CJA9fm69QO"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "# analysisData = pd.read_csv('/content/data2.csv')\n",
        "\n",
        "X = analysisData.loc[:, analysisData.columns != 'price']\n",
        "y = analysisData['price']\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hPKqrFDl6WyN"
      },
      "source": [
        "import xgboost as xgb\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.metrics import r2_score\n",
        "from matplotlib import pyplot as plt \n",
        "\n",
        "# Fitting the model\n",
        "xgb_reg = xgb.XGBRegressor()\n",
        "xgb_reg.fit(X_train, y_train)\n",
        "training_preds_xgb_reg = xgb_reg.predict(X_train)\n",
        "val_preds_xgb_reg = xgb_reg.predict(X_test)\n",
        "\n",
        "# Printing the results\n",
        "# print(f\"Time taken to run: {round((xgb_reg_end - xgb_reg_start)/60,1)} minutes\")\n",
        "print(\"\\nTraining RMSE:\", round(mean_squared_error(y_train, training_preds_xgb_reg, squared=False),4))\n",
        "print(\"Validation RMSE:\", round(mean_squared_error(y_test, val_preds_xgb_reg, squared=False),4))\n",
        "print(\"\\nTraining r2:\", round(r2_score(y_train, training_preds_xgb_reg),4))\n",
        "print(\"Validation r2:\", round(r2_score(y_test, val_preds_xgb_reg),4))\n",
        "\n",
        "# Producing a dataframe of feature importances\n",
        "ft_weights_xgb_reg = pd.DataFrame(xgb_reg.feature_importances_, columns=['weight'], index=X_train.columns)\n",
        "ft_weights_xgb_reg.sort_values('weight', inplace=True)\n",
        "\n",
        "# Plotting feature importances\n",
        "plt.figure(figsize=(8,20))\n",
        "plt.barh(ft_weights_xgb_reg.index, ft_weights_xgb_reg.weight, align='center') \n",
        "plt.title(\"Feature importances in the XGBoost model\", fontsize=14)\n",
        "plt.xlabel(\"Feature importance\")\n",
        "plt.margins(y=0.01)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "s6A9WFhG6y80"
      },
      "source": [
        "from sklearn.svm import SVR\n",
        "from sklearn.pipeline import make_pipeline\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.metrics import mean_absolute_error\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "import numpy as np\n",
        "\n",
        "features = ft_weights_xgb_reg.sort_values(by=['weight'], ascending=False).index[:20].tolist()\n",
        "print(features)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "o10OSCFYtvZc"
      },
      "source": [
        "from xgboost.sklearn import XGBRegressor\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import KFold, cross_val_score\n",
        "import xgboost as xgb\n",
        "from sklearn.model_selection import GridSearchCV"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JT-wdlNaqwwY"
      },
      "source": [
        "other_params = {'eta': 0.3, 'n_estimators': 500, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 1,\n",
        "                'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,\n",
        "                'seed': 33}\n",
        "cv_params = {'n_estimators': np.linspace(150, 250, 11, dtype=int)}\n",
        "regress_model = xgb.XGBRegressor(**other_params)  # 注意这里的两个 * 号！\n",
        "gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=5, n_jobs=-1)\n",
        "gs.fit(X_train[features], y_train)  # X为训练数据的特征值，y为训练数据的label\n",
        "# 性能测评\n",
        "print(\"参数的最佳取值：:\", gs.best_params_)\n",
        "print(\"最佳模型得分:\", gs.best_score_)\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vbBbX0FTOlvx"
      },
      "source": [
        "\n",
        "param = {\n",
        "  # 'max_depth':range(9,15,2),\n",
        "  # 'min_child_weight':range(5,7,1)\n",
        "  'gamma':[i/10.0 for i in range(0,5)],\n",
        "  #'subsample':[i/10.0 for i in range(6,10)],\n",
        "  #'colsample_bytree':[i/10.0 for i in range(6,10)],\n",
        "  #'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]\n",
        "}\n",
        "\n",
        "xgb = XGBRegressor(n_estimators=150, max_depth=9, min_child_weight=6)\n",
        "gsearch = GridSearchCV(estimator = xgb, param_grid = param, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=1)\n",
        "gsearch.fit(X, y)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2DonSUMKSkJn"
      },
      "source": [
        "gsearch.best_params_, gsearch.best_score_"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rEu5VI9pXONR"
      },
      "source": [
        "# Evalaute Fine-Tuned XGBoostRegressor on Analysis Data\n",
        "\n",
        "from xgboost.sklearn import XGBRegressor\n",
        "xgb = XGBRegressor(n_estimators=150, max_depth=9, min_child_weight=6, gamma=0, colsample_bytree=0.8, subsample=0.9, reg_alpha=0.1)\n",
        "\n",
        "xgb.fit(X_train, y_train)\n",
        "training_preds_xgb = xgb.predict(X_train)\n",
        "val_preds_xgb = xgb.predict(X_test)\n",
        "\n",
        "print(\"Validation RMSE:\", round(mean_squared_error(y_test, val_preds_xgb, squared=False),4))\n",
        "print(\"Validation r2:\", round(r2_score(y_test, val_preds_xgb),4))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iJQ-5fQjBIL2"
      },
      "source": [
        "##SVR 另一种方式\n",
        "pipe = Pipeline(steps=[(\"SC\", StandardScaler()), (\"svr\", SVR())])\n",
        "\n",
        "param_grid = {\n",
        "    \"svr__kernel\": ('linear', 'poly', 'rbf', 'sigmoid'),\n",
        "    \"svr__C\": [1,5,10],\n",
        "    \"svr__degree\": [3,8],\n",
        "    \"svr__coef0\": [0.01,0.5,10],\n",
        "    \"svr__gamma\": ('auto', 'scale')\n",
        "}\n",
        "\n",
        "regr = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=5, verbose=True)\n",
        "regr.fit(X[features], y)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "P-Gh-BnL-nij"
      },
      "source": [
        "from sklearn.metrics import mean_squared_error\n",
        "\n",
        "regr = make_pipeline(StandardScaler(), SVR(C=10))\n",
        "regr.fit(X[features], y)\n",
        "\n",
        "y_pred = regr.predict(X_test[features])\n",
        "mean_squared_error(y_test, y_pred, squared=False)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zv76K6-u_dA6"
      },
      "source": [
        "# Generate Submission File"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GnmKIPcCD8lS"
      },
      "source": [
        "# Read Scoring Data\n",
        "scoringData = pd.read_csv('/content/score3.csv')\n",
        "\n",
        "# Pre-process Data\n",
        "scoringData = scoringData.select_dtypes(['number'])\n",
        "scoringData.fillna(scoringData.mean(), inplace=True)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "daUp_qHW_PRv"
      },
      "source": [
        "# Using SVR\n",
        "scoringData['price'] = regr.predict(scoringData[features])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Y53Vsv6MD4Zp"
      },
      "source": [
        "# Using XGBoost\n",
        "#xgb = XGBRegressor(n_estimators=1000, max_depth=9, min_child_weight=1, gamma=0, colsample_bytree=0.8, subsample=0.8, reg_alpha=1e-2)\n",
        "from xgboost.sklearn import XGBRegressor\n",
        "from xgboost.sklearn import XGBRegressor\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import KFold, cross_val_score\n",
        "import xgboost as xgb\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "scoringData = pd.read_csv(scoringDataPath)\n",
        "pre_process(scoringData)\n",
        "scoringData = scoringData.select_dtypes(include='number')\n",
        "scoringData.fillna(0, inplace=True)\n",
        "\n",
        "other_params = {'eta': 0.36, 'n_estimators': 150, 'gamma': 0, 'max_depth': 9, 'min_child_weight': 6,\n",
        "                'colsample_bytree': 0.6, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 38, 'reg_alpha': 0,\n",
        "                'seed': 33}\n",
        "xgb = xgb.XGBRegressor(**other_params)\n",
        "# xgb = XGBRegressor(n_estimators=150, max_depth=9, min_child_weight=6, gamma=0, colsample_bytree=0.8, subsample=0.9, reg_alpha=0.01)\n",
        "X = analysisData.loc[:, analysisData.columns != 'price']\n",
        "y = analysisData['price']\n",
        "\n",
        "xgb.fit(X, y)\n",
        "scoringData['price'] = xgb.predict(scoringData.loc[:, scoringData.columns != 'id'])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yuvv1WreAGxS"
      },
      "source": [
        "scoringData = scoringData[['id', 'price']]\n",
        "scoringData.columns = ['id', 'price']\n",
        "scoringData.head()\n",
        "\n",
        "scoringData.to_csv('/content/submission2.1.csv', index=False)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Z037wN32FXRd"
      },
      "source": [
        "# copy\n",
        "#Read Scoring Data\n",
        "scoringData = pd.read_csv('/content/score3.csv')\n",
        "\n",
        "# Pre-process Data\n",
        "scoringData = scoringData.select_dtypes(['number'])\n",
        "scoringData.fillna(scoringData.mean(), inplace=True)\n",
        "\n",
        "# Using XGBoost\n",
        "xgb = XGBRegressor(eta=0.36,n_estimators=89, max_depth=5, min_child_weight=9, gamma=0, colsample_bytree=0.6, colsample_bylevel=1, subsample=1,reg_lambda=38,reg_alpha=0,\n",
        "                seed=33)\n",
        "X = analysisData.loc[:, analysisData.columns != 'price']\n",
        "y = analysisData['price']\n",
        "\n",
        "xgb.fit(X, y)\n",
        "scoringData['price'] = xgb.predict(scoringData.loc[:, scoringData.columns != 'id'])\n",
        "\n",
        "scoringData = scoringData[['id', 'price']]\n",
        "scoringData.columns = ['id', 'price']\n",
        "scoringData.head()\n",
        "\n",
        "scoringData.to_csv('/content/submission5.2.csv', index=False)"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}