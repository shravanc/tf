from sklearn.model_selection import train_test_split, GridSearchCV

def load_data(df):
  TARGETS = ['motor_UPDRS',	'total_UPDRS'	]
  TARGET = TARGETS[1]

  y = df[TARGET]
  X = df.drop(TARGET, axis=1).values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
  return X_train, X_test, y_train, y_test

def train_datasets(df, model):
  X_train, X_test, y_train, y_test = load_data(df)
  estimator   = model[1]
  param_grid  = model[2]  
 
  search = GridSearchCV(estimator = estimator, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2)
  search.fit(X_train, y_train)
  return search.best_score_


