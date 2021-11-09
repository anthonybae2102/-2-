#SVM+Bayesian search
class LinearClassifier(BaseEstimator,ClassifierMixin):
    """Linear classifier: logistic regression, SVM."""

    def __init__(self, classifier='svm', kernel='linear', C=1, p=3, gamma=1e0, class_weight='balanced', n_jobs=1, verbose=True):

        self.classifier = classifier
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.p = p
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._model = None
        self._calib = None

    def fit(self, X, y=None, sample_weight=None, param_search=False, calibrate=False, cv_split=None):
        """Fit the model."""

        if type(X) is list:
            X = np.vstack(X)#convert X into 2D if it is 1D array

        if param_search:
            # search for best hyperparameters
            self.C,self.gamma = self.param_search( X, y, self.C, self.gamma, sample_weight )

        # normalize to zero mean, unit std dev
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-4
        X = ( X - self.mu ) / self.sigma

        # create model
        if self.classifier.lower() == 'logistic':
            self._model = sklearn.linear_model.LogisticRegression( C=self.C, class_weight=self.class_weight, solver='sag' )
        elif self.classifier.lower() == 'svm':
            if self.kernel.lower() == 'linear':
                self._model = sklearn.svm.LinearSVC( C=self.C, class_weight=self.class_weight )
            elif self.kernel.lower() == 'poly':
                self._model = sklearn.svm.SVC( kernel='poly', degree=self.p, C=self.C, class_weight=self.class_weight )
            elif self.kernel.lower() == 'rbf':
                if self.gamma is None:
                    self.gamma = 1.0 / X.shape[1]
                self._model = sklearn.svm.SVC( kernel=self.kernel, C=self.C, gamma=self.gamma, class_weight=self.class_weight, probability=True )
        elif self.classifier.lower() == 'lda':
            self._model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis( solver='lsqr' )

        # calibrate
        if calibrate:
            cl = self._model

            method = 'isotonic'
            if len(y) < 1000:
                method = 'sigmoid'

            if cv_split is not None:
                self._calib = sklearn.calibration.CalibratedClassifierCV( cl, method=method, cv=cv_split )
            else:
                self._calib = sklearn.calibration.CalibratedClassifierCV( cl, method=method, cv=5 )
            y  = y.reshape((len(y),))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',category=ConvergenceWarning)
                self._calib.fit( X, y, sample_weight=sample_weight )
        else:
            self._calib = None

        # fit model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=ConvergenceWarning)
            if sample_weight is not None and self.classifier.lower() != 'lda':
                self._model.fit( X, y.flatten(), sample_weight=sample_weight )
            else:
                self._model.fit( X, y.flatten() )
        
        return self

    def predict(self, X, y=None, cv=None ):
        """Predict."""

        if type(X) is list:
            X = np.vstack(X)
        
        X = ( X - self.mu ) / self.sigma

        if self._calib is not None:
            # use calibration
            if cv is not None:
                p = self._calib.calibrated_classifiers_[cv].predict_proba( X )#calibrated_classifiers_[cv].predict_proba assigend by
            else:
                p = self._calib.predict_proba( X )
        else:
            if self.classifier in ['logistic','svm'] and self.kernel == 'linear':
                d = self._model.decision_function( X )
                p = 1.0 / ( np.exp(-d) + 1 )
                if len(p.shape) == 1:
                    p = p.reshape((len(p),1))
                    p = np.concatenate( (1-p,p), axis=1 )
            else:
                p = self._model.predict_proba( X )

        return p

    def param_search(self, X, y, quick=True, C=None, gamma=None, sample_weight=None):
        """Grid search for best hyperparameters."""

        if type(X) is list:
            X = np.vstack(X)
        
        # normalize to zero mean, unit std dev
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-4
        X = ( X - self.mu ) / self.sigma

        # hyperparameters to search over
        if self.classifier.lower() == 'lda':
            Cvals = [1.0]
            gvals = [1.0]
        elif C is None or not quick:
            Cvals = [float(2**e) for e in range(-15,10)]
            if self.kernel == 'rbf':
                gvals = [float(2**e) for e in range(-20,5)]
            else:
                gvals = [1.0]
        else:
            Cvals = [ float(2**e)*C for e in range(-2,3) ]
            if self.kernel == 'rbf':
                gvals = [ float(2**e)*gamma for e in range(-2,3) ]
            else:
                gvals = [1.0]

        # grid search to find best C
        cv = min(20,min(max((y==0).sum(),(y==-1).sum()),(y==1).sum()))
        metric = 'accuracy'
        print("new function starts")
        if self.classifier.lower() == 'logistic':
            clf = BayesSearchCV( sklearn.linear_model.LogisticRegression( class_weight=self.class_weight, solver='sag' ), [{'C':Real(1e-6, 1e+6, prior='log-uniform')}], cv=cv, scoring=metric, n_jobs=self.n_jobs, refit=False )#, fit_params={'sample_weight':sample_weight} )
        elif self.classifier.lower() == 'svm':
            if self.kernel.lower() == 'linear':
                clf = BayesSearchCV( sklearn.svm.LinearSVC( class_weight=self.class_weight ), [{'C':Real(1e-6, 1e+6, prior='log-uniform')}], cv=cv, scoring=metric, n_jobs=self.n_jobs, refit=False )#, fit_params={'sample_weight':sample_weight} 
            elif self.kernel == 'poly':
                clf = sklearn.model_selection.GridSearchCV( sklearn.svm.SVC( kernel='poly', degree=self.p, class_weight=self.class_weight ), [{'C':Cvals}], cv=cv, scoring=metric, n_jobs=self.n_jobs, refit=False )#, fit_params={'sample_weight':sample_weight} )
            elif self.kernel == 'rbf':
                clf = sklearn.model_selection.GridSearchCV( sklearn.svm.SVC( kernel='rbf', class_weight=self.class_weight ), [{'C':Cvals,'gamma':gvals}], cv=cv, scoring=metric, n_jobs=self.n_jobs, refit=False )#, fit_params={'sample_weight':sample_weight} )

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=ConvergenceWarning)
            clf.fit( X, y.flatten() )

        C = clf.best_params_['C']
        if self.kernel == 'rbf':
            gamma = clf.best_params_['gamma']

        self.C = C
        self.gamma = gamma

        return C,gamma

