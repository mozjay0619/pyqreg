import pandas as pd
from patsy import NAAction, dmatrices

from .quantile_regression import QuantReg


def quantreg(formula, data, missing="drop"):
    """To support the formula API similar to that of statsmodels."""
    na_action = NAAction(on_NA=missing)

    y, X = dmatrices(formula, data, return_type="matrix", NA_action=na_action)

    _, X_names = [elem.strip() for elem in formula.split("~")]
    X_names = [elem.strip() for elem in X_names.split("+")]
    X_names.insert(0, "Intercept")

    return QuantRegFormulaWrapper(y.ravel(), X, X_names)


class QuantRegFormulaWrapper(QuantReg):
    def __init__(self, y, X, X_names):
        super(QuantRegFormulaWrapper, self).__init__(y, X)
        self.X_names = X_names

    def fit(self, *args, **kwargs):
        """Upon invoking the fit method of QuantReg, wrap the numeric
        results in pandas Series to mimic the results of the statsmodels.
        """
        self.res = super(QuantRegFormulaWrapper, self).fit(*args, **kwargs)

        self.res.params = pd.Series(self.res.params, index=self.X_names)
        self.res.bse = pd.Series(self.res.bse, index=self.X_names)
        self.res.pvalues = pd.Series(self.res.pvalues, index=self.X_names)
        self.res.tvalues = pd.Series(self.res.tvalues, index=self.X_names)

        return self.res

    def conf_int(self, alpha=0.05):
        """Produce the confidence interval."""
        conf_int = super(QuantRegFormulaWrapper, self).conf_int(alpha)

        return pd.DataFrame(data=conf_int, index=self.X_names)

    def summary(self):
        """Produce summary table similar to the table value returned
        when summary method is invoked for the statsmodels models.
        """
        summary = pd.concat(
            [
                self.res.params,
                self.res.bse,
                self.res.tvalues,
                self.res.pvalues,
                self.res.conf_int(),
            ],
            axis=1,
        )
        summary.columns = ["coef", "std err", "t", "P>|t|", "[0.025", "0.975]"]
        summary["coef"] = summary["coef"].apply(lambda x: float("{:.4f}".format(x)))
        summary["std err"] = summary["std err"].apply(
            lambda x: float("{:.3f}".format(x))
        )
        summary["t"] = summary["t"].apply(lambda x: float("{:.3f}".format(x)))
        summary["P>|t|"] = summary["P>|t|"].apply(lambda x: float("{:.3f}".format(x)))
        summary["[0.025"] = summary["[0.025"].apply(lambda x: float("{:.3f}".format(x)))
        summary["0.975]"] = summary["0.975]"].apply(lambda x: float("{:.3f}".format(x)))

        return summary
