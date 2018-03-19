from __future__ import division
import numpy as np
from scipy.stats import norm

class BlackScholes:

    class OptionPrice:
        def d1(self, s, k, r, t, o):
            return (np.log(s / k) + (r + 0.5 * (o ** 2)) * t) / (o * np.sqrt(t))

        def d2(self, s, k, r, t, o):
            d1 = self.d1(s, k, r, t, o)
            return d1 - (o * np.sqrt(t))

        def returnoptionprice(self, s, k, r, t, o):
            d1 = self.d1(s, k, r, t, o)
            d2 = self.d2(s, k, r, t, o)
            return s * norm.cdf(d1) - k * np.exp((-1) * r * t) * norm.cdf(d2)

    class Greeks(OptionPrice):
        def delta(self, s, k, r, t, o):
            z = self.d1(s, k, r, t, o)
            return norm.cdf(z)

        def gamma(self, s, k, r, t, o):
            z = self.d1(s, k, r, t, o)
            return norm.pdf(z) / (s * o * np.sqrt(t))

        def vega(self, s, k, r, t, o):
            z = self.d1(s, k, r, t, o)
            return s * norm.pdf(z) * np.sqrt(t)

        def deltaapprox(self, s, k, r, t, o, d):
            one = self.returnoptionprice(s + d, k, r, t, o)
            two = self.returnoptionprice(s - d, k, r, t, o)
            return (1 / (2 * d)) * (one - two)

        def gammaapprox(self, s, k, r, t, o, d):
            one = self.returnoptionprice(s, k, r, t, o + d)
            two = self.returnoptionprice(s, k, r, t, o - d)
            return (1 / (2 * d)) * (one - two)

        def vegaapprox(self, s, k, r, t, o, d):
            one = self.returnoptionprice(s + d, k, r, t, o)
            two = self.returnoptionprice(s, k, r, t, o)
            three = self.returnoptionprice(s - d, k, r, t, o)
            return (1 / (d ** 2)) * (one - 2 * two + three)

    class PriceProcess:
        def St(self, S, u, o, t, Wt):
            exponent = (u - 0.5 * o ** 2) * t + o * Wt
            return S * np.exp(exponent)

        def dSt(self, S, u, o, t, Wt):  ##incomplete
            St = self.St(S, u, o, t, Wt)
            return St