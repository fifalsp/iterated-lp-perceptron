"""Particle approximation of the population recursion (state evolution).

The recursion of the paper describes, round by round, the limiting law of a
free column history (X, R) and of a row history (U, Lambda). Round t solves a
variational problem whose stationarity (KKT) equations fix the clipping scale
tau_t, the innovation scale rho_t of the row margins, and the history
coefficients eta_x, eta_r, eta_lambda. The new laws are then

    Y_t      = clip((S_t + eta_x . X + eta_r . R) / tau_t, -1, 1)   (free columns)
    U_t      = max(kappa_t, Z_t - eta_lambda . Lambda)                (rows)
    Lambda_t = (kappa_t - Z_t + eta_lambda . Lambda)_+ / V_t

and q_t = q_{t-1} P(|Y_t| < 1).

Numerical method
----------------
* Free column histories are represented by 2**column_power equally weighted
  particles, row histories by 2**row_power particles in each stratum, where a
  stratum collects the rows whose last active round is the same.
* The fresh Gaussian variables of each round are integrated in closed form
  (clipped Gaussian moments for columns, hinge moments for rows).
* The KKT equations are solved with scipy.optimize.least_squares (bounded
  trust region, analytic Jacobian) in coordinates whitened by Cholesky factors
  of the history Gram matrices.
* After an accepted round, column histories are reweighted by their survival
  probability and resampled with scrambled Sobol points, and the new
  coordinate is drawn from the truncated Gaussian law. Row strata are
  resampled in the same way. The Gram contribution of frozen coordinates is
  kept separately ("cohort" Gram), so resampling only affects the free part.

All of this is floating-point arithmetic without error bounds. The
residuals measure only how well the discretized equations are solved.

Notation used in the code
-------------------------
X, U, Lam         particle histories of free columns, row margins, row multipliers
row_weights       weights of the row particles (they sum to one)
gram_x            Gram matrix of the primal history (Q_x in the paper's notation),
                  equal to frozen_gram + free_gram
free_gram         contribution q * E[X X^T] of the free coordinates
frozen_gram       contribution of coordinates frozen in earlier rounds
gram_lambda       alpha * E[Lam Lam^T] (Q_lambda)
cross_gram        alpha * E[Lam U^T]   (Gamma)
cov_h, cov_hr     covariance of the unclipped Gaussian scores H and their
                  cross-covariance with R = -V
"""
import math
import time

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from .clipped_gaussian import clipped_moments, normal_pdf

# A round is accepted if the largest scaled KKT residual is below this value,
# the value is positive and the free fraction decreases.
ACCEPTANCE_RESIDUAL = 2e-6

# Initial guesses (tau, last eta_r coefficient) tried after the warm start.
FIXED_STARTS = [(2.15, .285), (4., .5), (1., .2)]


def first_round(alpha, kappa):
    """Closed-form first round (one LP from the cube, no history).

    The first LP reduces to a scalar equation for the clipping threshold a:
    q = 2 Phi(a) - 1 is the free fraction and must equal the fraction alpha *
    P(row active) of tight rows. Returns the scalar quantities that start the
    recursion: q_1, rho_1, the Gram entries, the width c_1 = q_1 / a, the value
    V_1 and gamma_1, and the row moments Q_lambda and Gamma.
    """
    def scalar(a):
        q = 2 * ndtr(a) - 1
        T = q - 2 * a * normal_pdf(a)
        v = T / a ** 2 + 1 - q
        return q, T, v

    a = brentq(lambda a: scalar(a)[0] - alpha * ndtr(kappa / math.sqrt(scalar(a)[2])), 1e-5, 5, xtol=5e-15)
    q, T, v = scalar(a)
    rho = math.sqrt(v)
    h = kappa / rho
    hinge1 = kappa * ndtr(h) + rho * normal_pdf(h)
    hinge2 = (kappa * kappa + v) * ndtr(h) + kappa * rho * normal_pdf(h)
    c = q / a
    value2 = c * c - alpha * hinge2
    if value2 <= 0:
        raise ValueError('First-round population value is not positive.')
    value = math.sqrt(value2)
    gamma = math.sqrt(alpha * hinge2 / value2)
    return dict(a=float(a), q=float(q), rho=rho, Qx=float(v), QFxx=float(T / a ** 2),
                width=float(c), value_squared=float(value2), value=value, gamma=gamma,
                Qlambda=gamma * gamma, Gamma=float(alpha * kappa * hinge1 / value))


class Population:
    """Particle state of the population recursion after the first round.

    Parameters
    ----------
    alpha : float
        Constraint density M/N.
    kappa1 : float
        First margin kappa_1.
    column_power, row_power : int
        2**column_power column particles and 2**row_power row particles per
        last-active stratum.
    seed : int
        Base seed of all scrambled Sobol streams.
    """

    def __init__(self, alpha, kappa1, column_power, row_power, seed):
        self.alpha = alpha
        self.column_power = column_power
        self.row_power = row_power
        self.seed = seed
        self.n = 2 ** column_power
        self.rows_per_stratum = 2 ** row_power
        self.first = first_round(alpha, kappa1)
        f = self.first
        self.q = f['q']

        # Covariance of the unclipped Gaussian score H_1 and its covariance with R_1 = -V.
        self.cov_h = np.array([[1 / f['a'] ** 2]])
        self.cov_hr = np.array([[-1 / (f['a'] * math.sqrt(1 + f['gamma'] ** 2))]])

        # Free coordinates after round 1: x = H/a with H a standard Gaussian
        # conditioned on |H| < a, sampled by inverse transform.
        u = qmc.Sobol(1, scramble=True, seed=seed + 1).random_base2(column_power)[:, 0]
        self.X = (ndtri(ndtr(-f['a']) + u * f['q']) / f['a'])[:, None]

        # Rows after round 1, in two strata: inactive (stratum 0) and active (stratum 1).
        active_probability = ndtr(kappa1 / f['rho'])
        points = qmc.Sobol(1, scramble=True, seed=seed + 2).random_base2(row_power)[:, 0]
        g = np.r_[-ndtri((1 - active_probability) * points), ndtri(active_probability * points)]
        self.U = np.maximum(kappa1, f['rho'] * g)[:, None]
        self.Lam = (np.maximum(kappa1 - f['rho'] * g, 0) / f['value'])[:, None]
        self.row_weights = np.r_[np.full(self.rows_per_stratum, (1 - active_probability) / self.rows_per_stratum),
                                 np.full(self.rows_per_stratum, active_probability / self.rows_per_stratum)]
        self.row_strata = np.repeat([0, 1], self.rows_per_stratum)

        # Gram matrix of the primal history: frozen part plus free part.
        self.frozen_gram = np.array([[1 - self.q]])
        self.gram_x = self.frozen_gram + self.q * (self.X.T @ self.X) / self.n
        self.last = None
        self.records = []

    # ------------------------------------------------------------------
    # One round: set up the whitened history, evaluate the KKT residual.
    # ------------------------------------------------------------------

    def prepare(self, kappa):
        """Compute the history quantities that stay fixed while round t is solved."""
        self.kappa = kappa
        self.p = self.X.shape[1]
        p = self.p
        self.free_gram = self.q * (self.X.T @ self.X) / self.n
        self.gram_x = self.frozen_gram + self.free_gram
        self.gram_lambda = self.alpha * (self.Lam.T @ (self.row_weights[:, None] * self.Lam))
        self.cross_gram = self.alpha * (self.Lam.T @ (self.row_weights[:, None] * self.U))
        self.chol_x = np.linalg.cholesky(self.gram_x)
        self.chol_lambda = np.linalg.cholesky(self.gram_lambda)
        self.inv_chol_x = np.linalg.inv(self.chol_x)
        self.inv_chol_lambda = np.linalg.inv(self.chol_lambda)

        # Whitened histories.
        self.X_white = self.X @ self.inv_chol_x.T
        self.U_white = self.U @ self.inv_chol_x.T
        self.Lam_white = self.Lam @ self.inv_chol_lambda.T
        self.cross_white = self.inv_chol_lambda @ self.cross_gram @ self.inv_chol_x.T
        self.B_white = self.U_white - self.Lam_white @ self.cross_white

        # Regression of the objective history R on the score history H (exact
        # Gaussian conditioning given the surviving primal history).
        self.regression = np.linalg.solve(self.cov_h, self.cov_hr).T
        self.regression_cov = np.eye(p) - self.regression @ self.cov_hr
        self.regression_cov = (self.regression_cov + self.regression_cov.T) / 2
        if np.linalg.eigvalsh(self.regression_cov)[0] < -1e-9:
            raise ValueError('Indefinite conditional regression covariance.')
        self.R_mean_white = self.X @ self.regression.T @ self.inv_chol_lambda.T
        self.regression_cov_white = self.inv_chol_lambda @ self.regression_cov @ self.inv_chol_lambda.T
        self.old_row_target = self.inv_chol_lambda @ self.cross_gram[:, -1]
        self.last_chol_row = self.chol_x[-1, :]
        self.calls = 0

    def evaluate(self, z, derivatives=False):
        """KKT residual of round t at z = (log tau, whitened eta_x, whitened eta_r).

        The residual has three blocks: the row-moment compatibility equations,
        the balance q_t = alpha * P(row active), and stationarity in eta_x.
        With derivatives=True the analytic Jacobian is returned as well.
        """
        self.calls += 1
        p = self.p
        tau = math.exp(z[0])
        vx = z[1:1 + p]
        vr = z[1 + p:]
        conditional_variance = float(vr @ self.regression_cov_white @ vr)
        if conditional_variance < -1e-9:
            raise ValueError('Negative conditional regression variance.')
        sig = math.sqrt(1 + max(0., conditional_variance))
        score = self.X_white @ vx + self.R_mean_white @ vr

        # Columns: clipped Gaussian moments of the new coordinate.
        ey, ey2, mass, first_inside, second_inside = clipped_moments(score, sig, tau)
        qnew = self.q * float(mass.mean())
        c = qnew / tau
        s = self.q * float(ey2.mean())
        aw = self.q * (self.X_white.T @ (ey - self.X[:, -1])) / self.n
        bw = self.q * (self.R_mean_white.T @ (ey - self.X[:, -1])) / self.n + self.regression_cov_white @ vr * c
        a = self.chol_x @ aw
        b = self.chol_lambda @ bw
        rho2 = s - self.free_gram[-1, -1] - 2 * a[-1] - aw @ aw
        if rho2 <= 0 or c <= 0:
            raise ValueError('Nonpositive transverse variance or width.')
        rho = math.sqrt(rho2)

        # Rows: hinge moments of the pre-projection margin Z_t ~ N(mean, rho^2).
        meanz = self.U[:, -1] + self.U_white @ aw + self.Lam_white @ (bw - self.cross_white @ aw)
        correction = c * (self.Lam_white @ vr)
        mean = meanz - correction
        deficit = self.kappa - mean
        h = deficit / rho
        prob = ndtr(h)
        ez = deficit * prob + rho * normal_pdf(h)
        ez2 = (deficit ** 2 + rho2) * prob + deficit * rho * normal_pdf(h)
        ew = mean + ez
        er = ez - correction
        d2 = self.alpha * float(self.row_weights @ (ez2 - 2 * correction * ez + correction ** 2))
        value2 = c * c - d2

        residual_row = (self.alpha * (self.Lam_white.T @ (self.row_weights * ew)) - self.old_row_target - bw) / max(self.q, 1e-12)
        residual_balance = (qnew - self.alpha * float(self.row_weights @ prob)) / max(self.q, 1e-9)
        residual_stationarity = vx - tau * (self.last_chol_row + aw) - self.alpha * (self.B_white.T @ (self.row_weights * er)) / c
        residual = np.concatenate((residual_row, [residual_balance], residual_stationarity))

        jacobian = None
        if derivatives:
            # Exact derivatives of the clipped and hinge moments of the finite
            # particle system (not of its sampling error).
            k = 2 * p + 1
            lo = (-tau - score) / sig
            hi = (tau - score) / sig
            pl = normal_pdf(lo)
            ph = normal_pdf(hi)
            diff = pl - ph
            edge = lo * pl - hi * ph
            first = first_inside
            second = second_inside
            dscore = np.column_stack((np.zeros(self.n), self.X_white, self.R_mean_white))
            dsig = np.r_[0., np.zeros(p), self.regression_cov_white @ vr / sig]
            dtau = np.zeros(k)
            dtau[0] = tau
            dmass = diff[:, None] / sig * dscore + edge[:, None] / sig * dsig + (pl + ph)[:, None] / sig * dtau
            dey = mass[:, None] / tau * dscore + diff[:, None] / tau * dsig - first[:, None] / tau ** 2 * dtau
            dey2 = (2 * first[:, None] / tau ** 2 * dscore
                    + 2 * (score * diff + sig * (mass + edge))[:, None] / tau ** 2 * dsig
                    - 2 * second[:, None] / tau ** 3 * dtau)
            dq = self.q * dmass.mean(axis=0)
            dc = dq / tau - c / tau * dtau
            ds = self.q * dey2.mean(axis=0)
            daw = self.q * (self.X_white.T @ dey) / self.n
            dbw = self.q * (self.R_mean_white.T @ dey) / self.n + np.outer(self.regression_cov_white @ vr, dc)
            dbw[:, 1 + p:] += c * self.regression_cov_white
            drho = (ds - 2 * self.chol_x[-1, :] @ daw - 2 * aw @ daw) / (2 * rho)
            dmean = self.U_white @ daw + self.Lam_white @ (dbw - self.cross_white @ daw)
            dcorrection = np.outer(self.Lam_white @ vr, dc)
            dcorrection[:, 1 + p:] += c * self.Lam_white
            dmean -= dcorrection
            dez = -prob[:, None] * dmean + normal_pdf(h)[:, None] * drho
            dew = dmean + dez
            der = dez - dcorrection
            drow = (self.alpha * (self.Lam_white.T @ (self.row_weights[:, None] * dew)) - dbw) / max(self.q, 1e-12)
            dh = (-dmean - h[:, None] * drho) / rho
            dbalance = (dq - self.alpha * ((self.row_weights * normal_pdf(h)) @ dh)) / max(self.q, 1e-9)
            dstat = -np.outer(self.last_chol_row + aw, dtau) - tau * daw - self.alpha * (self.B_white.T @ (self.row_weights[:, None] * der)) / c
            dstat += np.outer(self.alpha * (self.B_white.T @ (self.row_weights * er)) / (c * c), dc)
            dstat[:, 1:1 + p] += np.eye(p)
            jacobian = np.vstack((drow, dbalance, dstat))

        return dict(residual=residual, jacobian=jacobian, tau=tau,
                    eta_x=self.inv_chol_x.T @ vx, eta_r=self.inv_chol_lambda.T @ vr,
                    eta_lambda=c * self.inv_chol_lambda.T @ vr, rho=rho, a=a, b=b, s=s, width=c,
                    value_squared=value2, gamma=math.sqrt(max(0, d2) / value2) if value2 > 0 else None,
                    q=qnew, mean_score=score, sd_score=sig, mass=mass,
                    free_second_mass=self.q * float(second_inside.mean()) / tau ** 2,
                    free_squared_norm=self.q * float(second_inside.mean()) / (tau ** 2 * qnew),
                    row_mean=mean, row_cost_squared=d2, row_probability=prob,
                    active_probability=float(self.row_weights @ prob),
                    row_residual_max=float(np.max(np.abs(residual_row))),
                    balance_residual=abs(residual_balance),
                    stationarity_residual_max=float(np.max(np.abs(residual_stationarity))))

    def residual(self, z):
        """Residual for least_squares; a large constant if evaluation fails."""
        try:
            r = self.evaluate(z)['residual']
            if np.all(np.isfinite(r)):
                return r
        except (ValueError, OverflowError, FloatingPointError):
            pass
        return np.full(2 * self.p + 1, 1e8)

    def jacobian(self, z):
        """Jacobian for least_squares; a large diagonal if evaluation fails."""
        try:
            j = self.evaluate(z, derivatives=True)['jacobian']
            if np.all(np.isfinite(j)):
                return j
        except (ValueError, OverflowError, FloatingPointError):
            pass
        return np.eye(2 * self.p + 1) * 1e8

    # ------------------------------------------------------------------
    # Solving a round and advancing the particle state.
    # ------------------------------------------------------------------

    def solve(self, kappa, max_nfev=150):
        """Solve round t at margin kappa and, if accepted, advance the state.

        Starts from the previous solution shifted one round forward, then from
        a few fixed guesses, and stops at the first accepted root. Returns the
        round record (also appended to self.records).
        """
        self.prepare(kappa)
        p = self.p
        starts = []
        if self.last is not None:
            ex = np.r_[0., self.last['eta_x']]
            er = np.r_[0., self.last['eta_r']]
            starts.append(np.r_[math.log(self.last['tau']), self.chol_x.T @ ex, self.chol_lambda.T @ er])
        for tau, last_eta_r in FIXED_STARTS:
            ex = np.zeros(p)
            ex[-1] = tau
            er = np.zeros(p)
            er[-1] = last_eta_r
            starts.append(np.r_[math.log(tau), self.chol_x.T @ ex, self.chol_lambda.T @ er])

        attempts = []
        best = None
        started = time.monotonic()
        for start in starts:
            fit = least_squares(self.residual, start, jac=self.jacobian, max_nfev=max_nfev, x_scale='jac',
                                xtol=2e-9, ftol=2e-9, gtol=2e-9, diff_step=1e-5,
                                bounds=(np.r_[-20., np.full(2 * p, -1e4)], np.r_[10., np.full(2 * p, 1e4)]))
            try:
                result = self.evaluate(fit.x)
            except (ValueError, OverflowError):
                continue
            error = float(np.max(np.abs(result['residual'])))
            accepted = error < ACCEPTANCE_RESIDUAL and result['value_squared'] > 0 and 0 < result['q'] < self.q
            attempts.append(dict(nfev=int(fit.nfev), error=error, positive_value=result['value_squared'] > 0,
                                 accepted=accepted))
            if best is None or error < best[0]:
                best = (error, result, fit)
            if accepted:
                best = (error, result, fit)
                break
        if best is None:
            raise RuntimeError('All root attempts failed.')
        error, result, fit = best
        accepted = error < ACCEPTANCE_RESIDUAL and result['value_squared'] > 0 and 0 < result['q'] < self.q

        record = {key: result[key] for key in [
            'tau', 'rho', 's', 'width', 'value_squared', 'gamma', 'q', 'sd_score', 'row_cost_squared',
            'free_second_mass', 'free_squared_norm', 'active_probability', 'row_residual_max',
            'balance_residual', 'stationarity_residual_max']}
        record.update(
            round=p + 1,
            kappa=kappa,
            previous_q=self.q,
            survival_ratio=result['q'] / self.q,
            accepted=accepted,
            maximum_scaled_residual=error,
            elapsed_seconds=time.monotonic() - started,
            root_evaluations=self.calls,
            attempts=attempts,
            gram_condition_x=float(np.linalg.cond(self.gram_x)),
            gram_condition_lambda=float(np.linalg.cond(self.gram_lambda)),
            frozen_gram_min_eigenvalue=float(np.linalg.eigvalsh(self.gram_x - self.free_gram)[0]),
            previous_diagonal_drift=float(self.gram_x[-1, -1] - self.free_gram[-1, -1] - (1 - self.q)),
            previous_diagonal_drift_over_q=float((self.gram_x[-1, -1] - self.free_gram[-1, -1] - (1 - self.q)) / self.q),
            regression_covariance_min_eigenvalue=float(np.linalg.eigvalsh(self.regression_cov)[0]),
            effective_tau=result['tau'] / result['sd_score'],
            parameters={key: result[key].tolist() for key in ['a', 'b', 'eta_x', 'eta_r', 'eta_lambda']})
        if accepted:
            self.advance(result, record)
        self.records.append(record)
        return record

    def advance(self, r, record):
        """Move the particle state from round t-1 to round t after an accepted solve."""
        p = self.p
        tau = r['tau']
        ex = r['eta_x']
        er = r['eta_r']
        gamma = r['gamma']

        # Frozen Gram: extend old frozen histories by repeating their last
        # coordinate, and add the exact conditional moments of coordinates
        # that freeze in this round (computed before resampling).
        frozen_next = np.empty((p + 1, p + 1))
        frozen_next[:p, :p] = self.frozen_gram
        frozen_next[:p, p] = self.frozen_gram[:, -1]
        frozen_next[p, :p] = self.frozen_gram[-1, :]
        frozen_next[p, p] = self.frozen_gram[-1, -1]
        outside = 1 - r['mass']
        sign = ndtr((r['mean_score'] - tau) / r['sd_score']) - ndtr((-tau - r['mean_score']) / r['sd_score'])
        frozen_next[:p, :p] += self.q * (self.X.T @ (outside[:, None] * self.X)) / self.n
        freeze_cross = self.q * (self.X.T @ sign) / self.n
        frozen_next[:p, p] += freeze_cross
        frozen_next[p, :p] += freeze_cross
        frozen_next[p, p] += self.q * float(outside.mean())
        frozen_next = (frozen_next + frozen_next.T) / 2

        # Covariances of the unclipped Gaussian scores, extended by one round.
        cov_h_new = (self.cov_h @ ex + self.cov_hr @ er) / tau
        cov_r_new = (self.cov_hr.T @ ex + er) / tau
        var_new = (1 + ex @ self.cov_h @ ex + 2 * ex @ self.cov_hr @ er + er @ er) / tau ** 2
        cov_h = np.empty((p + 1, p + 1))
        cov_h[:p, :p] = self.cov_h
        cov_h[:p, p] = cov_h_new
        cov_h[p, :p] = cov_h_new
        cov_h[p, p] = var_new
        cov_hr = np.zeros((p + 1, p + 1))
        cov_hr[:p, :p] = self.cov_hr
        cov_hr[p, :p] = cov_r_new
        cov_hr[p, p] = -1 / (tau * math.sqrt(1 + gamma * gamma))
        record['unclipped_score_covariance'] = cov_h.tolist()
        record['unclipped_score_objective_covariance'] = cov_hr.tolist()
        innovation = np.diag(np.linalg.cholesky(cov_h))
        record['innovation_box_product'] = float(np.prod(2 * ndtr(1 / innovation) - 1))
        record['single_time_box_upper_bound'] = float(np.min(2 * ndtr(1 / np.sqrt(np.diag(cov_h))) - 1))
        self.cov_h = cov_h
        self.cov_hr = cov_hr

        # Gram matrix extended by the recursion formula, before resampling
        # (kept only to report its discrepancy with the resampled Gram below).
        cross = self.gram_x[:, -1] + r['a']
        old_diagonal = self.gram_x[-1, -1]
        gram_formula = np.empty((p + 1, p + 1))
        gram_formula[:p, :p] = self.gram_x
        gram_formula[:p, p] = cross
        gram_formula[p, :p] = cross
        gram_formula[p, p] = r['s'] + old_diagonal - self.free_gram[-1, -1]

        # Column resampling: weight each history by its survival probability,
        # resample with scrambled Sobol points, and draw the new coordinate
        # from the truncated Gaussian law by inverse transform.
        weights = r['mass'] / r['mass'].sum()
        cdf = np.cumsum(weights)
        cdf[-1] = 1
        points = qmc.Sobol(2, scramble=True, seed=self.seed + 100 * p).random_base2(self.column_power)
        index = np.searchsorted(cdf, points[:, 0])
        score = r['mean_score'][index]
        sd = r['sd_score']
        lo = ndtr((-tau - score) / sd)
        hi = ndtr((tau - score) / sd)
        fresh = ndtri(np.clip(lo + points[:, 1] * (hi - lo), 1e-15, 1 - 1e-15))
        y = (score + sd * fresh) / tau
        self.X = np.column_stack((self.X[index], np.clip(y, -1, 1)))

        self.resample_rows(r)
        new_lambda = self.Lam[:, -1]
        record['active_row_particles'] = int(np.count_nonzero(new_lambda))
        record['row_particles'] = len(self.row_weights)
        cross = self.alpha * (self.Lam[:, :p].T @ (self.row_weights * new_lambda))
        record['row_gram_schur_over_gamma_squared'] = float(
            (self.alpha * (self.row_weights @ (new_lambda ** 2)) - cross @ np.linalg.solve(self.gram_lambda, cross))
            / (gamma * gamma))
        self.q = r['q']
        self.last = r

        self.frozen_gram = frozen_next
        resampled_gram = self.frozen_gram + self.q * (self.X.T @ self.X) / self.n
        record['gram_update_discrepancy'] = float(np.max(np.abs(resampled_gram - gram_formula)))
        self.gram_x = resampled_gram
        record['frozen_diagonal_identity_error'] = float(self.frozen_gram[-1, -1] - (1 - self.q))
        record['updated_frozen_gram_min_eigenvalue'] = float(np.linalg.eigvalsh(self.frozen_gram)[0])
        record['updated_gram_min_eigenvalue'] = float(np.linalg.eigvalsh(self.gram_x)[0])
        record['updated_gram_max_diagonal'] = float(np.max(np.diag(self.gram_x)))
        record['free_squared_norm_after_resampling'] = float(np.mean(self.X[:, -1] ** 2))
        record['updated_diagonal_drift'] = float(self.gram_x[-1, -1] - self.q * np.mean(self.X[:, -1] ** 2) - (1 - self.q))
        record['updated_diagonal_drift_over_q'] = record['updated_diagonal_drift'] / self.q
        record['updated_gram_diagonal'] = float(self.gram_x[-1, -1])

    def resample_rows(self, r):
        """Resample the row strata and append the round-t margins and multipliers.

        Rows inactive in round t keep their previous last-active stratum; rows
        active in round t (from all old strata) form a new stratum. Each
        stratum is resampled to 2**row_power particles with scrambled Sobol
        points, and the new Gaussian innovation is drawn conditionally on
        being active or inactive.
        """
        p = self.p
        active_probability = r['row_probability']
        new_U = []
        new_Lam = []
        new_weights = []
        new_strata = []
        counter = 0
        pieces = [(int(g), np.flatnonzero(self.row_strata == g), False) for g in np.unique(self.row_strata)]
        pieces.append((p + 1, np.arange(len(self.row_weights)), True))
        for stratum, old, active in pieces:
            probabilities = active_probability[old] if active else 1 - active_probability[old]
            weights = self.row_weights[old] * probabilities
            mass = float(weights.sum())
            if mass <= 0:
                continue
            cdf = np.cumsum(weights / mass)
            cdf[-1] = 1
            points = qmc.Sobol(2, scramble=True, seed=self.seed + 10000 * p + counter).random_base2(self.row_power)
            counter += 1
            index = old[np.searchsorted(cdf, points[:, 0])]
            if active:
                g = ndtri(np.maximum(active_probability[index] * points[:, 1], 1e-300))
            else:
                g = -ndtri(np.maximum((1 - active_probability[index]) * points[:, 1], 1e-300))
            raw = r['row_mean'][index] + r['rho'] * g
            u_new = np.maximum(self.kappa, raw)
            lambda_new = np.maximum(self.kappa - raw, 0) / math.sqrt(r['value_squared'])
            new_U.append(np.column_stack((self.U[index], u_new)))
            new_Lam.append(np.column_stack((self.Lam[index], lambda_new)))
            new_weights.append(np.full(self.rows_per_stratum, mass / self.rows_per_stratum))
            new_strata.append(np.full(self.rows_per_stratum, stratum))
        self.U = np.vstack(new_U)
        self.Lam = np.vstack(new_Lam)
        self.row_weights = np.concatenate(new_weights)
        self.row_weights /= self.row_weights.sum()
        self.row_strata = np.concatenate(new_strata)
