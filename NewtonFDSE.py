"""
System Solver Class for Finite Difference Schrodinger Equation Simulation
and other helper methods.
"""
from typing import Dict, Any, Union, List
import numpy as np
import matplotlib.pyplot as plt


class SystemSolver:
    def __init__(self, params: Dict[str, Any], potential=None) -> None:
        self.params = params
        self._param_validation()

        # Define FD grid, grid spacing, and initialize ψ ≡ 0.
        self.grid, self.h = self.generate_grid()
        self.psi = np.zeros(len(self.grid))

        self.potential = potential if potential else lambda y: y ** 2

        if self.params["use_noumerov"]:
            self.FDEngine = self.noumerov
        else:
            self.FDEngine = self.three_point

    def _param_validation(self):
        if "N" not in self.params:
            self.params["N"] = 1000  # Number of grid segments w/ 2N+1 points
        if "ymax" not in self.params:
            self.params["ymax"] = 10  # Max +/- y axis value
        if "psi1" not in self.params:
            self.params["psi1"] = 1e-50  # ψ[0] = 0; ψ[1] = psi1 for confining v(y)
        if "tol" not in self.params:
            self.params["tol"] = 1e-10  # Percentage tolerance of solution
        if "maxiters" not in self.params:
            self.params["maxiters"] = 100  # Maximum number of iterations of Newton
        if "verbose" not in self.params:
            self.params["verbose"] = False  # Suppress output print statements
        if "use_noumerov" not in self.params:
            self.params["use_noumerov"] = False  # Use Noumerov FD scheme instead.
        if "ε_range" not in self.params:
            self.params["ε_range"] = [0.5, 1.5]  # Dimensionless energy search range

        if self.params["ε_range"][0] >= self.params["ε_range"][1]:
            raise ValueError(
                "Please provide valid range of ε to search over: [εmin, εmax], with εmin < εmax."
            )

    def generate_grid(
        self, N: int = None, ymax: float = None
    ) -> Union[np.ndarray, float]:
        """
        Function that returns a 1D grid of (2N+1) points from -ymax to ymax. By
        construction, grid[N] = 0.0 and (grid[N+1] - grid[N]) = h, the grid spacing.
        """

        N = N if N is not None else self.params["N"]
        ymax = ymax if ymax is not None else self.params["ymax"]

        grid = np.linspace(-ymax, ymax, 2 * N + 1)
        h = grid[N + 1] - grid[N]

        return grid, h

    def three_point(self, j: int, ε: float) -> float:
        """
        Finite Difference Engine w/ 3-Point Scheme. Computes state ψ_{j+1} 
        in terms of ψ_j and ψ_{j-1}. Assumes ψ_0 through ψ_j are given. 
        """
        h = self.h
        ψ = self.psi
        v = self.potential
        y_j = self.grid[j]

        return ψ[j] * ((v(y_j) - ε) * (h ** 2) + 2) - ψ[j - 1]

    def noumerov(self, j: int, ε: float) -> float:
        """
        Finite Difference Engine w/ Noumerov Scheme. Computes wavefunction
        ψ_{j+1} in terms of ψ_j and ψ_{j-1}. 
        """
        h = self.h
        ψ = self.psi
        v = self.potential
        y = self.grid

        return (
            24 * ψ[j]
            - 12 * ψ[j - 1]
            + h ** 2 * (10 * v(y[j]) * ψ[j] + v(y[j - 1]) * ψ[j - 1])
            - ε * h ** 2 * (10 * ψ[j] + ψ[j - 1])
        ) / (ε * h ** 2 + 12 - v(y[j + 1]) * h ** 2)

    def solve(self, ε_guess: float) -> Union[np.ndarray, List[Union[bool, float]]]:
        N = self.params["N"]

        # We initialized ψ ≡ 0. Now generate solution.
        self.psi[0] = 0
        self.psi[1] = self.params["psi1"]

        for n in range(1, N + 1):
            self.psi[n + 1] = self.FDEngine(n, ε_guess)

        # Parameter Δ (in %) measures wavefunction symmetry
        Δ = (self.psi[N + 1] - self.psi[N - 1]) / np.abs(self.psi[N - 1])

        # Converge if percentage difference Δ within allowed tolerance
        convergence = np.abs(Δ) < self.params["tol"]

        if convergence:
            # Use even symmetry to complete ψ. Note, rewrites ψ[N+1].
            for n in range(1, N + 1):
                self.psi[N + n] = self.psi[N - n]

            # Given a solution, normalize it!
            self.normalize_psi()

            if self.params["verbose"]:
                print(f"Convergence!? ε = {ε_guess}")
                # print(f"Δ: {Δ}, N+1: {self.psi[N + 1]}, N-1: {self.psi[N - 1]}")

        return self.psi, [convergence, np.sign(Δ)]

    def shootingNewton(self) -> Union[np.ndarray, float]:
        """
        Method that performs bisection search over ε_range = [εmin, εmax]
        to find a solution ε at which convergence is achieved. Returns 
        wavefunction and associated eigen-energy ε.
        """
        [εmin, εmax] = self.params["ε_range"]
        ε_sol = None

        for iter in range(self.params["maxiters"]):

            if self.params["verbose"]:
                print(f"Iter {iter}: Trying with [{εmin}, {εmax}]")

            ####### Try minimum of range #######
            _, [convergence, sgn] = self.solve(εmin)
            if convergence:
                ε_sol = εmin
                break

            if (sgn) < 0:
                raise RuntimeWarning(
                    "Provided εmin is too high! Try again with lower εmin."
                )

            ####### Try maximum of range #######
            _, [convergence, sgn] = self.solve(εmax)
            if convergence:
                ε_sol = εmax
                break

            if (sgn) > 0:
                raise RuntimeWarning(
                    "Provided εmax is too low! Try again with higher εmax."
                )

            ####### Determine new max/min #######
            εnew = (εmax + εmin) / 2
            _, [convergence, sgn] = self.solve(εnew)
            if convergence:
                ε_sol = εnew
                break

            # Bisect range using sgn of εnew solve!
            if sgn < 0:
                εmax = εnew
            elif sgn > 0:
                εmin = εnew

        if convergence:
            return self.psi, ε_sol
        else:
            print(f"No convergence after {iter} iterations!")
            return [], ε_sol

    def normalize_psi(self):
        """
        Normalize the wavefunction ψ(y) solution from shooting Newton.        
        """

        N = self.params["N"]

        # Integrate |ψ|²dy over entire grid
        norm = 0
        for n in range(0, 2 * N + 1):
            norm += np.abs(self.psi[n]) ** 2

        norm *= self.h  # Note: h = dy here!
        if norm == 0:
            raise ZeroDivisionError("Danger: wavefunction normalized to 0!")
        else:
            self.psi = self.psi / np.sqrt(norm)


def plot_wavefunction(grid, func, label=None, title=None, fig=None, ax=None):
    fig = fig if fig is not None else plt.figure(figsize=(4, 3), dpi=200)
    ax = ax if ax is not None else fig.subplots()

    ax.plot(grid, func, label=label)
    ax.set_xlabel("Position y")
    ax.set_ylabel("ψ(y)")
    ax.set_title(title, fontsize=8)
    if label:
        ax.legend(fontsize=7)


def plot_error(
    xvals,
    yvals,
    fit,
    fitparams,
    label=None,
    fitlabel=None,
    title=None,
    fig=None,
    ax=None,
):
    fig = fig if fig is not None else plt.figure(figsize=(4, 3), dpi=200)
    ax = ax if ax is not None else fig.subplots()

    ax.plot(xvals, yvals, "grey", linestyle="-", marker=".", label=label)
    ax.set_xlabel("Finite Difference Step Size h", fontsize=8)
    ax.set_ylabel("Error w(h) = |ψ - ϕ0|", fontsize=8)
    ax.set_title(title, fontsize=8)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    ax.plot(xvals, fit(xvals, *fitparams), "r-", label=fitlabel % tuple(fitparams))

    if label or fitlabel:
        ax.legend(fontsize=7)


def plot_error_log(
    xvals,
    yvals,
    fit,
    fitparams,
    label=None,
    fitlabel=None,
    title=None,
    fig=None,
    ax=None,
):
    fig = fig if fig is not None else plt.figure(figsize=(4, 3), dpi=200)
    ax = ax if ax is not None else fig.subplots()

    ax.loglog(10 ** xvals, 10 ** yvals, "grey", linestyle="-", marker=".", label=label)
    ax.set_xlabel("Finite Difference Step Size h", fontsize=8)
    ax.set_ylabel("Error w(h) = |ψ - ϕ0|", fontsize=8)
    ax.set_title(title, fontsize=8)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    ax.loglog(
        10 ** xvals,
        10 ** fit(xvals, *fitparams),
        "r-",
        label=fitlabel % tuple(fitparams),
    )

    if label or fitlabel:
        ax.legend(fontsize=7)
