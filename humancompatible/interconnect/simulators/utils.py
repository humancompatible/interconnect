import sympy as sp
import numpy as np


class Utils:
    @staticmethod
    def compute_lipschitz_constant_from_expression(expr: sp.Basic, interval: tuple = None):
        """
        Computes the lipschitz constant in the given expression.

        :param expr: The expression to compute the lipschitz constant from.
        :param interval: The known interval
        :type expr: sp.Basic
        :type interval: tuple

        :return: The lipschitz constant
        """
        lipschitz_const = 1.0
        if expr.is_Number:
            lipschitz_const = 0.0
        if expr.is_Add:
            args = [Utils.compute_lipschitz_constant_from_expression(arg, interval) for arg in expr.args]
            lipschitz_const = np.max(args)
        if expr.is_Mul:
            nums = [arg for arg in expr.args if arg.is_Number]
            lipschitz_const *= np.prod(nums)
            sub_expressions = [Utils.compute_lipschitz_constant_from_expression(arg, interval) for arg in expr.args if not arg.is_Number]
            lipschitz_const *= np.prod(sub_expressions)
        if expr.is_Pow:
            if interval is None:
                lipschitz_const = sp.oo
            else:
                l, r = interval
                variable = [arg for arg in expr.args if arg.is_symbol][0]
                df = sp.diff(expr, variable)
                lipschitz_const = max(df.subs(variable, l), df.subs(variable, r))
        return abs(lipschitz_const)
