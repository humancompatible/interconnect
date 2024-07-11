import sympy as sp
import numpy as np


class Utils:
    @staticmethod
    def compute_lipschitz_constant_from_expression(expr: sp.Basic) -> int:
        """
        Computes the lipschitz constant in the given expression.

        :param expr: The expression to compute the lipschitz constant from.
        :type expr: sp.Basic

        :return: The lipschitz constant
        :rtype: int
        """
        L = 1
        if expr.is_Number:
            return 0
        if expr.is_Add:
            args = [Utils.compute_lipschitz_constant_from_expression(arg) for arg in expr.args]
            L = np.max(args)
            return L
        if expr.is_Mul:
            nums = [arg for arg in expr.args if arg.is_Number]
            L *= np.prod(nums)
            exprs = [Utils.compute_lipschitz_constant_from_expression(arg) for arg in expr.args if not arg.is_Number]
            L *= np.prod(exprs)
        if expr.is_Pow:
            raise NotImplementedError
        return L
