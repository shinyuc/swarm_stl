import mosek.fusion as mf
import numpy as np

def solve_example(a, x, b, M_val=1000.0):
    """
    Solve the optimization problem:
      maximize c
      subject to:
         1) log(det(A)) >= 0   (i.e. det(A) >= 1, A is PSD)
         2) a^T A x + b + (1 - z) * M + c >= 0
         3) z in {0,1}
         4) A is 2x2
    """

    # Create a model
    model = mf.Model("logdet_MILP_example")

    # 1) Define variables
    # A is a 2x2 positive semidefinite matrix variable
    A = model.variable("A", [2, 2], mf.Domain.inPSDCone())

    # z is a binary variable
    z = model.variable("z", mf.Domain.binary())

    # c is an unrestricted real variable (we want to maximize it)
    c_var = model.variable("c", mf.Domain.unbounded())

    # If x is also a decision variable, define it instead of using a constant:
    # x_var = model.variable("x", 2, mf.Domain.unbounded())
    # Then use x_var in the expression below. If x is given (constant), do as below.

    # 2) log(det(A)) >= 0
    #    MOSEK Fusion provides a direct way to constrain logDet(A).
    #    We impose logDet(A) >= 0.0  => det(A) >= 1.
    model.constraint("logdetA",
                     mf.Expr.logDet(A),
                     mf.Domain.greaterThan(0.0))

    # 3) a^T A x + b + (1 - z)*M + c >= 0
    #
    #    We'll build this expression step by step:
    #    a^T A x is dot(a, A*x).
    #    Then add b, c, and M*(1 - z) = M - M*z.
    #
    #    Note: 'a' and 'x' should be NumPy arrays or lists of length 2 (for 2D).
    #
    # First, a^T A x:
    ax_expr = mf.Expr.dot(a, mf.Expr.mul(A, x))  # -> scalar expression

    # Add the constant b, the variable c, and M*(1 - z)
    lhs_expr = mf.Expr.add(ax_expr, b)
    lhs_expr = mf.Expr.add(lhs_expr, c_var)
    lhs_expr = mf.Expr.add(lhs_expr,
                           mf.Expr.sub(M_val,
                                       mf.Expr.mul(M_val, z)))  # (1 - z)*M

    # Now create the constraint
    model.constraint("linear_constraint", lhs_expr, mf.Domain.greaterThan(0.0))

    # 4) Objective: maximize c
    model.objective("obj", mf.ObjectiveSense.Maximize, c_var)

    # Solve the problem
    model.solve()

    # Retrieve solutions
    A_sol = A.level()       # Flattened [2*2]
    z_sol = z.level()[0]    # Single binary var
    c_sol = c_var.level()[0]

    # Reshape A_sol into a 2x2 matrix
    A_sol = np.array(A_sol).reshape((2, 2))

    # Dispose model
    model.dispose()

    return A_sol, z_sol, c_sol

# -------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Given data
    a = [1.0, 2.0]     # shape (2,)
    x = [0.5, -1.0]    # shape (2,) or treat as variable inside the model
    b = 3.0            # scalar
    big_M = 1000.0     # big-M

    # Solve
    A_star, z_star, c_star = solve_example(a, x, b, big_M)

    print("Optimal A:\n", A_star)
    print("Optimal z:", z_star)
    print("Optimal c:", c_star)
