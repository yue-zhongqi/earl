import random
import operator
import ast
from num2words import num2words
OPS = ['+', '-', '*', '/']

def random_number(max_digits):
    """first generates a random number between 1 to max_digits uniformly, then generate a random number of that number of digits."""
    num_digits = random.randint(1, max_digits)
    lower = 1 if num_digits == 1 else 10 ** (num_digits - 1)
    upper = 10 ** num_digits - 1
    return str(random.randint(lower, upper))

def random_split(n):
    assert n >= 0
    a = random.randint(0, n)
    b = n - a
    return a, b

def rand_num2words(num, p):
    if random.random() < p:
        return num2words(num, lang='en')
    else:
        return str(num)

def describe(left_desc, op, right_desc, nlp_p=0.0):
    if random.random() > nlp_p:
        return f"{left_desc} {op} {right_desc}"
    else:
        if op == '+':
            return f"addition of {left_desc} and {right_desc}"
        elif op == '-':
            return f"subtraction of {right_desc} from {left_desc}"
        elif op == '*':
            return f"multiplication of {left_desc} and {right_desc}"
        elif op == '/':
            return f"division of {left_desc} by {right_desc}"
        else:
            raise ValueError(f"Unknown operator: {op}")

def simple_describe(left_desc, op, right_desc, nlp_p=0.0):
    if random.random() > nlp_p:
        return f"{left_desc} {op} {right_desc}"
    else:
        if op == '+':
            return f"{left_desc} plus {right_desc}"
        elif op == '-':
            return f"{left_desc} minus {right_desc}"
        elif op == '*':
            return f"{left_desc} times {right_desc}"
        elif op == '/':
            return f"{left_desc} divided by {right_desc}"
        else:
            raise ValueError(f"Unknown operator: {op}")

def generate_random_expression(max_digits=2, max_ops=3, outer=True, nlp_p=0.0, nlp_digit=False, nlp_op=False):
    """Randomly generate a math expression as a string."""
    if outer:
        if random.random() < nlp_p:
            if random.random() < 0.5:
                nlp_digit = True
            else:
                nlp_op = True

    if max_ops == 0:
        num = random_number(max_digits)
        return f"{num}", num, rand_num2words(num, p=float(nlp_digit))
        
    # assign ops
    left_ops, right_ops = random_split(max_ops - 1)
    valid = False
    trial_count = 0
    desc = None
    while not valid and trial_count < 1000:
        try:
            op = random.choice(OPS)
            left, _, left_desc = generate_random_expression(max_digits, left_ops, outer=False, nlp_digit=nlp_digit, nlp_op=nlp_op)
            right, _, right_desc = generate_random_expression(max_digits, right_ops, outer=False, nlp_digit=nlp_digit, nlp_op=nlp_op)
            if random.random() < 0.3:
                left = f"({left})" if not left.isdigit() else left
                left_desc = f"({left_desc})" if not left.isdigit() else left_desc
            if random.random() < 0.3:
                right = f"({right})" if not right.isdigit() else right
                right_desc = f"({right_desc})" if not right.isdigit() else right_desc
            expr = f"{left} {op} {right}"
            results = eval(expr)
            assert results < 1000000000
            assert round(results, 3) == results
            trial_count += 1
        except Exception as e:
            # print(e)
            # print(expr, "failed to evaluate, retrying...")
            continue
        valid = True
        if outer:
            desc = describe(left_desc, op, right_desc, float(nlp_op))
        else:
            desc = simple_describe(left_desc, op, right_desc, float(nlp_op))

    if int(results) != float(results):
        results = f"{results:.3f}"
    else:
        results = int(results)
    return expr, results, desc

def gen_with_factor(deg_min=1, deg_max=5, coeff_range=(-5,5), r_range=(1,9), round_to=1):
    """Generate n polynomials p(x) = (x - r) * s(x) with integer coeffs and r>0."""
    from sympy import symbols, Poly, expand
    x = symbols('x')
    deg_s = random.randint(deg_min, deg_max)           # degree of s(x)
    # ensure leading coeff of s(x) ≠ 0
    while True:
        coeffs = [round(random.uniform(*coeff_range),round_to) for _ in range(deg_s+1)]
        if coeffs[-1] != 0:
            break
    r = random.randint(*r_range) + random.random() # solution
    r = round(r, round_to + 1)
    # generate a random float in (0, 1)
    
    s = sum(coeffs[i] * x**i for i in range(deg_s+1))
    p = expand((x - r) * s)
    return Poly(p, x), r

if __name__ == "__main__":
    # Example usage
    a, b, c = generate_random_expression(max_digits=3, max_ops=5, nlp_p=0.5)
    print(f"{a} = {b}, Description: {c}")