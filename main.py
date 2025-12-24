# main.py
from algorithm import calculate
from contextlib import redirect_stdout
from print import print_poly, print_poly_type
from util import make_all_polys
from polynomial import Poly

if __name__ == '__main__':
    made_powers: set[int] = {0, 1}
    poly = Poly([8, 2.6, 1.2, 1])
    res = calculate(poly, made_powers)
    res[1].print_params()

    # filename = "deg.txt"
    
    # with open(filename, 'w', encoding='utf-8') as f:
    #     with redirect_stdout(f):
    #         for deg in range(2, 6):
    #             print('#' * 20)
    #             print(f"Degree: {deg}")
    #             print('#' * 20)

    #             polys = make_all_polys(deg)
    #             for coeff in polys:
    #                 poly = Poly(coeff)
    #                 poly.print("poly")
    #                 print()
    #                 poly.print("type")
    #                 i, result, _ = calculate(poly, made_powers)
    #                 result.print_params()
    #                 print('-'*20)
