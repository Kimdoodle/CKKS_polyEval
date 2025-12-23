# print.py
from complexity import Complexity

def print_poly(poly: list[float], title: str="다항식: ") -> None:
    print(f"{title}", end='')
    for i, coeff in enumerate(poly):
        if coeff != 0:
            if coeff < 0:
                mark = ' -'
            elif coeff > 0:
                mark = ' +'
            if coeff.is_integer():
                print(f"{mark}{int(coeff)}(x^{i})", end='')
            else:
                print(f"{mark}{coeff}(x^{i})", end='')
                
def print_poly_type(poly: list[float], title: str="타입: ") -> None:
    type_codes = []
    for coeff in poly:
        if coeff == 0:
            type_codes.append('0')
        elif coeff.is_integer():
            type_codes.append('I')
        else:
            type_codes.append('F')
            
    if title:
        print(title, end='')
        
    print(f"({', '.join(type_codes)})")

def print_poly_sep(i: int, poly_p: list[float], poly_q: list[float]) -> None:
    # 다항식 분해
    print("분해식: ", end='')
    print(f"(x^{i})", end="")
    print("{", end='')
    print_poly(poly_p, "")
    print("}", end='')
    print_poly(poly_q, "")
    

def print_step(poly: list[float], i: int, poly_p: list[float], poly_q: list[float],
               comp_i: Complexity, comp_p: Complexity, comp_q: Complexity, comp_piq: Complexity,
               made_powers: set[int], mp: set[int]):
    print('#'*20)
    print_poly(poly)
    print()
    print_poly_sep(i, poly_p, poly_q)
    print()
    print(f"복잡도: ")
    print(f"Depth:\t{comp_i.depth}\t{comp_p.depth}\t{comp_q.depth}\t=>\t{comp_piq.depth}")
    print(f"CMult:\t{comp_i.cmult}\t{comp_p.cmult}\t{comp_q.cmult}\t=>\t{comp_piq.cmult}")
    print(f"PMult:\t{comp_i.pmult}\t{comp_p.pmult}\t{comp_q.pmult}\t=>\t{comp_piq.pmult}")
    print(f"Add:\t{comp_i.add}\t{comp_p.add}\t{comp_q.add}\t=>\t{comp_piq.add}")
    print(f"생성 차수: {made_powers} => {mp}")
    
