# algorithm.py
from complexity import Complexity, attach, compare
from math import log2, ceil, sqrt
from print import print_step
from collections import deque
from heapq import heappush, heappop

def calculate(poly: list[float], made_powers: set[int]) -> tuple[int, Complexity, set[int]]:
    max_deg = len(poly) - 1
    # =============================
    '''
    0. 분해가 필요없는 기초다항식 처리
        빈 다항식의 경우 복잡도 0,0,0,0을 반환.
    '''
    ## 0-1) 빈 다항식 or 0차식 처리
    if max_deg <= 0:
        return 0, Complexity(), made_powers
    
    ## 0-2) 1차식 처리
    elif max_deg == 1:
        comp_res = Complexity()
        comp_res.depth = 0 if poly[-1].is_integer() else 1
        comp_res.cmult = 0
        comp_res.pmult = 0 if poly[-1].is_integer() else 1
        comp_res.add = 1 if poly[0] != 0 else 0
        return 0, comp_res, made_powers
    
    # ==============================
    '''
    1. 분해하지 않는 경우
        x^0, x^1, ..., x^n 항을 모두 따로 계산할 때의 복잡도 측정.
        depth:  계수가 정수이면 log2(n)이며, 소수일 경우 log2(n+1).
        cmult:  x^0, x^1, ..., x^n을 각각 만들기 위한 최소 비용 알고리즘을 통해 측정.
        pmult:  (0, 정수)가 아닌 계수의 개수
        add:    덧셈 횟수(= 계수가 0이 아닌 총 항의 개수)
    '''
    comp_temp = Complexity()
    
    ## 1-1) depth
    if poly[max_deg].is_integer():
        comp_temp.depth = ceil(log2(max_deg))
    else:
        comp_temp.depth = ceil(log2(max_deg+1))
    
    ## 1-2) cmult
    final_powers, ops, add_count = build_addition_plan_for_poly(poly, made_powers)
    comp_temp.cmult = add_count
    
    ## 1-3) pmult
    for i in range(1, max_deg+1):
        if poly[i].is_integer():
            pass
        else:    
            comp_temp.pmult += 1

    
    ## 1-4) add
    rp = {i for i, c in enumerate(poly) if c != 0}
    comp_temp.add = len(rp) - 1
    
    results = [(0, comp_temp, final_powers)]
    
    '''
    2. f(x) = (x^i)*p(x) + q(x)로 분해하는 경우
        A. (x^i)의 계산복잡도 측정(made_powers 업데이트)
            TODO ax^i로 분해할 경우의 계산복잡도 및 made_powers 고려
        B. p(x), q(x) 중 최고차수가 가장 낮은 것부터 복잡도 측정.
            이 때 (x^i)를 구성할 수 있는 made_powers 각각에 대하여 측정.
        C. 모든 항의 복잡도 측정 후 결합
            (x^i) * p(x)의 경우 곱셈 -> depth(큰쪽) + 1, cmult(합) + 1
            (x^i)p(x) + q(x)의 경우 덧셈 -> add + 1        
    '''
    for i in range(max_deg-1, 0, -1):
        mp = made_powers.copy()
        
        # x^i 계산에 필요한 경우의 수 판별
        xi_routes = cal_xi_routes(i, mp)
        
        # x^i 기준으로 분할 - i는 (n-1) 부터 sqrt(n) 
        poly_p = trim(poly[i:])
        poly_q = trim(poly[:i])
        deg_p, deg_q = len(poly_p), len(poly_q)
        
        for depth, mp2, ops_list in xi_routes:
            add_count = len(ops_list)
            # x^i의 계산복잡도 정리
            '''
            TODO ax^i를 계산하는 경우?
            '''
            comp_i = Complexity()
            comp_i.insert_value(depth, add_count, 0, 0)
            
            # 차수가 작은 다항식부터 연산
            if deg_p < deg_q:
                _, comp_p, mp3 = calculate(poly_p, mp2)
                _, comp_q, mp4 = calculate(poly_q, mp3)
            else:
                _, comp_q, mp3 = calculate(poly_q, mp2)
                _, comp_p, mp4 = calculate(poly_p, mp3)
                
            # 계산복잡도 결합
            comp_pi = attach(comp_i, comp_p, 'x')
            if poly_q != []:
                comp_piq = attach(comp_pi, comp_q, '+')
            else:
                comp_piq = comp_pi
                
            results.append((i, comp_piq, mp))
    
    '''
    3. 최적의 결과 비교
        모든 분해식 결과는 (x^i 차수, 총 계산복잡도, 만들어진 x^i 차수들) tuple로 구성.
        총 계산복잡도 기준으로 depth, cmult, pmult, add 순으로 가장 낮은 값을 검색.
    '''
    best = results[0]
    for c in results[1:]:
        if compare(best[1], c[1]) == 2:
            best = c    
    
    
    return best
        
# 다항식 변수에서 최고차항이 0이 아니도록 다듬는 함수
def trim(poly: list[float]) -> list[float]:
    while poly and poly[-1] == 0:
        poly.pop()
    return poly

# 다항식에서 0이 아닌 값의 index를 반환하는 함수
def required_powers(poly: list[float]) -> set[int]:
    return {i for i, c in enumerate(poly) if c != 0}


# x^i를 구성하기 위한 모든 경우의 수를 탐색.
# BFS방식으로 구현.
def cal_xi_routes(target_i: int, made_powers: set[int]) -> list[tuple[int, set[int], list]]:
    """
    x^i를 구성할 수 있는 최적 경로 탐색 (BFS)
    Returns: list of (depth, new_set, operations_list)
    """
    if target_i in made_powers:
        return [(0, made_powers, [])]

    # {숫자: 깊이} (기존 숫자는 깊이 0으로 가정)
    initial_depths = {num: 0 for num in made_powers}
    queue = deque([(initial_depths, [])])
    
    visited_states = set()
    visited_states.add(tuple(sorted(initial_depths.items())))
    
    candidates = [] 
    min_ops_len = float('inf') 
    
    while queue:
        curr_depths, history = queue.popleft()
        
        # 가지치기
        if len(history) > min_ops_len:
            continue
            
        curr_elements = list(curr_depths.keys())
        
        if target_i in curr_depths:
            target_depth = curr_depths[target_i]
            current_len = len(history)
            
            if current_len <= min_ops_len:
                min_ops_len = current_len
                candidates.append((target_depth, set(curr_elements), history))
            continue
            
        next_moves = []
        n = len(curr_elements)
        for j in range(n):
            for k in range(j, n):
                val_a, val_b = curr_elements[j], curr_elements[k]
                new_val = val_a + val_b
                
                if new_val <= target_i and new_val not in curr_depths:
                    new_depth = max(curr_depths[val_a], curr_depths[val_b]) + 1
                    next_moves.append((new_val, new_depth, val_a, val_b))
        
        for val, d, a, b in next_moves:
            new_depths = curr_depths.copy()
            new_depths[val] = d
            state_key = tuple(sorted(new_depths.items()))
            
            if state_key not in visited_states:
                visited_states.add(state_key)
                new_hist = history + [(a, b, val)]
                queue.append((new_depths, new_hist))
    
    if not candidates:
        return []

    # 필터링: 1. Depth 최소, 2. Count 최소
    min_depth = min(c[0] for c in candidates)
    depth_filtered = [c for c in candidates if c[0] == min_depth]
    
    min_count = min(len(c[2]) for c in depth_filtered)
    final_results = [c for c in depth_filtered if len(c[2]) == min_count]
    
    return final_results

def build_addition_plan_for_poly(poly: list, made_powers: set[int]):
    """
    다항식 계산을 위한 최적 덧셈 사슬 계획 수립 (재귀적 탐색)
    """
    # 0, 1차항은 기본 제공으로 간주하고 그 이상 차수만 계산
    required = sorted([i for i, c in enumerate(poly) if c != 0 and i > 1])
    
    # 초기 집합
    base_powers = {0, 1} | set(made_powers)
    
    if not required:
        return base_powers, [], 0
    
    # 전역 최적해 저장용
    global_best = {
        'ops': None,
        'count': float('inf')
    }
    
    def recursive_solver(target_idx, current_set, accumulated_ops):
        # Base Case: 모든 타겟 해결
        if target_idx == len(required):
            if len(accumulated_ops) < global_best['count']:
                global_best['count'] = len(accumulated_ops)
                global_best['ops'] = accumulated_ops
            return

        # Pruning: 이미 최적해보다 나빠졌으면 중단
        if len(accumulated_ops) >= global_best['count']:
            return

        target = required[target_idx]
        
        # 이미 있다면 다음으로
        if target in current_set:
            recursive_solver(target_idx + 1, current_set, accumulated_ops)
            return

        # BFS를 통해 현재 상태에서 target을 만드는 후보 경로들 탐색
        candidates = cal_xi_routes(target, current_set)
        
        # Branching
        for _, new_set, ops in candidates:
            recursive_solver(target_idx + 1, new_set, accumulated_ops + ops)

    # 재귀 실행
    recursive_solver(0, base_powers, [])
    
    final_ops = global_best['ops'] if global_best['ops'] is not None else []
    final_count = global_best['count'] if global_best['count'] != float('inf') else 0
    
    # 연산 순서대로 집합 복원
    final_powers = set(base_powers)
    for a, b, c in final_ops:
        final_powers.add(c)
        
    return final_powers, final_ops, final_count

# def build_addition_plan_for_poly(poly, made_powers):
#     """
#     poly: [a0, a1, a2, ...]
#     made_powers: 이미 만들어져 있다고 가정하는 차수들의 set[int]
#                  (이들을 만드는 데 필요한 덧셈은 비용 0으로 간주)

#     반환:
#         final_powers : 최종적으로 만들어진 지수 집합
#         operations   : [(a, b, c), ...]  # c 기준 오름차순 정렬
#         add_count    : 총 덧셈 횟수 = len(operations)
#     """
#     required = required_powers(poly)
#     if not required:
#         final_powers = {0, 1} | set(made_powers)
#         return final_powers, [], 0

#     max_power = max(required)

#     # 초기 power: 0, 1, 그리고 이미 만들어져 있는 made_powers
#     base_powers = {0, 1} | set(made_powers)
#     # 계산 대상이 아닌 너무 큰 차수는 버림
#     base_powers = {p for p in base_powers if p <= max_power}

#     start = frozenset(base_powers)
#     target = required

#     # 비용 = (연산 횟수, |a-b| 총합)
#     INF = (10**9, 10**9)
#     dist = {start: (0, 0)}
#     parent = {start: None}
#     op_info = {}  # 상태 → (a, b, c)

#     pq = []
#     heappush(pq, (0, 0, start))  # (steps, imbalance_sum, state)

#     goal_state = None

#     while pq:
#         steps, imb, S = heappop(pq)

#         if (steps, imb) != dist.get(S, INF):
#             continue

#         # 이미 필요한 차수들이 전부 S 안에 있으면 종료
#         if target.issubset(S):
#             goal_state = S
#             break

#         lst = sorted(S)
#         n = len(lst)

#         for i in range(n):
#             a = lst[i]
#             for j in range(i, n):
#                 b = lst[j]
#                 c = a + b

#                 if c > max_power:
#                     continue
#                 if c in S:
#                     continue

#                 newS = frozenset(set(S) | {c})

#                 new_steps = steps + 1
#                 new_imb = imb + abs(a - b)
#                 new_cost = (new_steps, new_imb)

#                 if dist.get(newS, INF) > new_cost:
#                     dist[newS] = new_cost
#                     parent[newS] = S
#                     op_info[newS] = (a, b, c)
#                     heappush(pq, (new_steps, new_imb, newS))

#     if goal_state is None:
#         return set(start), [], 0

#     # 연산 경로 복원
#     ops = []
#     S = goal_state
#     while parent[S] is not None:
#         a, b, c = op_info[S]
#         ops.append((a, b, c))
#         S = parent[S]
#     ops.reverse()

#     # c 기준 오름차순 정렬
#     ops.sort(key=lambda x: x[2])

#     final_powers = set(goal_state)
#     add_count = len(ops)

#     return final_powers, ops, add_count
