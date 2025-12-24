import sys
import time
import re
import multiprocessing
import os
from copy import deepcopy

# 實戰優化：提升遞歸深度與記憶體利用
sys.setrecursionlimit(200000)

# 常量
UNKNOWN = -1
EMPTY = 0
FILLED = 1

# 比賽總時限 (30分鐘)
TOTAL_TIME_LIMIT = 1800
OUTPUT_FILE = 'result.txt'

# ======================================================
# 核心解題類別 (Fast Array-Based DP)
# ======================================================
class NonogramSolver:
    def __init__(self, col_hints, row_hints, pid="Unknown", timeout=5.0):
        self.N = 25
        self.col_hints = col_hints
        self.row_hints = row_hints
        self.pid = pid
        self.timeout = timeout
        self.grid = [[UNKNOWN] * self.N for _ in range(self.N)]
        self.start_time = 0

    def solve(self):
        self.start_time = time.time()
        if not self._propagate_queue(): return None 
        if self._is_solved(): return self.grid
        if self._backtrack(): return self.grid
        return None

    def _is_solved(self):
        for r in range(self.N):
            for c in range(self.N):
                if self.grid[r][c] == UNKNOWN: return False
        return True

    def _propagate_queue(self):
        queue = []
        for r in range(self.N): queue.append((0, r))
        for c in range(self.N): queue.append((1, c))
        in_queue = set(queue)

        while queue:
            if time.time() - self.start_time > self.timeout: return True
            q_type, idx = queue.pop(0)
            in_queue.remove((q_type, idx))

            if q_type == 0: # Row
                current = self.grid[idx]
                hints = self.row_hints[idx]
                new_line = self._solve_line_fast(current, hints)
                if new_line is None: return False
                if new_line != current:
                    self.grid[idx] = new_line
                    for c in range(self.N):
                        if current[c] != new_line[c]:
                            if (1, c) not in in_queue:
                                queue.append((1, c))
                                in_queue.add((1, c))
            else: # Col
                current = [self.grid[r][idx] for r in range(self.N)]
                hints = self.col_hints[idx]
                new_line = self._solve_line_fast(current, hints)
                if new_line is None: return False
                if new_line != current:
                    for r in range(self.N):
                        if self.grid[r][idx] != new_line[r]:
                            self.grid[r][idx] = new_line[r]
                            if (0, r) not in in_queue:
                                queue.append((0, r))
                                in_queue.add((0, r))
        return True

    def _solve_line_fast(self, line, hints):
        N = self.N
        K = len(hints)
        if K == 0 or (K == 1 and hints[0] == 0):
            for x in line:
                if x == FILLED: return None
            return [EMPTY] * N
        if sum(hints) + K - 1 > N: return None

        suffix_len = [0] * (K + 1)
        for i in range(K - 1, -1, -1):
            suffix_len[i] = suffix_len[i+1] + hints[i] + (1 if i < K - 1 else 0)

        memo = [[0] * (N + 2) for _ in range(K + 1)]
        pos_black = [False] * N
        pos_white = [False] * N

        def dp(h_idx, pos):
            if memo[h_idx][pos] != 0: return memo[h_idx][pos] == 1
            if h_idx == K:
                for i in range(pos, N):
                    if line[i] == FILLED:
                        memo[h_idx][pos] = 2; return False
                for i in range(pos, N): pos_white[i] = True
                memo[h_idx][pos] = 1; return True

            block_len = hints[h_idx]
            limit = N - suffix_len[h_idx+1] - block_len
            if h_idx < K - 1: limit -= 1
            
            can_solve = False
            for p in range(pos, limit + 1):
                valid_gap = True
                for i in range(pos, p):
                    if line[i] == FILLED: valid_gap = False; break
                if not valid_gap: break
                valid_block = True
                for i in range(p, p + block_len):
                    if line[i] == EMPTY: valid_block = False; break
                if not valid_block: continue
                next_start = p + block_len
                if h_idx < K - 1:
                    if next_start < N and line[next_start] == FILLED: continue
                    next_start += 1
                if dp(h_idx + 1, next_start):
                    can_solve = True
                    for i in range(pos, p): pos_white[i] = True
                    for i in range(p, p + block_len): pos_black[i] = True
                    if h_idx < K - 1 and p + block_len < N: pos_white[p + block_len] = True
            
            memo[h_idx][pos] = 1 if can_solve else 2
            return can_solve

        if not dp(0, 0): return None
        new_line = list(line)
        for i in range(N):
            if pos_black[i] and not pos_white[i]: new_line[i] = FILLED
            elif pos_white[i] and not pos_black[i]: new_line[i] = EMPTY
        return new_line

    def _backtrack(self):
        best_r, best_c = -1, -1
        min_u = 999
        for r in range(self.N):
            if UNKNOWN in self.grid[r]:
                cnt = self.grid[r].count(UNKNOWN)
                if cnt < min_u:
                    min_u = cnt
                    for c in range(self.N):
                        if self.grid[r][c] == UNKNOWN:
                            best_r, best_c = r, c; break
        if best_r == -1: return True
        saved = deepcopy(self.grid)
        self.grid[best_r][best_c] = FILLED
        if self._propagate_queue():
            if self._backtrack(): return True
        self.grid = saved
        self.grid[best_r][best_c] = EMPTY
        if self._propagate_queue():
            if self._backtrack(): return True
        return False

# ======================================================
# 解析器 (實戰嚴格模式：前25 Col, 後25 Row)
# ======================================================
def parse_input_file(filename):
    puzzles = [] 
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except: return []

    raw_blocks = content.split('$')
    for block in raw_blocks:
        block = block.strip()
        if not block: continue
        lines = block.split('\n')
        try:
            pid_match = re.search(r'\d+', lines[0])
            if not pid_match: continue
            pid = f"${pid_match.group()}"
        except: continue
            
        hints_buffer = []
        for l in lines[1:]:
            nums = [int(x) for x in re.findall(r'\d+', l)]
            if nums:
                if len(nums) == 1 and nums[0] == 0: hints_buffer.append([0])
                elif 0 in nums:
                    clean = [n for n in nums if n != 0]
                    hints_buffer.append(clean if clean else [0])
                else: hints_buffer.append(nums)
        
        if len(hints_buffer) >= 50:
            final_hints = hints_buffer[-50:]
            # 實戰鎖定順序：前面25為Column，後面25為Row
            col_hints = final_hints[:25] 
            row_hints = final_hints[25:] 
            puzzles.append((pid, col_hints, row_hints))

    puzzles.sort(key=lambda x: int(x[0].replace('$','')))
    return puzzles

# ======================================================
# Worker
# ======================================================
def solve_wrapper(args):
    pid, col, row, timeout = args
    start_t = time.time()
    solver = NonogramSolver(col, row, pid, timeout=timeout)
    res = solver.solve()
    return (pid, res, time.time() - start_t)

# ======================================================
# 主程式
# ======================================================
def main():
    start_global = time.time()
    input_file = 'Nonogram_Q.txt'
    
    print(f"実戦モード啟動：讀取 {input_file}...")
    all_puzzles = parse_input_file(input_file)
    total_count = len(all_puzzles)
    print(f"載入 {total_count} 題。使用直行 (Col) 優先模式。")
    
    results = {p[0]: (None, 0.0) for p in all_puzzles}
    
    def save_to_disk():
        sorted_pids = sorted(results.keys(), key=lambda x: int(x.replace('$','')))
        with open(OUTPUT_FILE, 'w') as f:
            for pid in sorted_pids:
                sol, _ = results[pid]
                f.write(f"{pid}\n")
                if sol:
                    for row in sol:
                        f.write("  ".join(["1" if x==1 else "0" for x in row]) + "\n")
                else:
                    f.write("Unsolvable\n")

    workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=workers)

    # Phase 1: Blitz (1秒快篩)
    print(f"\n[Phase 1] Blitz Mode (1s)...")
    tasks_p1 = [(p[0], p[1], p[2], 1.0) for p in all_puzzles]
    for pid, sol, dur in pool.imap_unordered(solve_wrapper, tasks_p1):
        if sol: results[pid] = (sol, dur)
    
    print(f"Phase 1 Solved: {sum(1 for v,d in results.values() if v is not None)}/{total_count}")
    save_to_disk()

    # Phase 2: Normal (20秒攻堅)
    unsolved = [pid for pid, (sol, _) in results.items() if sol is None]
    if unsolved:
        print(f"\n[Phase 2] Normal Mode (20s) - 處理剩餘 {len(unsolved)} 題...")
        data_p2 = [p for p in all_puzzles if p[0] in unsolved]
        tasks_p2 = [(p[0], p[1], p[2], 20.0) for p in data_p2]
        for pid, sol, dur in pool.imap_unordered(solve_wrapper, tasks_p2):
            if sol: 
                results[pid] = (sol, dur)
                print(f"  Solved {pid} ({dur:.2f}s)")
        save_to_disk()

    # Phase 3: Final Sprint (剩餘時間全開)
    unsolved = [pid for pid, (sol, _) in results.items() if sol is None]
    time_left = TOTAL_TIME_LIMIT - (time.time() - start_global)
    if unsolved and time_left > 60:
        time_per_puzzle = max(60.0, time_left / len(unsolved))
        print(f"\n[Phase 3] Final Sprint! 剩下 {int(time_left)}s，分配每題 {time_per_puzzle:.1f}s...")
        data_p3 = [p for p in all_puzzles if p[0] in unsolved]
        tasks_p3 = [(p[0], p[1], p[2], time_per_puzzle) for p in data_p3]
        for pid, sol, dur in pool.imap_unordered(solve_wrapper, tasks_p3):
            if sol: 
                results[pid] = (sol, dur)
                print(f"  !!! Solved HARD {pid} ({dur:.2f}s)")
                save_to_disk()

    pool.close()
    pool.join()
    print(f"\nFinal: {sum(1 for v,d in results.values() if v is not None)}/{total_count}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()