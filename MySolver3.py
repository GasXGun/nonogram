import sys
import time
import re
from copy import deepcopy

# 增加遞歸深度
sys.setrecursionlimit(5000)

# 常量設定
UNKNOWN = -1
EMPTY = 0
FILLED = 1

# 給每題的預算時間 (秒)
MAX_TIME_PER_PUZZLE = 15.0

# ======================================================
# 輔助函數
# ======================================================
def is_valid_hint(hints, N=25):
    """檢查一組 hint 是否在 N 格內合法"""
    if not hints: return True 
    min_len = sum(hints) + len(hints) - 1
    return min_len <= N

# ======================================================
# 智慧解析器 (Smart Parser v10.0 - 頁碼狙擊版)
# ======================================================
def parse_input_file(filename):
    """
    讀取並解析輸入檔。
    特色：針對 Nonogram pdf 轉檔常見的頁碼雜訊 (單獨的數字行) 進行智慧過濾。
    """
    puzzles = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
    except FileNotFoundError:
        print("Input file not found.")
        return {}

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        
        # 尋找題號
        if '$' in line:
            pid_match = re.search(r'\$(\d+)', line)
            if not pid_match:
                idx += 1
                continue
            
            pid = f"${pid_match.group(1)}"
            idx += 1
            
            raw_hints_buffer = []
            
            # 讀取直到下一題或檔案結束
            while idx < len(lines):
                curr_line = lines[idx]
                if '$' in curr_line: break
                
                idx += 1
                if not curr_line: continue

                # 清除 source 標籤
                # 修正後的 regex，避免語法錯誤
                clean_line = re.sub(r'\\', '', curr_line)
                clean_line = re.sub(r'source:.*', '', clean_line) 
                
                # 解析數字
                clean_line = clean_line.split('#')[0]
                numbers = re.findall(r'\d+', clean_line)
                hints = [int(x) for x in numbers]
                
                if not hints: continue

                if is_valid_hint(hints, 25):
                    raw_hints_buffer.append(hints)

            # --- 智慧過濾邏輯 ---
            # 如果行數正確，直接使用
            if len(raw_hints_buffer) == 50:
                col_hints = raw_hints_buffer[:25]
                row_hints = raw_hints_buffer[25:]
                puzzles[pid] = (col_hints, row_hints)
            
            # 如果行數過多，嘗試過濾「疑似頁碼」的行
            elif len(raw_hints_buffer) > 50:
                print(f"[{pid}] Found {len(raw_hints_buffer)} lines (expected 50). Filtering noise...")
                
                # 策略 1: 移除所有「長度為 1」的行 (針對頁碼如 [1], [7], [8])
                # 注意：這假設合法的 25x25 提示通常不會只有一個數字，或者如果有，刪除後剛好湊滿 50 才是正確的。
                filtered_hints = [h for h in raw_hints_buffer if len(h) > 1]
                
                if len(filtered_hints) == 50:
                    print(f"[{pid}] Success! Filtered out single-number lines.")
                    col_hints = filtered_hints[:25]
                    row_hints = filtered_hints[25:]
                    puzzles[pid] = (col_hints, row_hints)
                else:
                    # 策略 2: 如果策略 1 失敗 (例如刪太多了)，退回到「保留最後 50 行」
                    # 這適用於雜訊都在前面的情況
                    print(f"[{pid}] Filter strategy 1 failed (got {len(filtered_hints)}). Fallback to keeping last 50.")
                    # 再次檢查開頭是否為雜訊 (如 $4 的情況)
                    if len(raw_hints_buffer) == 51 and len(raw_hints_buffer[0]) == 1:
                         final_hints = raw_hints_buffer[1:]
                    else:
                         final_hints = raw_hints_buffer[-50:]
                         
                    col_hints = final_hints[:25]
                    row_hints = final_hints[25:]
                    puzzles[pid] = (col_hints, row_hints)
            
            else:
                print(f"Error: Puzzle {pid} only has {len(raw_hints_buffer)} valid lines. Skipping.")
        else:
            idx += 1
            
    return puzzles

# ======================================================
# Nonogram Solver Class (核心演算法保持不變)
# ======================================================
class NonogramSolver:
    def __init__(self, col_hints, row_hints, pid="Unknown"):
        self.N = 25
        self.col_hints = col_hints
        self.row_hints = row_hints
        self.pid = pid
        self.grid = [[UNKNOWN] * self.N for _ in range(self.N)]
        self.start_time = 0

    def solve(self):
        self.start_time = time.time()
        if not self._propagate(): return None 
        if self._is_solved(): return self.grid
        if self._backtrack(): return self.grid
        return None

    def _is_solved(self):
        for r in range(self.N):
            for c in range(self.N):
                if self.grid[r][c] == UNKNOWN: return False
        return True

    def _propagate(self):
        changed = True
        while changed:
            if time.time() - self.start_time > MAX_TIME_PER_PUZZLE: return True
            changed = False
            
            for r in range(self.N):
                new_line = self._solve_line_complete(self.grid[r], self.row_hints[r])
                if new_line is None: return False
                if new_line != self.grid[r]:
                    self.grid[r] = new_line
                    changed = True
            
            for c in range(self.N):
                current_col = [self.grid[r][c] for r in range(self.N)]
                new_col = self._solve_line_complete(current_col, self.col_hints[c])
                if new_col is None: return False
                if new_col != current_col:
                    for r in range(self.N):
                        self.grid[r][c] = new_col[r]
                    changed = True
        return True

    def _solve_line_complete(self, line, hints):
        N = self.N
        if not hints:
            for x in line:
                if x == FILLED: return None
            return [EMPTY] * N
        min_len = sum(hints) + len(hints) - 1
        if min_len > N: return None

        memo = {}
        can_be_0 = [False] * N
        can_be_1 = [False] * N
        found_any_solution = [False] 

        def recursive_search(hint_idx, start_pos):
            state = (hint_idx, start_pos)
            if state in memo: return memo[state]
            
            if hint_idx == len(hints):
                for k in range(start_pos, N):
                    if line[k] == FILLED:
                        memo[state] = False
                        return False
                found_any_solution[0] = True
                for k in range(start_pos, N): can_be_0[k] = True
                return True

            remaining_len = sum(hints[hint_idx:]) + (len(hints) - 1 - hint_idx)
            if start_pos + remaining_len > N:
                memo[state] = False
                return False

            current_block_len = hints[hint_idx]
            limit = N - remaining_len + 1
            has_valid_path = False
            
            for pos in range(start_pos, limit):
                gap_valid = True
                for k in range(start_pos, pos):
                    if line[k] == FILLED:
                        gap_valid = False
                        break
                if not gap_valid: break 

                block_valid = True
                for k in range(pos, pos + current_block_len):
                    if line[k] == EMPTY:
                        block_valid = False
                        break
                
                if block_valid and (pos + current_block_len < N):
                    if line[pos + current_block_len] == FILLED:
                        block_valid = False
                
                if block_valid:
                    if recursive_search(hint_idx + 1, pos + current_block_len + 1):
                        has_valid_path = True
                        for k in range(start_pos, pos): can_be_0[k] = True
                        for k in range(pos, pos + current_block_len): can_be_1[k] = True
                        if pos + current_block_len < N: can_be_0[pos + current_block_len] = True

            memo[state] = has_valid_path
            return has_valid_path

        recursive_search(0, 0)
        if not found_any_solution[0]: return None

        new_line = list(line)
        for i in range(N):
            if can_be_1[i] and not can_be_0[i]: new_line[i] = FILLED
            elif can_be_0[i] and not can_be_1[i]: new_line[i] = EMPTY
        return new_line

    def _backtrack(self):
        best_r, best_c = -1, -1
        for r in range(self.N):
            for c in range(self.N):
                if self.grid[r][c] == UNKNOWN:
                    best_r, best_c = r, c
                    break
            if best_r != -1: break
        
        if best_r == -1: return True

        saved_grid = deepcopy(self.grid)
        self.grid[best_r][best_c] = FILLED
        if self._propagate():
            if self._backtrack(): return True
        
        self.grid = saved_grid
        self.grid[best_r][best_c] = EMPTY
        if self._propagate():
            if self._backtrack(): return True
            
        return False

# ======================================================
# 主程式
# ======================================================
def main():
    input_file = 'test_input.txt' 
    output_file = 'result_py2.txt'
    
    print(f"Parsing {input_file}...")
    all_puzzles = parse_input_file(input_file)
    print(f"Loaded {len(all_puzzles)} puzzles.")

    with open(output_file, 'w') as f_out:
        for pid, (col_hints, row_hints) in all_puzzles.items():
            print(f"Solving {pid}...")
            
            solver = NonogramSolver(col_hints, row_hints, pid)
            sol = solver.solve()
            
            f_out.write(f"{pid}\n")
            if sol:
                print(f"-> Solved {pid}!")
                for row in sol:
                    # 使用雙空格，這是您通過 8 題時的設定
                    f_out.write("  ".join(["1" if x==1 else "0" for x in row]) + "\n")
            else:
                print(f"-> {pid} Unsolvable.")
                f_out.write("Unsolvable\n")
            
            f_out.flush()

if __name__ == "__main__":
    main()